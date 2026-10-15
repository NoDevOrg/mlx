// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cublas_utils.h"

#include <cstdlib>
#include <string>
#include <unordered_map>

#include <fmt/format.h>
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace cublas_utils {

cublasLtMatrixLayout_t create_matrix_layout(
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride) {
  cublasLtMatrixLayout_t desc;
  if (transposed) {
    std::swap(rows, cols);
  }
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&desc, type, rows, cols, ld));
  if (batch_count > 1) {
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(int32_t)));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_stride,
        sizeof(int64_t)));
  }
  return desc;
}

} // namespace cublas_utils

namespace {

auto& cublas_handles_cache() {
  struct CublasHandles {
    ~CublasHandles() {
      if (handle) {
        CHECK_CUBLAS_ERROR(cublasLtDestroy(handle));
        CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceDestroy(pref));
      }
    }
    cublasLtHandle_t handle{nullptr};
    cublasLtMatmulPreference_t pref{nullptr};
  };
  static thread_local std::vector<CublasHandles> cache(gpu::device_count());
  return cache;
}

auto get_cublas_handles(cu::Device& device) {
  auto& storage = cublas_handles_cache().at(device.cuda_device());
  if (!storage.handle) {
    // Create cublasLt handle.
    device.make_current();
    CHECK_CUBLAS_ERROR(cublasLtCreate(&storage.handle));
    // The recommended cublas workspace size is 4 MiB for pre-Hopper and 32
    // MiB for Hopper+:
    // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
    uint64_t MiB = 1024 * 1024;
    uint64_t workspace_size =
        device.compute_capability_major() >= 9 ? 32 * MiB : 4 * MiB;
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&storage.pref));
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceSetAttribute(
        storage.pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(uint64_t)));
  }
  return std::make_tuple(storage.handle, storage.pref);
}

} // namespace

void check_cublas_error(const char* name, cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    // TODO: Use cublasGetStatusString when it is widely available.
    throw std::runtime_error(
        fmt::format("{} failed with code: {}.", name, static_cast<int>(err)));
  }
}

void init_cublas_handles_cache() {
  cublas_handles_cache();
}

CublasMatmulBase::~CublasMatmulBase() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
}

void CublasMatmulBase::init_base(
    cu::Device& device,
    cudaDataType_t scale_type,
    cublasComputeType_t compute_type,
    cudaDataType_t data_type,
    cudaDataType_t output_type,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride) {
  M_ = a_rows;
  N_ = b_cols;
  scale_type_ = scale_type;
  compute_type_ = compute_type;
  data_type_ = data_type;
  tune_key_ = fmt::format(
      "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
      static_cast<int>(compute_type),
      static_cast<int>(data_type),
      static_cast<int>(output_type),
      a_transposed,
      a_rows,
      a_cols,
      lda,
      b_transposed,
      b_rows,
      b_cols,
      ldb,
      batch_count,
      a_batch_stride,
      b_batch_stride,
      static_cast<int>(scale_type));
  size_t out_elem = output_type == CUDA_R_64F ? 8
      : (output_type == CUDA_R_32F || output_type == CUDA_C_32F) ? 4
                                                                   : 2;
  out_bytes_ = b_cols * a_rows * std::max<int32_t>(batch_count, 1) * out_elem;
  std::tie(handle_, pref_) = get_cublas_handles(device);
  heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

  CHECK_CUBLAS_ERROR(
      cublasLtMatmulDescCreate(&matmul_desc_, compute_type, scale_type));

  // In cublasLt matrices use column-major layout, while it is possible to use
  // the CUBLASLT_ORDER_ROW option to switch to row-major layout, the bias
  // epilogue does not work with the option. So instead we swap A and B to make
  // cublasLt return the row-major result, which works because:
  // - the data of a matrix in row-major layout is identical to its transpose in
  //   column-major layout
  // - C^T = (A @ B)^T = B^T @ A^T
  cublasOperation_t a_op = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &a_op,
      sizeof(cublasOperation_t)));
  cublasOperation_t b_op = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &b_op,
      sizeof(cublasOperation_t)));

  a_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      b_cols,
      b_rows,
      b_transposed,
      ldb,
      batch_count,
      b_batch_stride);
  b_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      a_cols,
      a_rows,
      a_transposed,
      lda,
      batch_count,
      a_batch_stride);
  out_desc_ = cublas_utils::create_matrix_layout(
      output_type, b_cols, a_rows, false, b_cols, batch_count, b_cols * a_rows);
}

void CublasMatmulBase::execute_matmul(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr) {
  if (heuristic_.state != CUBLAS_STATUS_SUCCESS) {
    // Plain f32 (no TF32): cuBLASLt's first candidate can be far from the
    // best (on GB10 it is ~6 TFLOPS where another algorithm reaches ~12.5),
    // so time the candidates once per shape and keep the fastest. The result
    // is the same f32 math in a different order. Off with
    // MLX_CUBLASLT_AUTOTUNE=0; "verbose" prints each choice.
    static const char* tune_env = std::getenv("MLX_CUBLASLT_AUTOTUNE");
    bool tune = compute_type_ == CUBLAS_COMPUTE_32F && data_type_ == CUDA_R_32F &&
        !(tune_env && std::string(tune_env) == "0");
    std::string key = tune_key_ + (c ? "|c" : "") + (has_bias_ ? "|bias" : "");
    static std::unordered_map<std::string, cublasLtMatmulHeuristicResult_t>
        tuned;
    if (tune && tuned.count(key)) {
      heuristic_ = tuned[key];
    } else {
      constexpr int kCandidates = 8;
      cublasLtMatmulHeuristicResult_t results[kCandidates];
      int ret = 0;
      CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
          handle_,
          matmul_desc_,
          a_desc_,
          b_desc_,
          c ? c_desc_ : out_desc_,
          out_desc_,
          pref_,
          tune ? kCandidates : 1,
          results,
          &ret));
      if (ret == 0) {
        throw std::runtime_error("Can not find algorithm for matmul.");
      }
      heuristic_ = results[0];
      if (tune && ret > 1) {
        heuristic_ = autotune(a, b, c, alpha_ptr, beta_ptr, results, ret);
        if (tune_env && std::string(tune_env) == "verbose") {
          int id0 = 0, id = 0;
          size_t sz;
          cublasLtMatmulAlgoConfigGetAttribute(
              &results[0].algo, CUBLASLT_ALGO_CONFIG_ID, &id0, sizeof(id0), &sz);
          cublasLtMatmulAlgoConfigGetAttribute(
              &heuristic_.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &sz);
          fmt::print(stderr, "mlx cublasLt f32 autotune {}: algo {} -> {}\n", key, id0, id);
        }
      }
      if (tune) {
        tuned[key] = heuristic_;
      }
    }
  }

  void* workspace_ptr = allocate_workspace(encoder, heuristic_.workspaceSize);

  // Execute matmul
  auto capture = encoder.capture_context();
  CHECK_CUBLAS_ERROR(cublasLtMatmul(
      handle_,
      matmul_desc_,
      alpha_ptr,
      b, // a and b are swapped for row-major layout
      a_desc_,
      a,
      b_desc_,
      beta_ptr,
      c ? c : out,
      c ? c_desc_ : out_desc_,
      out,
      out_desc_,
      &heuristic_.algo,
      workspace_ptr,
      heuristic_.workspaceSize,
      encoder.stream()));
}

cublasLtMatmulHeuristicResult_t CublasMatmulBase::autotune(
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr,
    const cublasLtMatmulHeuristicResult_t* results,
    int count) {
  // Runs on a private stream before this op's graph capture begins. Inputs
  // are only read (their contents don't matter for timing); the output goes
  // to scratch memory, since the real output buffer may still be in use by
  // work that hasn't run yet.
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  size_t ws_bytes = 0;
  for (int i = 0; i < count; ++i) {
    ws_bytes = std::max(ws_bytes, results[i].workspaceSize);
  }
  void* scratch = nullptr;
  void* ws = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&scratch, out_bytes_));
  if (ws_bytes > 0) {
    CHECK_CUDA_ERROR(cudaMalloc(&ws, ws_bytes));
  }
  cudaEvent_t e0, e1;
  CHECK_CUDA_ERROR(cudaEventCreate(&e0));
  CHECK_CUDA_ERROR(cudaEventCreate(&e1));
  auto run = [&](const cublasLtMatmulHeuristicResult_t& r) {
    return cublasLtMatmul(
        handle_,
        matmul_desc_,
        alpha_ptr,
        b,
        a_desc_,
        a,
        b_desc_,
        beta_ptr,
        c ? c : scratch,
        c ? c_desc_ : out_desc_,
        scratch,
        out_desc_,
        &r.algo,
        ws,
        r.workspaceSize,
        stream);
  };
  float best_ms = 0, first_ms = 0;
  int best = 0;
  for (int i = 0; i < count; ++i) {
    if (results[i].state != CUBLAS_STATUS_SUCCESS ||
        run(results[i]) != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    cudaEventRecord(e0, stream);
    for (int r = 0; r < 3; ++r) {
      run(results[i]);
    }
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    if (i == 0) {
      first_ms = best_ms = ms;
    } else if (ms < best_ms) {
      best_ms = ms;
      best = i;
    }
  }
  cudaStreamSynchronize(stream);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(scratch);
  if (ws) {
    cudaFree(ws);
  }
  cudaStreamDestroy(stream);
  // Keep the default unless the winner is clearly faster, so near-ties can't
  // flip the algorithm (and the rounding) from one run to the next.
  return (first_ms > 0 && best_ms < 0.9f * first_ms) ? results[best]
                                                       : results[0];
}

void CublasMatmulBase::set_bias(
    cu::CommandEncoder& encoder,
    const array& bias) {
  encoder.set_input_array(bias);
  has_bias_ = true;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));
  auto* bias_ptr = gpu_ptr<void>(bias);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
}

} // namespace mlx::core
