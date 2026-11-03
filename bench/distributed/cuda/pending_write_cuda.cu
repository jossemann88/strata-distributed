#include "strata/kernels/native_gr_postops.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
}

void cuda_pending_write(float* state, int width, int channels) {
    int device;
    cudaDeviceProp info{};
    check(cudaGetDevice(&device), "get CUDA device");
    check(cudaGetDeviceProperties(&info, device), "get CUDA device properties");
    std::printf("device=%s ordinal=%d sm=%d%d\n", info.name, device, info.major, info.minor);
    const int residual = width * channels;
    const size_t bytes = (residual + width + channels) * sizeof(float);
    float* gpu = nullptr;
    check(cudaMalloc(&gpu, bytes), "allocate boundary");
    try {
        check(cudaMemcpy(gpu, state, bytes, cudaMemcpyHostToDevice), "upload boundary");
        strata::kernels::native_gr_post(gpu, gpu + residual, gpu + residual + width,
                                        gpu, width, channels, nullptr);
        check(cudaMemcpy(state, gpu, residual * sizeof(float), cudaMemcpyDeviceToHost), "download residual");
    } catch (...) {
        cudaFree(gpu);
        throw;
    }
    check(cudaFree(gpu), "free boundary");
}
