// Offline, single-row replay of the IQ3_S Metal boundary on a resident CUDA stage.
// The caller packs the input (residual, bo, inj) and independently checks the output.
#include "block_setup.h"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t row_floats = 4 * 2560 + 2560 + 4;
constexpr int warm_runs = 10;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Keep both handoffs alive until after the captured graph has been destroyed.
struct MappedRow {
    float* host = nullptr;
    float* device = nullptr;

    void allocate() {
        check(cudaHostAlloc(reinterpret_cast<void**>(&host), row_floats * sizeof(float), cudaHostAllocMapped),
              "allocate mapped handoff");
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0), "map handoff");
    }
    ~MappedRow() { if (host) cudaFreeHost(host); }
};

void read_input(const std::string& file, float* row) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(row), row_floats * sizeof(float)) || stream.peek() != EOF)
        throw std::runtime_error("expected exactly one 51,216-byte handoff in " + file);
    if (!std::all_of(row, row + row_floats, [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("input contains a nonfinite value");
}

void write_output(const std::string& file, const std::vector<float>& first_run) {
    FILE* stream = std::fopen(file.c_str(), "wbx");
    if (!stream) throw std::runtime_error("cannot create new output file: " + file);
    const size_t count = std::fwrite(first_run.data(), sizeof(float), first_run.size(), stream);
    const int status = std::fclose(stream);
    if (count != first_run.size() || status != 0) {
        std::remove(file.c_str());
        throw std::runtime_error("failed to write CUDA handoff");
    }
}

// Match generate.cpp's --native arithmetic for the interior stage. There is no
// embedding, PLE, or head in [4,10). Configure the QSA indexer before session_init.
void configure_native_stage() {
    using namespace strata;
    kernels::gr_set_native_mmvf(true);
    core::layer_set_fast_attn(true);
    core::layer_set_fused_gdn(true);
    core::layer_set_fast_select(true);
    core::layer_set_fused_gr(true);
    core::layer_set_native_bf16(true);
    kernels::shared_expert_set_native_bf16(true);
    kernels::native_moe_combine_set_enabled(true);
    kernels::native_gdn_set_enabled(true);
    kernels::native_router_set_enabled(true);
    kernels::native_qsa_set_enabled(true);
    kernels::native_qsa_indexer_set_enabled(true);
    kernels::native_rope_set_enabled(true);
}

void report_memory(const char* phase) {
    size_t available = 0, total = 0;
    check(cudaMemGetInfo(&available, &total), "read free VRAM");
    std::cout << "vram_free_bytes_" << phase << '=' << available << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    const std::string usage = "usage: strata_block_probe --pack DIR --shard MODEL.gguf --input INPUT.f32 --output NEW.f32";
    try {
        if (argc != 9) throw std::runtime_error(usage);
        std::string pack, shard, input, output;
        for (int i = 1; i < argc; i += 2) {
            const std::string flag = argv[i];
            if (flag == "--pack" && pack.empty()) pack = argv[i + 1];
            else if (flag == "--shard" && shard.empty()) shard = argv[i + 1];
            else if (flag == "--input" && input.empty()) input = argv[i + 1];
            else if (flag == "--output" && output.empty()) output = argv[i + 1];
            else throw std::runtime_error(usage);
        }
        if (pack.empty() || shard.empty() || input.empty() || output.empty()) throw std::runtime_error(usage);

        check(cudaSetDevice(0), "select CUDA device 0");
        cudaDeviceProp gpu{};
        check(cudaGetDeviceProperties(&gpu, 0), "identify CUDA device 0");
        MappedRow before, after;
        before.allocate();
        after.allocate();
        read_input(input, before.host);

        report_memory("before_setup");
        configure_native_stage();
        BlockSetup setup;
        std::string err;
        if (!setup.init(pack, shard, err)) throw std::runtime_error("stage setup: " + err);
        if (strata::core::Verifier::handoff_floats(setup.geometry) != row_floats)
            throw std::runtime_error("model geometry does not match the IQ3_S handoff");

        strata::core::Verifier stage;
        stage.set_stage(4, 10, before.device, after.device);
        if (!stage.init(setup.weights, setup.geometry, setup.session, setup.hits, nullptr, 2, err))
            throw std::runtime_error("verifier init: " + err);
        if (!stage.ar_live()) throw std::runtime_error("CUDA stage did not select the all-resident expert path");
        report_memory("after_init");

        const int32_t token = 9707;
        int32_t unused_head = -1;
        std::vector<float> first_run;
        double max_repeat_delta = 0;
        std::cout << std::setprecision(9);
        for (int i = 0; i <= warm_runs; ++i) {
            // Each measurement replays the same first position from empty KV and GDN state.
            // Reset, sentinel fill and host-side validation are outside the timed interval.
            strata::core::session_zero(setup.session, setup.geometry, nullptr, nullptr);
            check(cudaGetLastError(), "reset sequence state");
            check(cudaDeviceSynchronize(), "finish sequence reset");
            std::fill(after.host, after.host + row_floats, std::numeric_limits<float>::quiet_NaN());
            const auto start = std::chrono::steady_clock::now();
            if (!stage.run(1, &token, 0, nullptr, nullptr, &unused_head, err))
                throw std::runtime_error("verifier run: " + err);
            // Verifier::run synchronizes the stage stream before returning its handoff.
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            for (size_t j = 0; j < row_floats; ++j) {
                if (!std::isfinite(after.host[j])) throw std::runtime_error("nonfinite or unwritten CUDA handoff");
                if (i > 0)
                    max_repeat_delta = std::max(max_repeat_delta, std::abs(double(after.host[j]) - first_run[j]));
            }
            if (i == 0) {
                first_run.assign(after.host, after.host + row_floats);
                std::cout << "p0_first_graph_ms=" << ms << '\n';
                report_memory("after_first_graph");
            } else {
                std::cout << "p0_warm_" << i << "_ms=" << ms << '\n';
            }
        }
        write_output(output, first_run);
        std::cout << "p0_max_repeat_delta=" << max_repeat_delta << "\ncuda_device=" << gpu.name << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strata_block_probe: " << e.what() << '\n';
        return 1;
    }
}
