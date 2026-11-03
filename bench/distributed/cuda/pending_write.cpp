// Materialize the pending FFN write at the entry to layer 4, for a single captured row.
// Build on a CUDA development host, from the repository root (RTX 4070):
//   cmake -S bench/distributed/cuda -B build-boundary -DCMAKE_CUDA_ARCHITECTURES=89
//   cmake --build build-boundary --config Release --target pending_write
//   .\build-boundary\Release\pending_write.exe CAPTURE_DIRECTORY [--output NEW.f32] [--metal-reference METAL.f32]
// The optional output is 40960 bytes of R' at the start of layer 4. The layer-10
// block boundary is a different tensor; this diagnostic does not evaluate the block.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void cuda_pending_write(float* state, int width, int channels);

namespace {
constexpr int width = 2560, channels = 4, residual_size = channels * width;
constexpr double tolerance = 2e-6;  // scaled by the input magnitudes, including cancellation

void read_part(const std::filesystem::path& path, float* out, int count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(out), count * sizeof(float)) || in.peek() != EOF)
        throw std::runtime_error("missing or wrong-sized input: " + path.string());
    if (!std::all_of(out, out + count, [](float f) { return std::isfinite(f); }))
        throw std::runtime_error("nonfinite input: " + path.string());
}

void save(const char* path, const float* data) {
    FILE* out = std::fopen(path, "wbx");
    if (!out) throw std::runtime_error("cannot create new output file");
    const auto count = std::fwrite(data, sizeof(float), residual_size, out);
    const int closed = std::fclose(out);
    if (count != residual_size || closed != 0) {
        std::remove(path);
        throw std::runtime_error("failed to write pending residual");
    }
}
}

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    if (argc < 2 || argc > 6 || argc % 2 != 0) {
        std::cerr << "usage: pending_write CAPTURE_DIRECTORY [--output NEW.f32] [--metal-reference METAL.f32]\n";
        return 2;
    }
    try {
        const std::filesystem::path dir(argv[1]);
        const char* output_path = nullptr;
        const char* metal_path = nullptr;
        for (int i = 2; i < argc; i += 2) {
            const std::string flag(argv[i]);
            if (flag == "--output" && !output_path) output_path = argv[i + 1];
            else if (flag == "--metal-reference" && !metal_path) metal_path = argv[i + 1];
            else throw std::runtime_error("duplicate or unknown argument " + flag);
        }
        std::vector<float> metal;
        if (metal_path) {
            metal.resize(residual_size);
            read_part(metal_path, metal.data(), residual_size);
        }
        std::vector<float> state(residual_size + width + channels);
        read_part(dir / "layer04-residual.f32", state.data(), residual_size);
        read_part(dir / "layer04-bo.f32", state.data() + residual_size, width);
        read_part(dir / "layer04-inj.f32", state.data() + residual_size + width, channels);
        const auto input = state;
        cuda_pending_write(state.data(), width, channels);
        double max_error = 0.0, max_scaled = 0.0, max_metal_scaled = 0.0;
        for (int c = 0; c < channels; ++c) {
            const double inject = input[residual_size + width + c];
            const double weight = 2.0 / (1.0 + std::exp(-inject / channels));
            for (int d = 0; d < width; ++d) {
                const int i = c * width + d;
                const double delta = weight * input[residual_size + d];
                const double error = std::abs(double(state[i]) - (double(input[i]) + delta));
                if (!std::isfinite(error)) throw std::runtime_error("nonfinite CUDA result");
                max_error = std::max(max_error, error);
                const double scale = 1.0 + std::abs(input[i]) + std::abs(delta);
                max_scaled = std::max(max_scaled, error / scale);
                if (metal_path)
                    max_metal_scaled = std::max(max_metal_scaled, std::abs(double(state[i]) - metal[i]) / scale);
            }
        }
        std::cout << "backend=cuda component=layer04_pending_write rows=1 floats=" << residual_size
                  << " max_abs_error=" << max_error << " max_scaled_error=" << max_scaled
                  << " scaled_tolerance=" << tolerance << '\n';
        if (max_scaled > tolerance) throw std::runtime_error("CUDA residual differs from CPU reference");
        if (metal_path) {
            std::cout << "metal_max_scaled_error=" << max_metal_scaled << '\n';
            if (max_metal_scaled > tolerance) throw std::runtime_error("CUDA residual differs from Metal reference");
        }
        if (output_path) save(output_path, state.data());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "pending_write: " << e.what() << '\n';
        return 1;
    }
}
