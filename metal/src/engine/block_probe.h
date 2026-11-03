// Stops and host representation for the one-row Qwen3.8 layers [4,10) debug probe.
#pragma once
#include "mtl_rt.h"
#include <cerrno>
#include <cstdio>

namespace strata::metal_engine {

struct BlockProbeStops {
    static constexpr int begin = 4, end = 10;
    int dispatch[2] = {}, inj_bank[2] = {};

    void before_layer(int layer, int next_dispatch, int current_inj) {
        if (layer == begin || layer == end) {
            const int i = layer == begin ? 0 : 1;
            dispatch[i] = next_dispatch;
            inj_bank[i] = current_inj;
        }
    }

    std::vector<int> with_boundaries(std::vector<int> router_cuts) const {
        // One router stop per layer. The first insertion shifts the second by one.
        router_cuts.insert(router_cuts.begin() + begin, dispatch[0]);
        router_cuts.insert(router_cuts.begin() + end + 1, dispatch[1]);
        return router_cuts;
    }

    static int boundary_index(int stop) {
        return stop == begin ? 0 : stop == end + 1 ? 1 : -1;
    }
    static int router_layer(int stop) {
        return stop - (stop > begin) - (stop > end + 1);
    }
};

struct BlockBoundary {
    static constexpr int width = 2560, channels = 4;
    float residual[channels * width], bo[width], inj[channels];

    void capture(const mtl::Buffer& r, const mtl::Buffer& output, const mtl::Buffer& inject) {
        std::memcpy(residual, r.ptr, sizeof(residual));
        std::memcpy(bo, output.ptr, sizeof(bo));
        std::memcpy(inj, inject.ptr, sizeof(inj));
    }
};
static_assert(sizeof(BlockBoundary) == 51216);

inline bool write_block_boundaries(const BlockBoundary* boundary, const std::string& dir, std::string& err) {
    std::vector<std::string> created;
    auto save = [&](const std::string& path, const float* values, size_t count) {
        FILE* f = std::fopen(path.c_str(), "wbx");
        if (!f) return false;
        created.push_back(path);
        const bool written = std::fwrite(values, sizeof(float), count, f) == count;
        return std::fclose(f) == 0 && written;
    };
    for (int i = 0; i < 2; ++i) {
        const auto prefix = dir + (i == 0 ? "/layer04-" : "/layer10-");
        const auto& b = boundary[i];
        if (!save(prefix + "residual.f32", b.residual, BlockBoundary::channels * BlockBoundary::width) || !save(prefix + "bo.f32", b.bo, BlockBoundary::width)
                || !save(prefix + "inj.f32", b.inj, BlockBoundary::channels)) {
            err = "boundary capture: could not create new files in " + dir + ": " + std::strerror(errno);
            for (const auto& path : created) std::remove(path.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace strata::metal_engine
