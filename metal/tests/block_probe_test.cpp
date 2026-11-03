// Synthetic end-to-end check of the 48 router callbacks and both block handoffs.
// An optional existing empty directory receives both fixtures through the engine's writer.
#include "engine/block_probe.h"
#include <cstdio>

using namespace strata::metal_engine;

int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "usage: block_probe_test [empty-directory]\n"); return 2; }
    mtl::Context cx;
    // Each dispatch writes a unique stage tag plus the element's index. As the router
    // waits, the CPU should see the pre-routing stage; each boundary sees the previous
    // layer's final stage. The two banks deliberately differ at layers 4 and 10.
    const int lib = cx.compile({}, R"MSL(
#define NO_GGML_COMMON 1
#include <metal_stdlib>
using namespace metal;
kernel void stamp(device float* r, device float* bo, device float* inj, constant uint& stage,
                  uint i [[thread_position_in_grid]]) {
    uint base = stage * 16384;
    r[i] = float(base + i);
    if (i < 2560) bo[i] = float(base + 10240 + i);
    if (i < 4) inj[i] = float(base + 12800 + i);
}
)MSL");
    auto stamp = cx.pipeline(lib, "stamp");
    auto residual = cx.alloc(40960), bo = cx.alloc(10240);
    mtl::Buffer inj[2] = {cx.alloc(16), cx.alloc(16)};
    BlockProbeStops stops;
    std::vector<mtl::Dispatch> ds;
    std::vector<int> routers;
    int bank = 0;
    for (int l = 0; l < 48; ++l) {
        stops.before_layer(l, (int) ds.size(), bank);
        bank = (l / 6) % 2;
        for (int phase = 0; phase <= 1; ++phase) {
            mtl::Dispatch d;
            d.p = &stamp;
            d.args = {mtl::Arg::B(residual), mtl::Arg::B(bo), mtl::Arg::B(inj[bank]),
                      mtl::Arg::V((uint32_t) (2 * l + phase))};
            d.grid[0] = 10240 / 32;
            ds.push_back(d);
            if (phase == 0) routers.push_back((int) ds.size());
        }
    }
    struct Check {
        const BlockProbeStops& stops;
        mtl::Buffer& r;
        mtl::Buffer& bo;
        mtl::Buffer* inj;
        BlockBoundary snapshots[2]{};
        int routers = 0, boundaries = 0;
        bool ok = true;
    } c{stops, residual, bo, inj};
    cx.run_graph_seg(ds, stops.with_boundaries(routers), [](void* u, int stop) {
        auto& c = *static_cast<Check*>(u);
        const int b = BlockProbeStops::boundary_index(stop);
        if (b >= 0) {
            c.ok &= b == c.boundaries++ && c.routers == (b == 0 ? 4 : 10);
            c.snapshots[b].capture(c.r, c.bo, c.inj[c.stops.inj_bank[b]]);
        } else {
            const int l = BlockProbeStops::router_layer(stop);
            c.ok &= l == c.routers++ && c.r.as<float>()[0] == 2 * l * 16384;
        }
    }, &c);
    for (int b = 0; b < 2; ++b) {
        // Derive all 12,804 expected words independently of the packing code.
        const int stage = b == 0 ? 7 : 19; // after layer 3 / 9
        std::vector<float> expected(12804);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = float(stage * 16384 + i);
        c.ok &= std::memcmp(expected.data(), &c.snapshots[b], 51216) == 0;
    }
    if (!c.ok || c.routers != 48 || c.boundaries != 2) {
        std::fprintf(stderr, "block probe failed (%d router stops, %d boundaries)\n", c.routers, c.boundaries);
        return 1;
    }
    if (argc == 2) {
        std::string err;
        if (!write_block_boundaries(c.snapshots, argv[1], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    }
    std::printf("%s: 48 router stops, two 51,216-byte synthetic boundaries bit-exact\n", cx.device_name().c_str());
    return 0;
}
