#include "block_setup.h"
#include "ref_quant.h"

#include "strata/core/family_layout.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/mrope.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>

namespace {
constexpr int64_t context = 65536;  // same as the captured Metal run

bool checked(cudaError_t status, const char* operation, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string(operation) + ": " + cudaGetErrorString(status);
    return false;
}

// WeightTable keeps the shapes of skipped tensors. Verifier needs output.weight's
// vocabulary dimension even though this stage never runs the head or embedding.
bool stage_skip(const std::string& pack, const std::vector<std::string>& shards,
                std::set<std::string>& skip, std::string& err) {
    if (!strata::core::NativeDense::served_names(shards, false, skip, err)) return false;
    std::ifstream index(pack + "/index.txt");
    if (!index) { err = "cannot open " + pack + "/index.txt"; return false; }
    std::string line;
    while (std::getline(index, line)) {
        std::istringstream row(line);
        std::string name;
        if (!(row >> name) || name.front() == '#') continue;
        int layer = -1;
        if (std::sscanf(name.c_str(), "blk.%d.", &layer) != 1 ||
            layer < BlockSetup::first_layer || layer >= BlockSetup::end_layer)
            skip.insert(name);
    }
    if (index.bad()) { err = "cannot read pack index"; return false; }
    return true;
}
}

bool BlockSetup::match_metal_hc(std::string& err) {
    // The captured Metal window quantizes BF16 down, up AND inject matrices.
    // Reuse its CPU conversion so the weight bytes match across backends.
    for (int layer = first_layer; layer < end_layer; ++layer) {
        for (const char* half : {"attn", "ffn"}) {
            for (const char* projection : {"down", "up", "inject"}) {
                const std::string name = "blk." + std::to_string(layer) + ".hc_" + half + "_" + projection + ".weight";
                // weights is owned and mutable here. find exposes a const view for engine readers.
                auto* weight = const_cast<strata::core::WeightRef*>(weights.find(name));
                if (!weight || !weight->data || weight->hc_q8 ||
                    weight->kind != strata::core::WeightKind::Bf16InF32 ||
                    weight->ne0 % 32 || weight->bytes != (uint64_t) weight->elements * 2) {
                    err = "expected unconverted BF16 hyper-connection weight: " + name;
                    return false;
                }
                std::vector<uint16_t> raw(weight->elements);
                if (!checked(cudaMemcpy(raw.data(), weight->data, weight->bytes, cudaMemcpyDeviceToHost),
                             "read HC weight for Metal conversion", err)) return false;
                std::vector<float> values(raw.size());
                for (size_t i = 0; i < raw.size(); ++i) values[i] = ref::bf16_to_f32(raw[i]);
                std::vector<uint8_t> quantized(raw.size() / 32 * 34);
                ref::quantize_q8_0(values.data(), values.size(), quantized.data());
                void* buffer = nullptr;
                if (!checked(cudaMalloc(&buffer, quantized.size()), "allocate HC Q8 weight", err)) return false;
                hc_weights_.push_back(buffer);
                if (!checked(cudaMemcpy(buffer, quantized.data(), quantized.size(), cudaMemcpyHostToDevice),
                             "upload HC Q8 weight", err)) return false;
                weight->hc_q8 = buffer;
            }
        }
    }
    std::printf("hc_weight_mode=metal_q8_down_up_inject\n");
    return true;
}

BlockSetup::~BlockSetup() {
    for (void* buffer : hc_weights_) cudaFree(buffer);
    // qsa_states uses global ordinals even in a trimmed session.
    if (session.qsa_states) {
        for (int64_t j = 0; j < session.qsa_alloc; ++j) {
            auto& state = session.qsa_states[session.qsa_ord0 + j];
            if (state.owns_rope) strata::kernels::rope_table_release(state.cos_tab);
            if (state.host_step) cudaFreeHost(state.host_step);
            if (state.host_pos) cudaFreeHost(state.host_pos);
        }
        delete[] session.qsa_states;
    }
    if (device_residency_) cudaFree(device_residency_);
    if (session_arena_) cudaFree(session_arena_);
    if (weights_arena_) cudaFree(weights_arena_);
}

bool BlockSetup::init(const std::string& pack, const std::string& gguf, std::string& err) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    if (weights_arena_) { err = "block setup already initialized"; return false; }
    int64_t top_k = 0;
    {
        strata::GgufFile file(gguf);
        if (!geometry_from_gguf(file, geometry, top_k, err)) return false;
        err = compiled_geometry_mismatch(geometry);
        if (!err.empty()) return false;
    }
    if (geometry.n_expert != 512 || top_k != 10) {
        err = "the capture requires 512 experts and top-10 routing";
        return false;
    }
    if (!expert_layout_load(pack, geometry.n_layers, geometry.n_expert, err)) return false;
    const auto& layout = expert_layout();
    if (!layout.native || layout.n_expert != geometry.n_expert) {
        err = "expected the IQ3_S native expert layout";
        return false;
    }
    std::vector<int64_t> sizes;
    for (int layer = first_layer; layer < end_layer; ++layer) {
        const auto& f = layout.fmt[layer];
        if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
            err = "unsupported expert format in layer " + std::to_string(layer);
            return false;
        }
        sizes.insert(sizes.end(), geometry.n_expert, layout.blob_bytes(layer));
    }

    const std::vector<std::string> shards{gguf};
    std::set<std::string> skip;
    uint64_t pool = 0;
    if (!stage_skip(pack, shards, skip, err) || !WeightTable::pool_bytes(pack, pool, err, &skip) ||
        !checked(cudaMalloc(&weights_arena_, pool), "allocate canonical weights", err) ||
        !weights.load(pack, weights_arena_, pool, err, &skip) ||
        !dense_.load(shards, weights, err, false, first_layer, end_layer) ||
        !match_metal_hc(err))
        return false;

    const auto state_bytes = session_bytes(geometry, context, top_k, first_layer, end_layer);
    if (!checked(cudaMalloc(&session_arena_, state_bytes), "allocate session", err)) return false;
    if (!session_init(geometry, context, top_k, session_arena_, session, first_layer, end_layer)) {
        err = "initialize stage session";
        return false;
    }
    session_zero(session, geometry, nullptr, nullptr);
    if (!checked(cudaDeviceSynchronize(), "zero stage session", err)) return false;

    FileExpertSource source;
    source.set_gguf(gguf);
    if (!source.open(pack, geometry.n_layers, geometry.n_expert, err) ||
        !cache_.open_sized(sizes, geometry.n_layers, geometry.n_expert, err))
        return false;
    residency_.assign(geometry.n_layers * geometry.n_expert, kNotResident);
    for (int layer = first_layer; layer < end_layer; ++layer) {
        for (int e = 0; e < geometry.n_expert; ++e) {
            const auto slot = cache_.admit(layer, e);
            const auto* bytes = source.blob_stable(layer, e);
            if (slot < 0 || !bytes) {
                err = "missing expert " + std::to_string(e) + " in layer " + std::to_string(layer);
                return false;
            }
            if (!cache_.fill_slot_blocking(slot, bytes, err, layout.blob_bytes(layer))) return false;
            if (e == 0 && !cache_.verify_slot(slot, bytes, err, layout.blob_bytes(layer))) return false;
            residency_[layer * geometry.n_expert + e] = slot;
        }
    }
    const size_t table_bytes = residency_.size() * sizeof(int32_t);
    if (!checked(cudaMalloc(&device_residency_, table_bytes), "allocate residency map", err) ||
        !checked(cudaMemcpy(device_residency_, residency_.data(), table_bytes, cudaMemcpyHostToDevice),
                 "upload residency map", err))
        return false;
    hits.d_res = device_residency_;
    hits.h_res = residency_.data();
    hits.cache_base = cache_.device_slot(0);
    hits.slot_off = cache_.slot_offsets();
    hits.n_slots = cache_.slots();
    hits.blob = layout.max_blob;
    std::printf("resident_experts=%lld expert_bytes=%lld canonical_bytes=%llu native_dense_bytes=%llu session_bytes=%llu\n",
                (long long) cache_.resident(), (long long) cache_.bytes(),
                (unsigned long long) pool, (unsigned long long) dense_.allocated_bytes(),
                (unsigned long long) state_bytes);
    return true;
}
