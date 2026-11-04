#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"

#include <string>
#include <vector>

// Owns the state read by the standalone [4,10) verifier. Destroy the verifier first.
struct BlockSetup {
    static constexpr int first_layer = 4;
    static constexpr int end_layer = 10;
    strata::core::WeightTable weights;
    strata::core::ModelGeometry geometry;
    strata::core::SessionState session;
    strata::core::VerifyHits hits;

    bool init(const std::string& pack, const std::string& gguf, std::string& err);
    ~BlockSetup();

private:
    strata::core::NativeDense dense_;
    strata::core::ExpertCache cache_;
    void* weights_arena_ = nullptr;
    void* session_arena_ = nullptr;
    int32_t* device_residency_ = nullptr;
    std::vector<int32_t> residency_;
};
