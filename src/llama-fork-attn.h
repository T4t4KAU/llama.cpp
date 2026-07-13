#pragma once

#include <cstdint>
#include <vector>

struct llama_fork_attn_plan {
    static constexpr int32_t  magic       = 0x4641544e;  // FATN
    static constexpr int32_t  version     = 1;
    static constexpr uint32_t header_size = 8;

    bool     active             = false;
    uint32_t n_kv               = 0;
    uint32_t n_queries          = 0;
    uint32_t common_length      = 0;
    uint32_t max_private_length = 0;
    uint64_t saved_kv_reads     = 0;

    std::vector<int32_t>              common;
    std::vector<std::vector<int32_t>> private_cells;

    size_t serialized_size() const { return header_size + n_kv + n_queries + n_queries * n_kv; }

    std::vector<int32_t> serialize() const;
};
