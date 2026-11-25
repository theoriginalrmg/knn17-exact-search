#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace knn {
std::vector<float> make_family(const std::string& fam, size_t n, size_t d, uint64_t seed);
void write_fbin(const std::string& path, const float* x, size_t n, size_t d);
std::vector<float> read_fbin(const std::string& path, size_t& n, size_t& d);
std::vector<float> read_fvecs(const std::string& path, size_t& n, size_t& d, size_t max_n = 0);
}  // namespace knn
