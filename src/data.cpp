// Input families + simple binary I/O (".fbin": int32 n, int32 d, then n*d
// float32 — the big-ann-benchmarks format, readable from numpy in 2 lines).
#include "data.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace knn {

// Family 1: i.i.d. uniform in [0,1)^d — the "hard" case for exact search, all
// distances concentrate, nothing prunes.
// Family 2: Gaussian mixture (64 centres, sigma 0.1) — clustered like real
// embeddings; queries are drawn from the same mixture.
// Family 3: low-entropy integer vectors (coordinates in {0,1,2,3}) with ~10%
// exact duplicate rows — produces massive numbers of exact distance ties, so
// the tie policy is actually exercised. Integer values keep every float
// computation exact, so results must match the reference bit-for-bit.
std::vector<float> make_family(const std::string& fam, size_t n, size_t d,
                               uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> v(n * d);
    if (fam == "uniform") {
        std::uniform_real_distribution<float> U(0.f, 1.f);
        for (auto& x : v) x = U(rng);
    } else if (fam == "clustered") {
        const size_t C = 64;
        std::mt19937_64 crng(12345);  // centres shared by db and queries
        std::uniform_real_distribution<float> U(0.f, 1.f);
        std::vector<float> cent(C * d);
        for (auto& x : cent) x = U(crng);
        std::normal_distribution<float> N(0.f, 0.1f);
        std::uniform_int_distribution<size_t> pick(0, C - 1);
        for (size_t i = 0; i < n; i++) {
            size_t c = pick(rng);
            for (size_t t = 0; t < d; t++) v[i * d + t] = cent[c * d + t] + N(rng);
        }
    } else if (fam == "ties") {
        std::uniform_int_distribution<int> U(0, 3);
        for (auto& x : v) x = float(U(rng));
        std::uniform_real_distribution<double> P(0, 1);
        for (size_t i = 1; i < n; i++)
            if (P(rng) < 0.1) {  // duplicate an earlier row
                size_t src = std::uniform_int_distribution<size_t>(0, i - 1)(rng);
                for (size_t t = 0; t < d; t++) v[i * d + t] = v[src * d + t];
            }
    } else {
        throw std::runtime_error("unknown family: " + fam);
    }
    return v;
}

void write_fbin(const std::string& path, const float* x, size_t n, size_t d) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    int32_t hdr[2] = {int32_t(n), int32_t(d)};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(x, sizeof(float), n * d, f);
    std::fclose(f);
}

std::vector<float> read_fbin(const std::string& path, size_t& n, size_t& d) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    int32_t hdr[2];
    if (std::fread(hdr, sizeof(hdr), 1, f) != 1) throw std::runtime_error("bad header");
    n = size_t(hdr[0]);
    d = size_t(hdr[1]);
    std::vector<float> v(n * d);
    if (std::fread(v.data(), sizeof(float), n * d, f) != n * d)
        throw std::runtime_error("short read " + path);
    std::fclose(f);
    return v;
}

// .fvecs (SIFT/GIST format): per vector int32 d, then d floats.
std::vector<float> read_fvecs(const std::string& path, size_t& n, size_t& d,
                              size_t max_n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<float> v;
    n = 0;
    int32_t dim;
    while ((max_n == 0 || n < max_n) && std::fread(&dim, 4, 1, f) == 1) {
        d = size_t(dim);
        size_t off = v.size();
        v.resize(off + d);
        if (std::fread(v.data() + off, sizeof(float), d, f) != d) break;
        n++;
    }
    std::fclose(f);
    return v;
}

}  // namespace knn
