// STREAM-style read bandwidth: every thread sums its own contiguous part of one
// large array (64-byte aligned like the packed database, touched first by the
// threads that read it). Integer adds, so latency of FP adds is not the limit.
//
//   bandwidth MIB THREADS REPS   -> one CSV row:  threads,MiB,median,best,std (GB/s)
//   bandwidth header
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "knn.h"

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "header")) {
        std::puts("threads,MiB,median_GBs,best_GBs,std_GBs");
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: bandwidth MIB THREADS REPS | bandwidth header\n");
        return 1;
    }
    const size_t mib = std::strtoul(argv[1], nullptr, 10);
    const int T = std::atoi(argv[2]), reps = std::atoi(argv[3]);
    if (mib == 0 || T < 1 || reps < 1) return 1;
    const size_t n = (mib << 20) / sizeof(uint64_t);
    void* mem = nullptr;
    if (posix_memalign(&mem, 64, n * sizeof(uint64_t))) return 1;
    uint64_t* a = (uint64_t*)mem;
#pragma omp parallel for schedule(static) num_threads(T)
    for (size_t i = 0; i < n; i++) a[i] = i;

    const uint64_t want = uint64_t(n) * (n - 1) / 2;
    std::vector<double> gbs;
    for (int r = 0; r <= reps; r++) {  // r = 0 is a warm-up
        uint64_t sum = 0;
        double t0 = knn::now_sec();
#pragma omp parallel for simd schedule(static) num_threads(T) reduction(+ : sum)
        for (size_t i = 0; i < n; i++) sum += a[i];
        double t = knn::now_sec() - t0;
        if (sum != want) { std::fprintf(stderr, "wrong sum\n"); return 2; }
        if (r) gbs.push_back(n * sizeof(uint64_t) / t / 1e9);
    }
    std::free(mem);
    std::sort(gbs.begin(), gbs.end());
    double mean = 0, var = 0;
    for (double x : gbs) mean += x / gbs.size();
    for (double x : gbs) var += (x - mean) * (x - mean);
    std::printf("%d,%zu,%.3f,%.3f,%.3f\n", T, mib, gbs[gbs.size() / 2], gbs.back(),
                gbs.size() > 1 ? std::sqrt(var / (gbs.size() - 1)) : 0.0);
    return 0;
}
