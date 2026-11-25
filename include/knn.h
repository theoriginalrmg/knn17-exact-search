// knn.h — Memory-efficient exact k-nearest-neighbour search (Project 17)
//
// Problem: given a database Y (nb x d, float32, row-major) and queries X
// (nq x d), return for every query the k database ids with the smallest
// squared Euclidean distance, plus those distances.
//
// Output contract (shared by every implementation in this project):
//   * D[nq*k] float, I[nq*k] int64, row i = neighbours of query i.
//   * Each row is sorted by (distance ascending, id ascending).
//   * TIE POLICY: when two database vectors are at the same distance, the one
//     with the SMALLER id wins (both for "who makes the top-k" and ordering).
//   * If nb < k the row is padded with distance +inf and id -1.
//   * Distances are squared L2 (no square root), like Faiss IndexFlatL2.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace knn {

using idx_t = int64_t;

// ---------------------------------------------------------------- D1 -------
// Sequential brute-force reference. Distances are computed in double
// precision as sum((x-y)^2) (no norm expansion), so it is the "ground truth"
// that everything else is checked against.
void reference_search(const float* X, size_t nq, const float* Y, size_t nb,
                      size_t d, size_t k, float* D, idx_t* I);

struct CheckOptions {
    double rtol = -1;     // relative tolerance; <0 => auto (scaled by sqrt(d))
    double atol = 1e-30;  // absolute tolerance
    bool verbose = false;
    size_t max_queries = 0;  // 0 = check all queries, else a strided sample
};

struct CheckReport {
    size_t queries_checked = 0;
    size_t strict_rows_ok = 0;     // ids identical to the reference (tie policy)
    size_t tolerant_rows_ok = 0;   // correct up to floating-point tolerance
    size_t bad_ids = 0;            // out-of-range / duplicate ids
    size_t wrong_neighbours = 0;   // returned id farther than k-th true dist+tol
    size_t bad_distances = 0;      // reported distance != true distance (+-tol)
    size_t unsorted_rows = 0;
    double recall_at_k = 0;        // |found ∩ ref| / k, averaged (strict sets)
    bool passed() const { return tolerant_rows_ok == queries_checked; }
    std::string summary() const;
};

// Exact-neighbour checker. A row passes the *tolerant* check when every
// returned id is valid and unique, its true distance is <= (k-th true
// distance + tol), the reported distance equals the true distance (+-tol) and
// the row is sorted. The *strict* check additionally requires the ids to equal
// the reference ids exactly (i.e. the tie policy was applied identically).
CheckReport check_results(const float* X, size_t nq, const float* Y, size_t nb,
                          size_t d, size_t k, const float* D, const idx_t* I,
                          const CheckOptions& opt = {});

// ---------------------------------------------------------------- D2 -------
// Database pre-packed into "panels" of PANEL consecutive rows, stored
// dimension-major inside a panel: panel p, dim t, lane j -> Y[(p*PANEL+j)*d+t].
// This lets the micro-kernel load PANEL database values for one dimension with
// a single vector load and broadcast-FMA them against a query coordinate —
// the same data layout trick a GEMM uses for its packed B matrix.
constexpr size_t PANEL = 16;

struct PackedDB {
    size_t nb = 0, d = 0, npanels = 0;
    float* panels = nullptr;  // npanels * d * PANEL floats, 64-byte aligned
    float* norms = nullptr;   // npanels * PANEL; padding rows get +inf
    PackedDB() = default;
    PackedDB(const PackedDB&) = delete;
    PackedDB& operator=(const PackedDB&) = delete;
    ~PackedDB();
    size_t bytes() const;
};
// Preprocessing step (done once per database, timed separately).
void pack_database(const float* Y, size_t nb, size_t d, PackedDB& out);

// ---------------------------------------------------------------- D3 -------
struct SearchParams {
    size_t tq = 0;        // queries per tile        (0 = let selector decide)
    size_t tb = 0;        // database rows per block (0 = let selector decide)
    int db_splits = 0;    // S: how many database chunks run in parallel
                          //    1 = query-parallel, >1 = (partly) db-parallel
    int nthreads = 0;     // 0 = omp_get_max_threads()
    std::string describe() const;
};

struct MachineInfo {
    size_t l1d = 48 * 1024, l2 = 2 * 1024 * 1024;
    int nthreads = 1;
    static MachineInfo detect();
};

// The tile/parallelism selector: picks tq, tb and S from the problem shape.
SearchParams select_params(size_t nq, size_t nb, size_t d, size_t k,
                           const MachineInfo& m,
                           size_t mem_budget_bytes = 64u << 20);

// Search with the OpenMP tiled kernel. Fields left 0 in `p` are filled in by
// select_params. Returns the parameters actually used.
SearchParams tiled_search(const PackedDB& db, const float* X, size_t nq,
                          size_t k, float* D, idx_t* I, SearchParams p = {});

// Extra working memory (bytes) tiled_search allocates for given params —
// used in the report to compare against Faiss's 16 MiB ip_block.
size_t tiled_working_bytes(size_t nq, size_t d, size_t k, const SearchParams& p);

// ------------------------------------------------------------ utils -------
double now_sec();
size_t current_rss_bytes();
size_t peak_rss_bytes();

}  // namespace knn
