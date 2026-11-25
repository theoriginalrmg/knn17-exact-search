// D2: OpenMP tiled exact k-NN with per-query top-k buffers (fused selection)
// D3: tile / parallelism selector for small vs large query batches.
//
// Idea (adapted from Johnson, Douze & Jégou, "Billion-scale similarity search
// with GPUs", sec. 5.1 "Exact search"):
//   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
// ||y||^2 is precomputed once per database vector, ||x||^2 is constant per
// query so it can be left out of the selection and added back at the end, and
// <x, y> for a tile of queries x a tile of database rows is a small GEMM.
// The paper fuses the k-selection into the kernel that consumes the GEMM
// output so the nq x nb distance matrix is never written to memory. We go one
// step further on the CPU: the distances are consumed straight out of the
// micro-kernel's registers, so no distance tile is stored at all. The only
// working memory is the k-entry heaps (one per query) plus a copy of the
// current query tile.
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "knn.h"

namespace knn {

typedef float v16 __attribute__((vector_size(64)));  // GCC/Clang vector ext.
static_assert(sizeof(v16) == PANEL * sizeof(float), "panel = one vector");

constexpr size_t MR = 4;  // queries per micro-kernel call (register rows)
constexpr float INF = std::numeric_limits<float>::infinity();

static void* aligned_alloc64(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, std::max<size_t>(bytes, 64)) != 0) return nullptr;
    return p;
}

// ------------------------------------------------------------ packing -----
PackedDB::~PackedDB() {
    std::free(panels);
    std::free(norms);
}
size_t PackedDB::bytes() const {
    return npanels * PANEL * (d + 1) * sizeof(float);
}

void pack_database(const float* Y, size_t nb, size_t d, PackedDB& out) {
    std::free(out.panels);
    std::free(out.norms);
    out.nb = nb;
    out.d = d;
    out.npanels = (nb + PANEL - 1) / PANEL;
    out.panels = (float*)aligned_alloc64(out.npanels * d * PANEL * sizeof(float));
    out.norms = (float*)aligned_alloc64(out.npanels * PANEL * sizeof(float));
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < out.npanels; p++) {
        float* P = out.panels + p * d * PANEL;
        for (size_t j = 0; j < PANEL; j++) {
            size_t row = p * PANEL + j;
            float n = 0;
            for (size_t t = 0; t < d; t++) {
                float v = row < nb ? Y[row * d + t] : 0.f;
                P[t * PANEL + j] = v;
                n += v * v;
            }
            out.norms[p * PANEL + j] = row < nb ? n : INF;  // padding never wins
        }
    }
}

// ---------------------------------------------------------- top-k heap ----
// Max-heap of size k on (dist, id): the root is the current WORST neighbour,
// so a candidate only has to beat hd[0] to get in. Ordering implements the
// tie policy: larger distance is worse; equal distance -> larger id is worse.
static inline bool worse(float da, idx_t ia, float db, idx_t ib) {
    return da > db || (da == db && ia > ib);
}

static inline void heap_replace_top(float* hd, idx_t* hi, size_t k, float d, idx_t id) {
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, c;
        if (l >= k) break;
        c = (r < k && worse(hd[r], hi[r], hd[l], hi[l])) ? r : l;
        if (!worse(hd[c], hi[c], d, id)) break;
        hd[i] = hd[c];
        hi[i] = hi[c];
        i = c;
    }
    hd[i] = d;
    hi[i] = id;
}

static inline void heap_init(float* hd, idx_t* hi, size_t k) {
    for (size_t r = 0; r < k; r++) { hd[r] = INF; hi[r] = -1; }
}

// ------------------------------------------------------- micro-kernel -----
// acc[r][c] = <query r, database rows of panel c> for MR queries x NP panels.
// Each step loads NP vectors of database values for dimension t and does
// MR*NP broadcast-FMAs: the CPU analogue of the GEMM inner loop.
template <int NP>
static inline void micro_kernel(const float* __restrict Q, size_t d,
                                const float* __restrict P, v16 acc[MR][NP]) {
    for (size_t r = 0; r < MR; r++)
        for (int c = 0; c < NP; c++) acc[r][c] = (v16){};
    const size_t ps = d * PANEL;  // panel stride
    for (size_t t = 0; t < d; t++) {
        v16 b[NP];
        for (int c = 0; c < NP; c++) b[c] = *(const v16*)(P + c * ps + t * PANEL);
        for (size_t r = 0; r < MR; r++) {
            float a = Q[r * d + t];
            for (int c = 0; c < NP; c++) acc[r][c] += a * b[c];
        }
    }
}

// Consume the micro-kernel output directly into the per-query heaps.
// pd = ||y||^2 - 2<x,y>  (the ||x||^2 term is added after selection).
template <int NP>
static inline void select_from_registers(v16 acc[MR][NP], const float* yn,
                                         idx_t id0, size_t nrows, float** hd,
                                         idx_t** hi, float* thr, size_t k) {
    for (size_t r = 0; r < nrows; r++) {
        for (int c = 0; c < NP; c++) {
            v16 pd = *(const v16*)(yn + c * PANEL) - 2.f * acc[r][c];
            // cheap vector filter: skip all 16 lanes if none beats the worst
            v16 t = (v16){} + thr[r];
            auto m = pd < t;
            bool any = false;
            for (size_t j = 0; j < PANEL; j++) any |= (m[j] != 0);
            if (!any) continue;
            for (size_t j = 0; j < PANEL; j++) {
                float v = pd[j];
                // ids are scanned in increasing order inside a task, so a tie
                // (v == thr) must lose to the earlier id: strict < is exact.
                if (v < thr[r]) {
                    heap_replace_top(hd[r], hi[r], k, v, id0 + c * PANEL + j);
                    thr[r] = hd[r][0];
                }
            }
        }
    }
}

// ------------------------------------------------------------ selector ----
MachineInfo MachineInfo::detect() {
    MachineInfo m;
    long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE), l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l1 > 0) m.l1d = size_t(l1);
    if (l2 > 0) m.l2 = size_t(l2);
    m.nthreads = omp_get_max_threads();
    return m;
}

std::string SearchParams::describe() const {
    std::ostringstream os;
    os << "tq=" << tq << " tb=" << tb << " S=" << db_splits << " T=" << nthreads
       << (db_splits == 1 ? " (query-parallel)" : " (db-parallel)");
    return os.str();
}

static size_t round_up(size_t a, size_t b) { return (a + b - 1) / b * b; }

SearchParams select_params(size_t nq, size_t nb, size_t d, size_t k,
                           const MachineInfo& m, size_t mem_budget) {
    SearchParams p;
    p.nthreads = m.nthreads > 0 ? m.nthreads : 1;
    const int T = p.nthreads;

    // Database block: half of L2 so the block stays resident while every
    // query group of the tile streams over it (reuse factor tq / MR).
    size_t tb = (m.l2 / 2) / (sizeof(float) * std::max<size_t>(d, 1));
    tb = std::max<size_t>(2 * PANEL, tb / (2 * PANEL) * (2 * PANEL));
    tb = std::min(tb, round_up(std::max<size_t>(nb, 1), 2 * PANEL));
    p.tb = tb;

    // Query tile: 64 queries is enough to amortise one L2 block (16 reuses),
    // but keep the tile itself + its heaps within ~L1..L2/4.
    size_t tq = 64;
    while (tq > MR && tq * (d * sizeof(float) + k * 12) > m.l2 / 4) tq /= 2;
    tq = std::min(tq, round_up(std::max<size_t>(nq, 1), MR));
    p.tq = std::max(tq, MR);

    // Parallelism: large batches -> query-parallel (no merge, no extra
    // memory); small batches -> split the database into S chunks so all T
    // threads work, each streaming only nb/S rows. Pick the smallest S that
    // gives >= T tasks with <= 25% load imbalance, subject to chunks of at
    // least one block and the partial-heap memory budget.
    size_t nqt = (nq + p.tq - 1) / p.tq;
    int smax = int(std::max<size_t>(1, std::min<size_t>(size_t(4 * T), nb / tb)));
    int S = 1;
    for (int s = 1; s <= smax; s++) {
        S = s;
        size_t tasks = nqt * size_t(s);
        size_t per = (tasks + T - 1) / T;
        double imbalance = double(per * T) / double(tasks);
        if (tasks >= size_t(T) && imbalance <= 1.25) break;
    }
    while (S > 1 && size_t(S) * nq * k * 12 > mem_budget) S--;
    p.db_splits = S;
    return p;
}

size_t tiled_working_bytes(size_t nq, size_t d, size_t k, const SearchParams& p) {
    size_t per_thread = round_up(p.tq, MR) * (d + 1) * sizeof(float);
    size_t partial = p.db_splits > 1 ? size_t(p.db_splits) * nq * k * 12 : 0;
    return per_thread * size_t(p.nthreads) + partial;
}

// --------------------------------------------------------------- search ----
SearchParams tiled_search(const PackedDB& db, const float* X, size_t nq,
                          size_t k, float* D, idx_t* I, SearchParams p) {
    const size_t d = db.d, nb = db.nb;
    MachineInfo m = MachineInfo::detect();
    if (p.nthreads > 0) m.nthreads = p.nthreads;
    SearchParams auto_p = select_params(nq, nb, d, k, m);
    if (!p.tq) p.tq = auto_p.tq;
    if (!p.tb) p.tb = auto_p.tb;
    if (!p.db_splits) p.db_splits = auto_p.db_splits;
    if (!p.nthreads) p.nthreads = auto_p.nthreads;
    p.tq = round_up(p.tq, MR);
    p.tb = std::max<size_t>(PANEL, round_up(p.tb, PANEL));
    if (nq == 0 || k == 0) return p;

    const size_t S = size_t(std::max(1, p.db_splits));
    const size_t nqt = (nq + p.tq - 1) / p.tq;
    const size_t np = db.npanels;
    const size_t bp = p.tb / PANEL;  // panels per block

    // Heap storage: with S == 1 the output arrays themselves are the heaps
    // (zero extra memory). With S > 1 each chunk gets its own nq x k heaps.
    std::vector<float> part_d;
    std::vector<idx_t> part_i;
    float* HD = D;
    idx_t* HI = I;
    if (S > 1) {
        part_d.resize(S * nq * k);
        part_i.resize(S * nq * k);
        HD = part_d.data();
        HI = part_i.data();
    }

#pragma omp parallel num_threads(p.nthreads)
    {
        const size_t tqp = p.tq;
        float* Qt = (float*)aligned_alloc64(tqp * d * sizeof(float));
        std::vector<float> thr(tqp);
        std::vector<float*> hdp(tqp);
        std::vector<idx_t*> hip(tqp);

#pragma omp for schedule(dynamic, 1)
        for (size_t task = 0; task < nqt * S; task++) {
            size_t qt = task / S, s = task % S;
            size_t q0 = qt * tqp, nqt_rows = std::min(tqp, nq - q0);
            size_t p0 = np * s / S, p1 = np * (s + 1) / S;  // this chunk's panels

            // copy query tile (zero-pad to a multiple of MR) and set up heaps
            std::memcpy(Qt, X + q0 * d, nqt_rows * d * sizeof(float));
            std::memset(Qt + nqt_rows * d, 0, (tqp - nqt_rows) * d * sizeof(float));
            for (size_t r = 0; r < nqt_rows; r++) {
                hdp[r] = HD + (s * nq + q0 + r) * k;
                hip[r] = HI + (s * nq + q0 + r) * k;
                heap_init(hdp[r], hip[r], k);
                thr[r] = INF;
            }

            for (size_t b0 = p0; b0 < p1; b0 += bp) {        // L2 block of DB
                size_t b1 = std::min(p1, b0 + bp);
                for (size_t g = 0; g < nqt_rows; g += MR) {    // query group
                    size_t rows = std::min(MR, nqt_rows - g);
                    const float* Qg = Qt + g * d;
                    size_t pp = b0;
                    for (; pp + 2 <= b1; pp += 2) {
                        v16 acc[MR][2];
                        micro_kernel<2>(Qg, d, db.panels + pp * d * PANEL, acc);
                        select_from_registers<2>(acc, db.norms + pp * PANEL,
                                                 idx_t(pp * PANEL), rows, &hdp[g],
                                                 &hip[g], &thr[g], k);
                    }
                    if (pp < b1) {
                        v16 acc[MR][1];
                        micro_kernel<1>(Qg, d, db.panels + pp * d * PANEL, acc);
                        select_from_registers<1>(acc, db.norms + pp * PANEL,
                                                 idx_t(pp * PANEL), rows, &hdp[g],
                                                 &hip[g], &thr[g], k);
                    }
                }
            }
        }
        std::free(Qt);

        // ---- finalise: merge chunks (if any), add ||x||^2, sort by policy
        std::vector<std::pair<float, idx_t>> cand(S * k);
#pragma omp for schedule(static)
        for (size_t i = 0; i < nq; i++) {
            size_t nc = 0;
            for (size_t s = 0; s < S; s++)
                for (size_t r = 0; r < k; r++) {
                    idx_t id = HI[(s * nq + i) * k + r];
                    if (id >= 0) cand[nc++] = {HD[(s * nq + i) * k + r], id};
                }
            auto by_policy = [](const std::pair<float, idx_t>& a,
                                const std::pair<float, idx_t>& b) {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            };
            size_t kk = std::min(k, nc);
            std::partial_sort(cand.begin(), cand.begin() + kk, cand.begin() + nc, by_policy);
            const float* x = X + i * d;
            float xn = 0;
            for (size_t t = 0; t < d; t++) xn += x[t] * x[t];
            for (size_t r = 0; r < kk; r++)
                cand[r].first = std::max(0.f, cand[r].first + xn);
            // adding ||x||^2 can create float ties: re-apply the policy
            std::sort(cand.begin(), cand.begin() + kk, by_policy);
            for (size_t r = 0; r < k; r++) {
                D[i * k + r] = r < kk ? cand[r].first : INF;
                I[i * k + r] = r < kk ? cand[r].second : -1;
            }
        }
    }
    return p;
}

}  // namespace knn
