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

// Compile-time kernel choice: AVX-512 -> 4 x 32 tile (GCC vectors, also the
// portable fallback); AVX2+FMA -> hand-written 6 x 16 tile.
#ifdef __AVX512F__
#define KNN_AVX2_KERNEL 0
#elif defined(__AVX2__) && defined(__FMA__)
#define KNN_AVX2_KERNEL 1
#include <immintrin.h>
#else
#define KNN_AVX2_KERNEL 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "knn.h"

namespace knn {

// MR = queries per micro-kernel call (register rows), NP_MAX = panels per call.
#if KNN_AVX2_KERNEL
struct v16 { __m256 lo, hi; };  // one panel of results = two AVX2 registers
constexpr size_t MR = 6;
constexpr int NP_MAX = 1;
#else
typedef float v16 __attribute__((vector_size(64)));  // GCC/Clang vector ext.
constexpr size_t MR = 4;
constexpr int NP_MAX = 2;
#endif
static_assert(sizeof(v16) == PANEL * sizeof(float), "v16 holds one panel");
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
    double mx = 0;
#pragma omp parallel for schedule(static) reduction(max : mx)
    for (size_t row = 0; row < nb; row++) {
        double n = 0;
        for (size_t t = 0; t < d; t++) n += double(Y[row * d + t]) * Y[row * d + t];
        mx = std::max(mx, n);
    }
    out.max_norm = mx;
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
// acc[r][c] = <query r, database rows of panel c> for R queries x NP panels.
#if KNN_AVX2_KERNEL
// 6 rows: 12 accumulators + 2 loads + 1 broadcast = 15 of the 16 AVX2 registers.
// 1 and 2 rows: tiny batches (nq < MR), so no work is spent on padding rows.
// Accumulators are named locals (not an array) so they stay in registers.
#define KNN_INIT(r)                                                  \
    __m256 lo##r = _mm256_setzero_ps(), hi##r = _mm256_setzero_ps(); \
    const float* q##r = Q + r * d;
#define KNN_STEP(r)                         \
    a = _mm256_broadcast_ss(q##r + t);      \
    lo##r = _mm256_fmadd_ps(a, blo, lo##r); \
    hi##r = _mm256_fmadd_ps(a, bhi, hi##r);
#define KNN_STORE(r) acc[r][0] = {lo##r, hi##r};
#define KNN_ROWS1(X) X(0)
#define KNN_ROWS2(X) X(0) X(1)
#define KNN_ROWS6(X) X(0) X(1) X(2) X(3) X(4) X(5)
#define KNN_KERNEL(ROWS)                                              \
    ROWS(KNN_INIT)                                                    \
    for (size_t t = 0; t < d; t++) {                                  \
        const __m256 blo = _mm256_load_ps(P + t * PANEL);             \
        const __m256 bhi = _mm256_load_ps(P + t * PANEL + 8);         \
        __m256 a;                                                     \
        ROWS(KNN_STEP)                                                \
    }                                                                 \
    ROWS(KNN_STORE)

template <int R, int NP>
static inline void micro_kernel(const float* __restrict Q, size_t d,
                                const float* __restrict P, v16 (*acc)[NP]) {
    static_assert(NP == 1, "the AVX2 tile is one panel wide");
    static_assert(R == 1 || R == 2 || R == 6, "no AVX2 kernel with R rows");
    if constexpr (R == 1) {
        KNN_KERNEL(KNN_ROWS1)
    } else if constexpr (R == 2) {
        KNN_KERNEL(KNN_ROWS2)
    } else {
        KNN_KERNEL(KNN_ROWS6)
    }
}
#undef KNN_KERNEL
#undef KNN_ROWS1
#undef KNN_ROWS2
#undef KNN_ROWS6
#undef KNN_INIT
#undef KNN_STEP
#undef KNN_STORE
#else
// Each step loads NP vectors of database values for dimension t and does
// R*NP broadcast-FMAs: the CPU analogue of the GEMM inner loop.
template <int R, int NP>
static inline void micro_kernel(const float* __restrict Q, size_t d,
                                const float* __restrict P, v16 (*acc)[NP]) {
    for (size_t r = 0; r < size_t(R); r++)
        for (int c = 0; c < NP; c++) acc[r][c] = (v16){};
    const size_t ps = d * PANEL;  // panel stride
    for (size_t t = 0; t < d; t++) {
        v16 b[NP];
        for (int c = 0; c < NP; c++) b[c] = *(const v16*)(P + c * ps + t * PANEL);
        for (size_t r = 0; r < size_t(R); r++) {
            float a = Q[r * d + t];
            for (int c = 0; c < NP; c++) acc[r][c] += a * b[c];
        }
    }
}
#endif

// --------------------------------------------------- buffer selection ----
// Alternative to the heap for large k: append every candidate below the
// threshold to a buffer of capacity 2k (4k was slower); when it is full, keep the k best with
// nth_element and set the threshold to the k-th. O(1) amortised per candidate
// instead of O(log k), at the cost of a looser threshold between compactions.
using Cand = std::pair<float, idx_t>;
constexpr size_t BUF_FACTOR = 2;  // buffer capacity = BUF_FACTOR * k

static inline bool by_policy(const Cand& a, const Cand& b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
}
// Keeps the k best of buf[0..n) in buf[0..k) (unordered); returns the k-th.
static inline float compact(Cand* buf, size_t n, size_t k) {
    std::nth_element(buf, buf + (k - 1), buf + n, by_policy);
    return buf[k - 1].first;
}

// Per-query selection state inside one task.
struct QSel {
    float* hd;    // heap mode: k distances (max-heap)
    idx_t* hi;    //            k ids
    Cand* buf;    // buffer mode: capacity BUF_FACTOR * k
    size_t n;     //              entries in buf
};

// Consume the micro-kernel output directly into the per-query heaps/buffers.
// pd = ||y||^2 - 2<x,y>  (the ||x||^2 term is added after selection).
template <bool BUF, int NP>
static inline void select_from_registers(v16 (*acc)[NP], const float* yn,
                                         idx_t id0, size_t nrows, QSel* q,
                                         float* thr, size_t k) {
    for (size_t r = 0; r < nrows; r++) {
        for (int c = 0; c < NP; c++) {
#if KNN_AVX2_KERNEL
            const __m256 two = _mm256_set1_ps(2.f), t = _mm256_set1_ps(thr[r]);
            const float* y = yn + c * PANEL;
            __m256 plo = _mm256_sub_ps(_mm256_load_ps(y), _mm256_mul_ps(two, acc[r][c].lo));
            __m256 phi = _mm256_sub_ps(_mm256_load_ps(y + 8), _mm256_mul_ps(two, acc[r][c].hi));
            // cheap vector filter: skip all 16 lanes if none beats the worst
            __m256 any = _mm256_or_ps(_mm256_cmp_ps(plo, t, _CMP_LT_OQ),
                                      _mm256_cmp_ps(phi, t, _CMP_LT_OQ));
            if (_mm256_testz_ps(any, any)) continue;
            alignas(32) float pd[PANEL];
            _mm256_store_ps(pd, plo);
            _mm256_store_ps(pd + 8, phi);
#else
            v16 pd = *(const v16*)(yn + c * PANEL) - 2.f * acc[r][c];
            // cheap vector filter: skip all 16 lanes if none beats the worst
            v16 t = (v16){} + thr[r];
            auto m = pd < t;
            bool any = false;
            for (size_t j = 0; j < PANEL; j++) any |= (m[j] != 0);
            if (!any) continue;
#endif
            for (size_t j = 0; j < PANEL; j++) {
                float v = pd[j];
                // ids are scanned in increasing order inside a task, so a tie
                // (v == thr) must lose to the earlier id: strict < is exact.
                if (v < thr[r]) {
                    const idx_t id = id0 + c * PANEL + j;
                    if constexpr (BUF) {
                        QSel& s = q[r];
                        s.buf[s.n++] = {v, id};
                        if (s.n == BUF_FACTOR * k) {
                            thr[r] = compact(s.buf, s.n, k);
                            s.n = k;
                        }
                    } else {
                        heap_replace_top(q[r].hd, q[r].hi, k, v, id);
                        thr[r] = q[r].hd[0];
                    }
                }
            }
        }
    }
}

// All queries of a tile against database panels [b0, b1), R queries at a time.
template <bool BUF, int R>
static inline void scan_block(const PackedDB& db, const float* Qt, size_t nqt_rows,
                              size_t b0, size_t b1, QSel* q, float* thr, size_t k) {
    const size_t d = db.d;
    for (size_t g = 0; g < nqt_rows; g += R) {
        const size_t rows = std::min<size_t>(R, nqt_rows - g);
        const float* Qg = Qt + g * d;
        size_t pp = b0;
        for (; pp + NP_MAX <= b1; pp += NP_MAX) {
            v16 acc[R][NP_MAX];
            micro_kernel<R, NP_MAX>(Qg, d, db.panels + pp * d * PANEL, acc);
            select_from_registers<BUF, NP_MAX>(acc, db.norms + pp * PANEL, idx_t(pp * PANEL),
                                               rows, q + g, thr + g, k);
        }
        if (NP_MAX > 1 && pp < b1) {
            v16 acc[R][1];
            micro_kernel<R, 1>(Qg, d, db.panels + pp * d * PANEL, acc);
            select_from_registers<BUF, 1>(acc, db.norms + pp * PANEL, idx_t(pp * PANEL),
                                          rows, q + g, thr + g, k);
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
       << " mr=" << mr << " sel=" << (sel == 2 ? "buf" : "heap")
       << (strict ? " strict rescans=" + std::to_string(strict_rescans) : std::string())
       << (db_splits == 1 ? " (query-parallel)" : " (db-parallel)");
    return os.str();
}

static size_t round_up(size_t a, size_t b) { return (a + b - 1) / b * b; }

// Smallest k at which the buffer was no slower than the heap at nq = 16 and
// 1024 (14 threads), for that k and every larger one tested (k = 1..1000).
constexpr size_t SEL_BUF_MIN_K = 128;

// Strict mode: extra candidates kept per query beyond k.
static size_t strict_extra(size_t k) { return 8 + k / 8; }

// selection memory per query of a tile
static size_t sel_bytes_per_query(size_t k, int sel) {
    return sel == 2 ? BUF_FACTOR * k * sizeof(Cand) : k * (sizeof(float) + sizeof(idx_t));
}

// force_sel = 1/2 sizes the tile for that selection method instead of the
// selector's choice (so forcing heap or buffer gives a fair configuration).
static SearchParams select_params_for(size_t nq, size_t nb, size_t d, size_t k,
                                      const MachineInfo& m, size_t mem_budget,
                                      int force_sel) {
    SearchParams p;
    p.nthreads = m.nthreads > 0 ? m.nthreads : 1;
    const int T = p.nthreads;

    // Database block: half of L2 so the block stays resident while every
    // query group of the tile streams over it (reuse factor tq / MR).
    size_t tb = (m.l2 / 2) / (sizeof(float) * std::max<size_t>(d, 1));
    tb = std::max<size_t>(2 * PANEL, tb / (2 * PANEL) * (2 * PANEL));
    tb = std::min(tb, round_up(std::max<size_t>(nb, 1), 2 * PANEL));
    p.tb = tb;

    // Top-k selection: the heap costs O(log k) per accepted candidate, the
    // buffer O(1) amortised but with a looser threshold. Buffer from k >=
    // SEL_BUF_MIN_K (measured on the i7-13650HX; E10 checks it).
    p.sel = k >= SEL_BUF_MIN_K ? 2 : 1;
    if (force_sel == 1 || force_sel == 2) p.sel = force_sel;

    // Query tile: about 64 queries (a multiple of MR) is enough to amortise one
    // L2 block, but keep the tile itself + its heaps within ~L1..L2/4.
    size_t tq = round_up(64, MR);
    while (tq > MR && tq * (d * sizeof(float) + sel_bytes_per_query(k, p.sel)) > m.l2 / 4)
        tq = std::max(MR, tq / 2 / MR * MR);
    tq = std::min(tq, round_up(std::max<size_t>(nq, 1), MR));
    p.tq = std::max(tq, MR);

    // Kernel height: MR, except for 1 or 2 queries, where padding up to MR rows
    // wastes most of the arithmetic. From 3 queries on, one padded MR-row pass
    // beats several 1/2-row passes over the block. (Measured on AVX2 only.)
    p.mr = int(MR);
#if KNN_AVX2_KERNEL
    if (nq <= 2) p.mr = int(std::max<size_t>(nq, 1));
#endif

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

SearchParams select_params(size_t nq, size_t nb, size_t d, size_t k,
                           const MachineInfo& m, size_t mem_budget) {
    return select_params_for(nq, nb, d, k, m, mem_budget, 0);
}

size_t tiled_working_bytes(size_t nq, size_t d, size_t k, const SearchParams& p) {
    if (p.strict) k += strict_extra(k);
    size_t per_thread = round_up(p.tq, MR) * (d + 1) * sizeof(float);
    if (p.sel == 2) per_thread += round_up(p.tq, MR) * sel_bytes_per_query(k, 2);
    size_t partial = p.db_splits > 1 ? size_t(p.db_splits) * nq * k * 12 : 0;
    return per_thread * size_t(p.nthreads) + partial;
}

// --------------------------------------------------------------- search ----
// ------------------------------------------------------------ strict ----
// Copies database row id out of the packed panels.
static void gather_row(const PackedDB& db, idx_t id, float* y) {
    const float* P = db.panels + size_t(id) / PANEL * db.d * PANEL + size_t(id) % PANEL;
    for (size_t t = 0; t < db.d; t++) y[t] = P[t * PANEL];
}

// Writes the m best of e[0..n) by (double distance, id) to D/I, pads to k.
static void write_exact(std::vector<std::pair<double, idx_t>>& e, size_t m, size_t k,
                        float* D, idx_t* I) {
    std::partial_sort(e.begin(), e.begin() + m, e.end());  // pair order = tie policy
    for (size_t r = 0; r < k; r++) {
        D[r] = r < m ? float(e[r].first) : INF;
        I[r] = r < m ? e[r].second : -1;
    }
}

// Strict mode for one query. cand[0..kk) are the K = k + extra float
// candidates, sorted, with final float distances; nc = rows that reached the
// merge, pdK = pd of the K-th candidate before ||x||^2 was added.
//
// eb bounds |float distance - true distance| for every row: the norm
// expansion in float has error <= about (2d + 4) u (||x||^2 + ||y||^2), u =
// 2^-24; (2d + 8) leaves a margin. Rows whose float distance exceeds
// hi = f_k + 3 eb (f_k = k-th float distance) are truly farther than the
// k-th by at least eb, so only rows with float distance <= hi can be in the
// reference's top k. They are all among the candidates if every row was
// merged (nc == nb) or the K-th candidate is already beyond hi; those are
// recomputed with exact_l2sq and ranked by (double, id), like the reference.
// Otherwise (e.g. more than K - k rows tie at the k-th distance) the query is
// rescanned in double. Returns 1 for a rescan, else 0.
static size_t strict_rerank(const PackedDB& db, const float* x, const Cand* cand, size_t kk,
                            size_t nc, size_t K, float pdK, float xn, size_t k,
                            std::vector<std::pair<double, idx_t>>& e, float* yrow,
                            float* D, idx_t* I) {
    const size_t d = db.d, nb = db.nb, m = std::min(k, nb);
    e.clear();
    if (m == 0) { write_exact(e, 0, k, D, I); return 0; }
    double xnd = 0;
    for (size_t t = 0; t < d; t++) xnd += double(x[t]) * x[t];
    const double eb = (2.0 * double(d) + 8.0) * 0x1p-24 * (xnd + db.max_norm);
    // every row of the database is among the candidates
    const bool all_rows = nc == nb && kk == nc;
    // pdK is only meaningful when the candidate list is full (kk == K); with
    // fewer, rows may have been dropped (e.g. NaN distances): rescan
    if (all_rows || (kk == K && kk >= m)) {
        const double hi = all_rows ? INFINITY : double(cand[m - 1].first) + 3 * eb;
        const bool complete = all_rows || double(std::max(0.f, pdK + xn)) > hi;
        if (complete) {
            for (size_t r = 0; r < kk; r++)
                if (double(cand[r].first) <= hi) {
                    gather_row(db, cand[r].second, yrow);
                    e.push_back({exact_l2sq(x, yrow, d), cand[r].second});
                }
            if (e.size() >= m) { write_exact(e, m, k, D, I); return 0; }
            e.clear();
        }
    }
    e.reserve(nb);  // full rescan in double
    for (size_t j = 0; j < nb; j++) {
        gather_row(db, idx_t(j), yrow);
        e.push_back({exact_l2sq(x, yrow, d), idx_t(j)});
    }
    write_exact(e, m, k, D, I);
    return 1;
}

// Per-thread buffers (query tile, selection buffers), kept between searches
// and only ever grown. Allocating and freeing them in every search made the
// process grow.
struct TileBuffer {
    void* p = nullptr;
    size_t bytes = 0;
    TileBuffer() = default;
    TileBuffer(const TileBuffer&) = delete;
    TileBuffer& operator=(const TileBuffer&) = delete;
    ~TileBuffer() { std::free(p); }
    void* get(size_t need) {
        if (need > bytes) {
            std::free(p);
            p = aligned_alloc64(need);
            bytes = p ? need : 0;
        }
        return p;
    }
};
static thread_local TileBuffer tile_buffer, cand_buffer;

SearchParams tiled_search(const PackedDB& db, const float* X, size_t nq,
                          size_t k, float* D, idx_t* I, SearchParams p) {
    const size_t d = db.d, nb = db.nb;
    MachineInfo m = MachineInfo::detect();
    if (p.nthreads > 0) m.nthreads = p.nthreads;
    // strict mode selects K = k + extra candidates; everything up to the
    // final rerank works on K
    const size_t K = p.strict ? k + strict_extra(k) : k;
    p.strict_rescans = 0;
    SearchParams auto_p = select_params_for(nq, nb, d, K, m, 64u << 20, p.sel);
    if (!p.tq) p.tq = auto_p.tq;
    if (!p.tb) p.tb = auto_p.tb;
    if (!p.db_splits) p.db_splits = auto_p.db_splits;
    if (!p.nthreads) p.nthreads = auto_p.nthreads;
    if (!p.mr) p.mr = auto_p.mr;
    if (p.mr != 1 && p.mr != 2) p.mr = int(MR);
    if (!p.sel) p.sel = auto_p.sel;
    if (p.sel != 2) p.sel = 1;
    const bool buf = p.sel == 2;
    p.tq = round_up(p.tq, MR);
    p.tb = std::max<size_t>(PANEL, round_up(p.tb, PANEL));
    if (nq == 0 || k == 0) return p;

    const size_t S = size_t(std::max(1, p.db_splits));
    const size_t nqt = (nq + p.tq - 1) / p.tq;
    const size_t np = db.npanels;
    const size_t bp = p.tb / PANEL;  // panels per block

    // Heap storage: with S == 1 the output arrays themselves are the heaps
    // (zero extra memory). With S > 1 (or strict mode's K > k) each chunk gets
    // its own nq x K heaps.
    std::vector<float> part_d;
    std::vector<idx_t> part_i;
    float* HD = D;
    idx_t* HI = I;
    if (S > 1 || K != k) {
        part_d.resize(S * nq * K);
        part_i.resize(S * nq * K);
        HD = part_d.data();
        HI = part_i.data();
    }

    size_t rescans = 0;
#pragma omp parallel num_threads(p.nthreads)
    {
        const size_t tqp = p.tq;
        float* Qt = (float*)tile_buffer.get(tqp * d * sizeof(float));
        Cand* cb = buf ? (Cand*)cand_buffer.get(tqp * BUF_FACTOR * K * sizeof(Cand)) : nullptr;
        std::vector<float> thr(tqp);
        std::vector<QSel> qs(tqp);

#pragma omp for schedule(dynamic, 1)
        for (size_t task = 0; task < nqt * S; task++) {
            size_t qt = task / S, s = task % S;
            size_t q0 = qt * tqp, nqt_rows = std::min(tqp, nq - q0);
            size_t p0 = np * s / S, p1 = np * (s + 1) / S;  // this chunk's panels

            // copy query tile (zero-pad to a multiple of MR) and set up heaps
            std::memcpy(Qt, X + q0 * d, nqt_rows * d * sizeof(float));
            std::memset(Qt + nqt_rows * d, 0, (tqp - nqt_rows) * d * sizeof(float));
            for (size_t r = 0; r < nqt_rows; r++) {
                qs[r].hd = HD + (s * nq + q0 + r) * K;
                qs[r].hi = HI + (s * nq + q0 + r) * K;
                qs[r].buf = buf ? cb + r * BUF_FACTOR * K : nullptr;
                qs[r].n = 0;
                if (!buf) heap_init(qs[r].hd, qs[r].hi, K);
                thr[r] = INF;
            }

            for (size_t b0 = p0; b0 < p1; b0 += bp) {        // L2 block of DB
                size_t b1 = std::min(p1, b0 + bp);
                QSel* q = qs.data();
                float* th = thr.data();
                if (buf) {
                    if (p.mr == 1) scan_block<true, 1>(db, Qt, nqt_rows, b0, b1, q, th, K);
                    else if (p.mr == 2) scan_block<true, 2>(db, Qt, nqt_rows, b0, b1, q, th, K);
                    else scan_block<true, int(MR)>(db, Qt, nqt_rows, b0, b1, q, th, K);
                } else {
                    if (p.mr == 1) scan_block<false, 1>(db, Qt, nqt_rows, b0, b1, q, th, K);
                    else if (p.mr == 2) scan_block<false, 2>(db, Qt, nqt_rows, b0, b1, q, th, K);
                    else scan_block<false, int(MR)>(db, Qt, nqt_rows, b0, b1, q, th, K);
                }
            }
            if (buf)  // keep the K best of each buffer as this chunk's result
                for (size_t r = 0; r < nqt_rows; r++) {
                    QSel& sq = qs[r];
                    if (sq.n > K) compact(sq.buf, sq.n, K), sq.n = K;
                    for (size_t j = 0; j < K; j++) {
                        sq.hd[j] = j < sq.n ? sq.buf[j].first : INF;
                        sq.hi[j] = j < sq.n ? sq.buf[j].second : -1;
                    }
                }
        }

        // ---- finalise: merge chunks (if any), add ||x||^2, sort by policy
        std::vector<std::pair<float, idx_t>> cand(S * K);
        std::vector<std::pair<double, idx_t>> exact;  // strict mode only
        std::vector<float> yrow(p.strict ? d : 0);
#pragma omp for schedule(static) reduction(+ : rescans)
        for (size_t i = 0; i < nq; i++) {
            size_t nc = 0;
            for (size_t s = 0; s < S; s++)
                for (size_t r = 0; r < K; r++) {
                    idx_t id = HI[(s * nq + i) * K + r];
                    if (id >= 0) cand[nc++] = {HD[(s * nq + i) * K + r], id};
                }
            size_t kk = std::min(K, nc);
            std::partial_sort(cand.begin(), cand.begin() + kk, cand.begin() + nc, by_policy);
            // every row not in cand[0..kk) has pd >= pdK (it lost to K others)
            const float pdK = kk == K ? cand[K - 1].first : INF;
            const float* x = X + i * d;
            float xn = 0;
            for (size_t t = 0; t < d; t++) xn += x[t] * x[t];
            for (size_t r = 0; r < kk; r++)
                cand[r].first = std::max(0.f, cand[r].first + xn);
            // adding ||x||^2 can create float ties: re-apply the policy
            std::sort(cand.begin(), cand.begin() + kk, by_policy);
            if (p.strict) {
                rescans += strict_rerank(db, x, cand.data(), kk, nc, K, pdK, xn, k, exact,
                                         yrow.data(), D + i * k, I + i * k);
                continue;
            }
            for (size_t r = 0; r < k; r++) {
                D[i * k + r] = r < kk ? cand[r].first : INF;
                I[i * k + r] = r < kk ? cand[r].second : -1;
            }
        }
    }
    p.strict_rescans = rescans;
    return p;
}

}  // namespace knn
