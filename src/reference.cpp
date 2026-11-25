// D1: sequential brute-force reference + exact-neighbour checker.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "knn.h"

namespace knn {

namespace {
// True squared distance in double precision, computed directly (no expansion,
// so no cancellation error).
inline double exact_l2sq(const float* a, const float* b, size_t d) {
    double s = 0;
    for (size_t t = 0; t < d; t++) {
        double diff = double(a[t]) - double(b[t]);
        s += diff * diff;
    }
    return s;
}

// Fills `order` with the ids of the k best (distance, id) pairs, sorted.
void topk_of_row(const std::vector<double>& dist, size_t k,
                 std::vector<idx_t>& order) {
    size_t nb = dist.size();
    order.resize(nb);
    std::iota(order.begin(), order.end(), idx_t(0));
    auto less = [&](idx_t a, idx_t b) {
        return dist[a] < dist[b] || (dist[a] == dist[b] && a < b);  // tie policy
    };
    size_t kk = std::min(k, nb);
    std::partial_sort(order.begin(), order.begin() + kk, order.end(), less);
    order.resize(kk);
}
}  // namespace

void reference_search(const float* X, size_t nq, const float* Y, size_t nb,
                      size_t d, size_t k, float* D, idx_t* I) {
    std::vector<double> dist(nb);
    std::vector<idx_t> order;
    for (size_t i = 0; i < nq; i++) {
        const float* x = X + i * d;
        for (size_t j = 0; j < nb; j++) dist[j] = exact_l2sq(x, Y + j * d, d);
        topk_of_row(dist, k, order);
        for (size_t r = 0; r < k; r++) {
            if (r < order.size()) {
                D[i * k + r] = float(dist[order[r]]);
                I[i * k + r] = order[r];
            } else {
                D[i * k + r] = std::numeric_limits<float>::infinity();
                I[i * k + r] = -1;
            }
        }
    }
}

std::string CheckReport::summary() const {
    std::ostringstream os;
    os << (passed() ? "PASS" : "FAIL") << "  checked=" << queries_checked
       << " tolerant_ok=" << tolerant_rows_ok << " strict_ok=" << strict_rows_ok
       << " recall@k=" << recall_at_k << " bad_ids=" << bad_ids
       << " wrong_nbrs=" << wrong_neighbours << " bad_dist=" << bad_distances
       << " unsorted=" << unsorted_rows;
    return os.str();
}

CheckReport check_results(const float* X, size_t nq, const float* Y, size_t nb,
                          size_t d, size_t k, const float* D, const idx_t* I,
                          const CheckOptions& opt) {
    CheckReport rep;
    // float32 norm-expansion error grows roughly like eps * sqrt(d) * scale.
    double rtol = opt.rtol >= 0 ? opt.rtol
                                : 16.0 * 1.2e-7 * std::sqrt(double(std::max<size_t>(d, 1)));
    double ymax = 0;
    for (size_t j = 0; j < nb; j++) {
        double s = 0;
        for (size_t t = 0; t < d; t++) s += double(Y[j * d + t]) * Y[j * d + t];
        ymax = std::max(ymax, s);
    }
    size_t stride = (opt.max_queries && nq > opt.max_queries) ? nq / opt.max_queries : 1;
    std::vector<double> dist(nb);
    std::vector<idx_t> ref;
    double recall_sum = 0;
    size_t kk = std::min(k, nb);

    for (size_t i = 0; i < nq; i += stride) {
        rep.queries_checked++;
        const float* x = X + i * d;
        double xn = 0;
        for (size_t t = 0; t < d; t++) xn += double(x[t]) * x[t];
        double tol = opt.atol + rtol * (xn + ymax);
        for (size_t j = 0; j < nb; j++) dist[j] = exact_l2sq(x, Y + j * d, d);
        topk_of_row(dist, k, ref);
        double kth = kk ? dist[ref[kk - 1]] : 0;

        const float* Dr = D + i * k;
        const idx_t* Ir = I + i * k;
        bool ok = true, strict = true;
        std::unordered_set<idx_t> seen;
        for (size_t r = 0; r < k; r++) {
            idx_t id = Ir[r];
            if (r >= kk) {  // padding expected
                if (id != -1) { rep.bad_ids++; ok = false; }
                continue;
            }
            if (id < 0 || size_t(id) >= nb || !seen.insert(id).second) {
                rep.bad_ids++; ok = false; strict = false; continue;
            }
            double td = dist[id];
            if (td > kth + tol) { rep.wrong_neighbours++; ok = false; }
            if (std::fabs(double(Dr[r]) - td) > tol) { rep.bad_distances++; ok = false; }
            if (r > 0 && double(Dr[r]) < double(Dr[r - 1]) - tol) { rep.unsorted_rows++; ok = false; }
            if (id != ref[r]) strict = false;
        }
        // set recall against the reference ids
        std::unordered_set<idx_t> refset(ref.begin(), ref.end());
        size_t hit = 0;
        for (size_t r = 0; r < kk; r++) hit += refset.count(Ir[r]);
        recall_sum += kk ? double(hit) / kk : 1.0;
        if (ok) rep.tolerant_rows_ok++;
        if (ok && strict) rep.strict_rows_ok++;
        if (!ok && opt.verbose)
            std::fprintf(stderr, "query %zu failed (kth=%g tol=%g)\n", i, kth, tol);
    }
    rep.recall_at_k = rep.queries_checked ? recall_sum / rep.queries_checked : 1;
    return rep;
}

}  // namespace knn
