// Correctness tests:
//  1. hand-checkable cases (known answers, explicit ties, k > nb)
//  2. checker self-test (it must reject deliberately broken results)
//  3. randomized differential tests: tiled_search vs reference_search over
//     random shapes x all three data families x forced strategies / tiles /
//     thread counts.
#include <omp.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/data.h"
#include "knn.h"

using namespace knn;

static int failures = 0;
#define EXPECT(cond, msg)                                              \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); \
            failures++;                                                \
        }                                                              \
    } while (0)

static void run_tiled(const std::vector<float>& Y, size_t nb, const std::vector<float>& X,
                      size_t nq, size_t d, size_t k, SearchParams p,
                      std::vector<float>& D, std::vector<idx_t>& I) {
    PackedDB db;
    pack_database(Y.data(), nb, d, db);
    D.assign(nq * k, -1);
    I.assign(nq * k, -7);
    tiled_search(db, X.data(), nq, k, D.data(), I.data(), p);
}

static void test_hand_checked() {
    std::printf("[hand-checked cases]\n");
    // 1-D database: ids 0..5 at positions 0,1,2,3,4,5 ; query at 2.5
    // -> distances .25(2) .25(3) 2.25(1) 2.25(4) 6.25(0) 6.25(5)
    // tie policy (smaller id first): 2,3,1,4,0,5
    std::vector<float> Y = {0, 1, 2, 3, 4, 5}, X = {2.5f};
    std::vector<float> D(6);
    std::vector<idx_t> I(6);
    reference_search(X.data(), 1, Y.data(), 6, 1, 6, D.data(), I.data());
    idx_t want[6] = {2, 3, 1, 4, 0, 5};
    float wantd[6] = {.25f, .25f, 2.25f, 2.25f, 6.25f, 6.25f};
    for (int r = 0; r < 6; r++) EXPECT(I[r] == want[r] && D[r] == wantd[r], "reference 1-D ties");
    run_tiled(Y, 6, X, 1, 1, 6, {}, D, I);
    for (int r = 0; r < 6; r++) EXPECT(I[r] == want[r] && D[r] == wantd[r], "tiled 1-D ties");

    // k > nb: padding with (+inf, -1)
    run_tiled(Y, 6, X, 1, 1, 8, {}, D, I);
    EXPECT(I[6] == -1 && I[7] == -1 && std::isinf(D[6]), "padding when k > nb");

    // exact duplicates: all database rows identical -> ids 0..k-1 in order
    std::vector<float> Yd(40 * 3, 1.f), Xd = {0, 0, 0};
    run_tiled(Yd, 40, Xd, 1, 3, 5, {}, D, I);
    for (int r = 0; r < 5; r++) EXPECT(I[r] == r && D[r] == 3.f, "all-duplicate database");
    // same thing, forced db-parallel with 3 chunks: merge must keep policy
    SearchParams p; p.db_splits = 3; p.tb = 16; p.nthreads = 3;
    run_tiled(Yd, 40, Xd, 1, 3, 5, p, D, I);
    for (int r = 0; r < 5; r++) EXPECT(I[r] == r, "duplicates across db chunks");

    // query equal to a database vector -> distance 0, that id first
    std::vector<float> Y2 = make_family("uniform", 100, 8, 1);
    std::vector<float> X2(Y2.begin() + 37 * 8, Y2.begin() + 38 * 8);
    run_tiled(Y2, 100, X2, 1, 8, 3, {}, D, I);
    EXPECT(I[0] == 37 && D[0] < 1e-5f, "self-query finds itself");
}

static void test_checker_rejects() {
    std::printf("[checker self-test]\n");
    size_t nb = 200, nq = 10, d = 16, k = 5;
    auto Y = make_family("uniform", nb, d, 3), X = make_family("uniform", nq, d, 4);
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    reference_search(X.data(), nq, Y.data(), nb, d, k, D.data(), I.data());
    auto ok = check_results(X.data(), nq, Y.data(), nb, d, k, D.data(), I.data());
    EXPECT(ok.passed() && ok.strict_rows_ok == nq, "reference passes its own checker");

    auto I2 = I; I2[3] = I2[4];  // duplicate id
    EXPECT(!check_results(X.data(), nq, Y.data(), nb, d, k, D.data(), I2.data()).passed(), "duplicate id rejected");
    I2 = I; I2[0] = nb + 5;      // out of range
    EXPECT(!check_results(X.data(), nq, Y.data(), nb, d, k, D.data(), I2.data()).passed(), "bad id rejected");
    I2 = I;                       // replace a true neighbour by the farthest point
    {
        std::vector<float> Dall(nb);
        std::vector<idx_t> Iall(nb);
        reference_search(X.data(), 1, Y.data(), nb, d, nb, Dall.data(), Iall.data());
        I2[k - 1] = Iall[nb - 1];
        auto D2 = D; D2[k - 1] = Dall[nb - 1];
        EXPECT(!check_results(X.data(), nq, Y.data(), nb, d, k, D2.data(), I2.data()).passed(), "wrong neighbour rejected");
    }
    auto D2 = D; D2[2] *= 1.5f;  // wrong distance
    EXPECT(!check_results(X.data(), nq, Y.data(), nb, d, k, D2.data(), I.data()).passed(), "wrong distance rejected");
    D2 = D; std::swap(D2[0], D2[4]); auto I3 = I; std::swap(I3[0], I3[4]);
    EXPECT(!check_results(X.data(), nq, Y.data(), nb, d, k, D2.data(), I3.data()).passed(), "unsorted row rejected");
}

static void test_randomized() {
    std::printf("[randomized differential tests]\n");
    std::mt19937_64 rng(2026);
    const char* fams[3] = {"uniform", "clustered", "ties"};
    int cases = 0, strict_total = 0, rows_total = 0;
    for (int it = 0; it < 120; it++) {
        size_t nb = 1 + rng() % 700, nq = 1 + rng() % 70, d = 1 + rng() % 140;
        size_t k = 1 + rng() % 40;
        std::string fam = fams[it % 3];
        auto Y = make_family(fam, nb, d, rng()), X = make_family(fam, nq, d, rng());
        SearchParams p;
        switch (it % 5) {  // exercise the selector AND forced configurations
            case 0: break;                                              // auto
            case 1: p.db_splits = 1; p.nthreads = 4; break;              // query-par
            case 2: p.db_splits = 1 + rng() % 5; p.tb = 16 * (1 + rng() % 4); p.nthreads = 3; break;
            case 3: p.tq = 4 * (1 + rng() % 8); p.tb = 32; p.nthreads = 2; break;
            case 4: p.db_splits = 2; p.tq = 4; p.nthreads = 1; break;    // serial merge
        }
        const int mrs[4] = {0, 1, 2, 99};  // selector, 1-row, 2-row, full kernel
        p.mr = mrs[(it / 15) % 4];
        std::vector<float> D;
        std::vector<idx_t> I;
        run_tiled(Y, nb, X, nq, d, k, p, D, I);
        auto rep = check_results(X.data(), nq, Y.data(), nb, d, k, D.data(), I.data());
        cases++;
        strict_total += rep.strict_rows_ok;
        rows_total += rep.queries_checked;
        if (!rep.passed()) {
            std::printf("  case %d fam=%s nb=%zu nq=%zu d=%zu k=%zu %s\n   %s\n", it,
                        fam.c_str(), nb, nq, d, k, p.describe().c_str(), rep.summary().c_str());
            failures++;
        }
        // integer data => float arithmetic is exact => tie policy must match
        // the reference exactly, not just "within tolerance"
        if (fam == "ties" && rep.strict_rows_ok != rep.queries_checked) {
            std::printf("  case %d: strict tie-policy mismatch (%zu/%zu)\n", it,
                        rep.strict_rows_ok, rep.queries_checked);
            failures++;
        }
    }
    std::printf("  %d cases, %d/%d rows strictly identical to reference\n", cases,
                strict_total, rows_total);
}

// Tiny batches: every kernel height, query-parallel and split, strict ties.
static void test_tiny_batches() {
    std::printf("[tiny batches]\n");
    const size_t nb = 333, d = 17, k = 7;
    auto Y = make_family("ties", nb, d, 11);
    for (size_t nq = 1; nq <= 8; nq++)
        for (int mr : {0, 1, 2, 99})
            for (int S : {0, 1, 3}) {
                auto X = make_family("ties", nq, d, 100 + nq);
                SearchParams p; p.mr = mr; p.db_splits = S; p.tb = 32; p.nthreads = 2;
                std::vector<float> D;
                std::vector<idx_t> I;
                run_tiled(Y, nb, X, nq, d, k, p, D, I);
                auto rep = check_results(X.data(), nq, Y.data(), nb, d, k, D.data(), I.data());
                EXPECT(rep.passed() && rep.strict_rows_ok == nq, "tiny batch strict match");
            }
}

static void test_selector() {
    std::printf("[selector sanity]\n");
    MachineInfo m; m.l1d = 48 << 10; m.l2 = 2 << 20; m.nthreads = 8;
    auto small = select_params(1, 1000000, 128, 10, m);
    auto large = select_params(10000, 1000000, 128, 10, m);
    EXPECT(small.db_splits >= 8, "nq=1 should split the database across all threads");
    EXPECT(large.db_splits == 1, "large batch should be query-parallel");
    std::printf("  nq=1     -> %s\n  nq=10000 -> %s\n", small.describe().c_str(), large.describe().c_str());
}

int main() {
    test_hand_checked();
    test_checker_rejects();
    test_randomized();
    test_tiny_batches();
    test_selector();
    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
