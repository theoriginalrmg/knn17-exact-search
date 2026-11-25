# Project 17 — Memory-efficient exact nearest-neighbour search (CS F422)

Exact Euclidean top-k over dense float32 vectors on a multicore CPU with OpenMP,
built on the exact-search ideas of Johnson, Douze & Jégou, *Billion-scale
Similarity Search with GPUs* (arXiv 2017), and compared against Faiss's CPU
exact flat L2 search (`IndexFlatL2`).

```
include/knn.h            public API + output contract (tie policy)
src/reference.cpp        D1  sequential brute-force reference + exact-neighbour checker
src/tiled.cpp            D2  OpenMP tiled distances + per-query top-k heaps
                         D3  tile / parallelism selector (select_params)
src/data.cpp             3 input families, .fbin / .fvecs I/O
tests/test_knn.cpp       hand-checked cases, checker self-test, 120 randomized differential tests
bench/knn_bench.cpp      benchmark driver (gen / run / check), CSV output
bench/faiss_bench.py     Faiss IndexFlatL2 baseline with identical CSV columns
scripts/run_experiments.sh   reproduces every result (E1–E6)
scripts/analyze.py       results/*.csv -> results/summary.md + figures
results/                 raw CSVs, summary.md, fig_e1_latency.png, fig_e5_selector.png
```

## Build & run

```bash
# deps: g++ >= 9 (OpenMP), cmake >= 3.16; for the baseline: pip install numpy faiss-cpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/knn_test                         # all correctness tests
scripts/run_experiments.sh               # E1..E6 (~15 min on 2 cores); or e.g. "scripts/run_experiments.sh E1"
python3 scripts/analyze.py               # tables + plots
# sanitizer build: cmake -S . -B build-asan -DKNN_SANITIZE=ON && cmake --build build-asan && ./build-asan/knn_test
```

Versions used for the included results: Ubuntu 24.04, g++ 13.3, cmake 3.28,
Python 3.13, numpy, faiss-cpu 1.16.0 (pip wheel, bundles OpenBLAS 0.3.15),
2-vCPU Intel Xeon @ 2.1 GHz (AVX-512), 7 GiB RAM, a shared cloud VM (so timings
are noisy; we report median ± std over 5 reps).

## Input/output contract

* Input: database `Y` (nb × d), queries `X` (nq × d), float32 row-major; `k ≥ 1`.
* Output: `D[nq×k]` squared L2 distances, `I[nq×k]` int64 ids.
* **Tie policy:** rows are sorted by *(distance ascending, id ascending)*; when two
  vectors are equally far, the **smaller id wins**, both for membership in the
  top-k and for order. If nb < k the row is padded with `(+inf, -1)`.

## What each deliverable does

### D1 — reference + checker (`src/reference.cpp`)
`reference_search` computes every distance in **double precision** directly as
Σ(x−y)² (no norm expansion → no cancellation error), then `partial_sort`s with
the tie policy. It is slow on purpose (≈ 50 ms/query at nb=200k, d=128) — it's
ground truth.

`check_results` decides if any result is exact. Floating-point means the fast
code can legitimately disagree with the reference on near-ties, so there are two
levels:
* **tolerant** (pass/fail): every id valid and unique; each returned id's true
  distance ≤ k-th true distance + tol; each reported distance = true distance ± tol;
  row sorted. `tol = 16·ε₃₂·√d·(‖x‖² + max‖y‖²)`.
* **strict**: ids identical to the reference (tie policy applied identically).
  For the integer-valued `ties` family all float maths is exact, so the tests
  *require* strict equality there.

The checker is itself tested: it must reject duplicate ids, out-of-range ids, a
far point swapped in, a wrong distance, and an unsorted row.

### D2 — OpenMP tiled distances + per-query top-k (`src/tiled.cpp`)
1. **Norm expansion (paper §5.1):** ‖x−y‖² = ‖x‖² + ‖y‖² − 2⟨x,y⟩. ‖y‖² is computed
   once at pack time; ‖x‖² is constant per query so selection runs on
   `‖y‖² − 2⟨x,y⟩` and ‖x‖² is added at the end (exactly the paper's trick).
2. **Packed database (preprocessing, timed separately):** rows grouped into
   panels of 16, stored dimension-major inside a panel, so one 64-byte load
   gives dimension t of 16 database rows. Padding rows get ‖y‖² = +∞ so they never win.
3. **Register-blocked micro-kernel:** 4 queries × 32 database rows → 8 vector
   accumulators; each step does 2 loads and 8 broadcast-FMAs (the inner loop of
   a GEMM). Uses GCC/Clang vector extensions → AVX-512 or 2×AVX2 automatically.
4. **Two-level tiling:** a database block of `tb` rows (sized to ½ L2) is kept hot
   while every 4-query group of a `tq`-query tile streams over it, so each
   database element fetched from memory is reused `tq` times.
5. **Fused selection from registers:** the paper fuses k-selection with the
   kernel consuming the GEMM output so the nq×nb matrix is never written. We go
   further: distances are consumed straight from the micro-kernel registers into
   per-query **bounded max-heaps** (root = current worst). A 16-lane vector
   compare against the heap threshold skips whole panels; only survivors touch
   the heap. **No distance tile is ever stored.**
6. **Exact tie handling in the hot path:** within a task ids are scanned in
   increasing order, so a candidate equal to the threshold must lose → the
   strict `<` filter is exact. After adding ‖x‖² the row is re-sorted by the policy
   (adding ‖x‖² can create new float ties).
7. **Parallelism:** tasks = (query tile) × (database chunk), `schedule(dynamic)`.
   With S = 1 chunk the output arrays *are* the heaps (zero extra memory). With
   S > 1 each chunk keeps its own heaps and a per-query merge picks the final k.

### D3 — tile / parallelism selector (`select_params`)
* `tb` = rows such that `tb·d·4 B ≈ L2/2` (read from `sysconf`).
* `tq` = 64 (shrunk when the tile + heaps exceed L2/4, and when nq is small).
* `S` (database splits): smallest S giving ≥ T tasks with ≤ 25% load imbalance,
  limited by blocks per chunk and a partial-heap memory budget (64 MiB).
  → **small batches become database-parallel** (each thread scans nb/S rows —
  the only way to use all cores for 1 query), **large batches become
  query-parallel** (no merge, no extra memory).

## Results (from `results/summary.md`)

![E1](results/fig_e1_latency.png)

* All runs of both libraries pass the checker (32 sampled queries per run).
* **Small/medium batches (nq ≤ 256), where Faiss 1.16 uses its per-pair
  sequential path:** 1.8–8.8× faster with 1 thread (4 ≤ nq ≤ 256) and 2.3–6.6× with 2 threads (1 ≤ nq ≤ 256).
  At nq = 1, 2 threads: 4.1 ms vs 10.8 ms. Faiss only parallelizes over queries,
  so its single query gets *slower* with 2 threads (0.88×); our db-parallel
  split gives 2.43×.
* **Large batches (BLAS path):** 2.5–3.1× faster, and Faiss's search grows peak
  RSS by **~20 MiB** (its 4096×1024 float `ip_block` + heaps) vs **≤ 80 KiB** working
  memory for ours (S=1 needs only a 64-query tile copy per thread).
* **Selector (E5):** picks the right strategy (S=2 for nq ≤ 64, S=1 for nq ≥ 256)
  every time; within 1.12–1.31× of the best of 27 grid configs, while the worst
  config is 2.5–3.7× slower. Part of the gap is VM noise (the selector's own
  config shows up as "best" in the grid for nq=1 and 256).
* **Unfavourable cases:** nq=1 single-threaded is a tie with Faiss (0.96×): with
  one query there is no reuse, so both are memory-bandwidth bound. k=1000 at
  nq=1024 drops to 1.32×: heap updates (O(log k) each) start to dominate the
  fused kernel.
* Scaling 1→2 threads: 1.5–2.4× (super-linear at nq=1 is noise + the halved scan).

**Fairness notes.** Same machine, same data files, same thread count via
`OMP_NUM_THREADS`/`omp_set_num_threads`, only `search` timed (Faiss `add` and
our packing are reported as preprocessing), outputs preallocated for both.
Faiss here is the pip wheel with OpenBLAS 0.3.15; an MKL build of Faiss would
likely be faster on the BLAS path, so we **don't claim general superiority** —
our claim is for the constrained regime the brief suggests: **small/medium
batches and a tight working-memory budget.** Faiss's current `main` adds a
database-parallel path for nq < threads that 1.16 doesn't use here; that would
narrow the nq=1 gap.

## Suggested team split (3 people)
* Baseline/correctness: D1, checker, tests, tie policy, data families.
* Parallel implementation: packing, micro-kernel, heaps, OpenMP tasks, merge.
* Evaluation: D3 selector + ablation, Faiss harness, experiments, plots, report.

## Ideas for the remaining weeks / report
* Run on a real dataset: SIFT1M via `read_fvecs` (`src/data.cpp`).
* Test on a 4–8 core laptop to get the 1,2,4,8-thread scaling curve.
* Replace the heap with a small sorted buffer / two-pass threshold for k ≥ 256.
* Calibrate the selector's crossover empirically at install time.
* Profile with `perf stat` (FLOP rate vs peak, cache misses) for the bottleneck section.
