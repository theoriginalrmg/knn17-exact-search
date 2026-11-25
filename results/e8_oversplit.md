# E8: over-splitting the database in the selector (tried, not kept)

**Result:** giving each thread about 4 database chunks makes nq = 64 at 8 threads
19–24% faster, but it is more than 5% slower in other cases, so it was removed.
The selector is unchanged.

**How to reproduce (version 2):** `python3 scripts/e8_oversplit.py OLD_BENCH NEW_BENCH`, where
NEW_BENCH is `knn_bench` built with `e8_oversplit_rule.txt` applied; build steps are at the top
of the script. For version 1, apply its line instead and add `15 7`; that run did not include
the k = 100 cases or nq = 4 at k = 1000.

## Motivation

At nq = 64 (k = 10) the batch is one query tile, and the selector picks S = T: one
equal chunk per thread. Timing each chunk at 8 threads (instrumented copy, `close`
binding, 200 searches per row, medians):

| T | S | total ms | slowest / fastest chunk | summed busy ms | time threads are busy |
|---|---|---|---|---|---|
| 1 | 1 | 29.98 | 1.00 | 30.0 | 100% |
| 6 | 6 | 7.95 | 1.32 | 40.3 | 85% |
| 6 | 24 | 8.50 | 1.91 | 45.6 | 91% |
| 8 | 8 | 9.02 | 1.84 | 51.2 | 74% |
| 8 | 32 | 8.52 | 2.49 | 60.0 | 90% |

Busy % is summed busy time / (T × time the last chunk finished).
Equal chunks do not take equal time, so the search waits for the slowest.
Binding (`close`, `spread`, unpinned) made no meaningful difference at S = 8. The total
busy time also grows with threads, which splitting cannot fix.

## Method

- Laptop: i7-13650HX, WSL2 Ubuntu 22.04, g++ 11.4, AVX2 kernels, nb = 200k, d = 128,
  uniform data, `OMP_PROC_BIND=close OMP_PLACES=threads`.
- Each round runs every case with three binaries in shuffled order, 5 reps each; a run's
  value is the median of its reps, and the tables show the median over rounds.
- *Control* is a second copy of the old binary (measures noise). `new/old` uses both
  copies of the old binary as its baseline; the interval is a bootstrap 90% interval.
- **Rule set beforehand:** keep the change only if it is never more than about 5% worse
  anywhere and clearly better at nq = 64 with 8 threads. "worse"/"better" in the last
  column mark new/old above 1.05 / below 0.95.
- All results passed the exactness checker (2 queries per case, first round; just the
  one query when nq = 1), and `knn_test` passed (normal and sanitizer builds) for both
  versions.

## Version 1: about 4 tasks per thread whenever the selector splits

```cpp
if (S > 1) S = std::max(S, std::min(smax, int((4 * size_t(T) + nqt - 1) / nqt)));
```

15 rounds. **Rejected:** 33–46% slower at k = 1000, and no clear gain with several query
tiles (nq >= 256).

| nq | k | T | S old → new | old ms | control ms | new ms | new/old [90% interval] | control/old | |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 6 | 6 → 24 | 2.83 | 2.78 | 2.86 | 1.019 [0.89-1.13] | 0.980 | |
| 1 | 10 | 8 | 8 → 32 | 3.00 | 2.98 | 2.78 | 0.931 [0.88-0.95] | 0.991 | better |
| 1 | 10 | 14 | 14 → 56 | 2.89 | 2.97 | 2.96 | 1.010 [0.99-1.13] | 1.027 | |
| 1 | 10 | 20 | 20 → 80 | 3.01 | 3.06 | 2.99 | 0.983 [0.93-1.01] | 1.019 | |
| 4 | 10 | 6 | 6 → 24 | 2.98 | 2.80 | 2.73 | 0.961 [0.91-1.01] | 0.940 | |
| 4 | 10 | 8 | 8 → 32 | 3.30 | 3.30 | 2.80 | 0.850 [0.83-0.93] | 0.998 | better |
| 4 | 10 | 14 | 14 → 56 | 3.14 | 3.17 | 2.93 | 0.928 [0.87-0.95] | 1.009 | better |
| 4 | 10 | 20 | 20 → 80 | 2.96 | 3.05 | 2.92 | 0.968 [0.92-1.02] | 1.030 | |
| 16 | 10 | 6 | 6 → 24 | 3.59 | 3.51 | 3.65 | 1.033 [0.96-1.16] | 0.980 | |
| 16 | 10 | 8 | 8 → 32 | 4.29 | 4.38 | 3.65 | 0.840 [0.82-0.89] | 1.021 | better |
| 16 | 10 | 14 | 14 → 56 | 3.84 | 3.94 | 3.72 | 0.947 [0.90-0.99] | 1.025 | better |
| 16 | 10 | 20 | 20 → 80 | 4.03 | 5.02 | 4.52 | 1.069 [0.80-1.34] | 1.247 | **worse** |
| 64 | 10 | 6 | 6 → 24 | 8.97 | 7.61 | 8.19 | 0.940 [0.86-1.06] | 0.849 | better |
| 64 | 10 | 8 | 8 → 32 | 10.18 | 10.06 | 8.13 | 0.806 [0.71-0.86] | 0.989 | better |
| 64 | 10 | 14 | 14 → 56 | 7.35 | 7.60 | 6.81 | 0.922 [0.87-0.95] | 1.033 | better |
| 64 | 10 | 20 | 20 → 80 | 6.69 | 6.72 | 6.84 | 1.020 [0.94-1.10] | 1.005 | |
| 16 | 1000 | 6 | 3 → 12 | 9.87 | 9.36 | 13.48 | 1.395 [1.34-1.46] | 0.948 | **worse** |
| 16 | 1000 | 8 | 4 → 16 | 9.68 | 10.02 | 13.05 | 1.326 [1.27-1.36] | 1.036 | **worse** |
| 16 | 1000 | 14 | 7 → 28 | 9.17 | 9.09 | 12.35 | 1.353 [1.32-1.39] | 0.992 | **worse** |
| 16 | 1000 | 20 | 10 → 40 | 8.57 | 8.56 | 12.51 | 1.461 [1.39-1.60] | 0.999 | **worse** |
| 256 | 10 | 6 | 3 → 6 | 26.89 | 26.89 | 28.23 | 1.050 [1.03-1.13] | 1.000 | |
| 256 | 10 | 8 | 2 → 8 | 29.18 | 29.02 | 28.04 | 0.963 [0.91-1.00] | 0.994 | |
| 256 | 10 | 14 | 6 → 14 | 23.36 | 23.72 | 24.28 | 1.032 [0.99-1.06] | 1.015 | |
| 256 | 10 | 20 | 5 → 20 | 22.36 | 21.26 | 21.74 | 1.018 [0.94-1.06] | 0.951 | |
| 1024 | 10 | 14 | 3 → 4 | 86.15 | 86.09 | 86.78 | 1.008 [0.98-1.02] | 0.999 | |
| 1024 | 10 | 20 | 2 → 5 | 73.95 | 72.62 | 79.83 | 1.089 [0.97-1.23] | 0.982 | **worse** |

## Version 2: one query tile only, and at least 200·k rows per chunk

Code in `e8_oversplit_rule.txt`. 25 rounds, new random order, added cases: nq = 16 and
64 at k = 100, and nq = 4 at k = 1000. **Rejected:** fixes k = 1000, but is worse than
the 5% limit in three cases, all at 6 threads.

| nq | k | T | S old → new | old ms | control ms | new ms | new/old [90% interval] | control/old | |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 6 | 6 → 24 | 2.62 | 2.68 | 2.74 | 1.029 [0.97-1.05] | 1.025 | |
| 1 | 10 | 8 | 8 → 32 | 3.12 | 3.00 | 2.80 | 0.920 [0.88-0.95] | 0.964 | better |
| 1 | 10 | 14 | 14 → 56 | 2.95 | 2.94 | 2.87 | 0.974 [0.95-1.01] | 0.998 | |
| 1 | 10 | 20 | 20 → 80 | 3.09 | 3.04 | 3.15 | 1.026 [0.96-1.08] | 0.983 | |
| 4 | 10 | 6 | 6 → 24 | 2.78 | 2.94 | 2.83 | 1.011 [0.93-1.03] | 1.059 | |
| 4 | 10 | 8 | 8 → 32 | 3.11 | 3.26 | 2.80 | 0.874 [0.85-0.92] | 1.049 | better |
| 4 | 10 | 14 | 14 → 56 | 3.19 | 3.13 | 2.97 | 0.946 [0.92-0.97] | 0.982 | better |
| 4 | 10 | 20 | 20 → 80 | 3.05 | 3.03 | 2.88 | 0.944 [0.91-1.09] | 0.994 | better |
| 16 | 10 | 6 | 6 → 24 | 3.60 | 3.54 | 3.71 | 1.041 [0.96-1.12] | 0.983 | |
| 16 | 10 | 8 | 8 → 32 | 4.21 | 4.30 | 3.70 | 0.866 [0.85-0.89] | 1.020 | better |
| 16 | 10 | 14 | 14 → 56 | 3.94 | 3.92 | 3.65 | 0.927 [0.90-0.96] | 0.992 | better |
| 16 | 10 | 20 | 20 → 80 | 3.99 | 4.22 | 3.80 | 0.916 [0.86-0.99] | 1.058 | better |
| 64 | 10 | 6 | 6 → 24 | 7.64 | 7.69 | 8.39 | 1.095 [1.04-1.15] | 1.007 | **worse** |
| 64 | 10 | 8 | 8 → 32 | 9.91 | 9.84 | 7.45 | 0.756 [0.72-0.81] | 0.993 | better |
| 64 | 10 | 14 | 14 → 56 | 7.45 | 7.47 | 6.76 | 0.906 [0.88-0.95] | 1.003 | better |
| 64 | 10 | 20 | 20 → 80 | 6.81 | 6.66 | 6.65 | 0.994 [0.93-1.02] | 0.978 | |
| 16 | 1000 | 6 | 3 → 3 | 9.64 | 9.71 | 9.81 | 1.014 [0.98-1.05] | 1.008 | |
| 16 | 1000 | 8 | 4 → 4 | 10.00 | 9.96 | 10.05 | 1.007 [0.95-1.03] | 0.997 | |
| 16 | 1000 | 14 | 7 → 7 | 9.07 | 9.27 | 9.13 | 1.004 [0.97-1.03] | 1.022 | |
| 16 | 1000 | 20 | 10 → 10 | 8.88 | 8.44 | 8.56 | 0.987 [0.95-1.03] | 0.950 | |
| 16 | 100 | 6 | 6 → 10 | 4.07 | 4.10 | 4.55 | 1.114 [1.06-1.15] | 1.006 | **worse** |
| 16 | 100 | 8 | 8 → 10 | 4.90 | 4.79 | 4.33 | 0.895 [0.86-0.96] | 0.977 | better |
| 16 | 100 | 14 | 14 → 14 | 4.34 | 4.25 | 4.25 | 0.992 [0.97-1.05] | 0.978 | |
| 16 | 100 | 20 | 20 → 20 | 4.55 | 4.39 | 4.41 | 0.987 [0.91-1.01] | 0.966 | |
| 64 | 100 | 6 | 6 → 10 | 10.33 | 10.66 | 12.91 | 1.226 [1.19-1.30] | 1.032 | **worse** |
| 64 | 100 | 8 | 8 → 10 | 12.47 | 12.49 | 12.85 | 1.029 [1.01-1.07] | 1.001 | |
| 64 | 100 | 14 | 14 → 14 | 10.17 | 10.12 | 10.11 | 0.997 [0.98-1.04] | 0.995 | |
| 64 | 100 | 20 | 20 → 20 | 9.46 | 9.50 | 9.55 | 1.008 [0.97-1.12] | 1.004 | |
| 4 | 1000 | 6 | 6 → 6 | 3.78 | 3.88 | 3.88 | 1.021 [0.96-1.07] | 1.026 | |
| 4 | 1000 | 8 | 8 → 8 | 4.18 | 4.16 | 4.09 | 0.981 [0.96-1.00] | 0.995 | |
| 4 | 1000 | 14 | 14 → 14 | 4.00 | 4.01 | 4.03 | 1.005 [0.98-1.03] | 1.004 | |
| 4 | 1000 | 20 | 20 → 20 | 4.36 | 4.45 | 4.25 | 0.974 [0.94-1.08] | 1.022 | |
| 256 | 10 | 6 | 3 → 3 | 26.74 | 26.64 | 26.81 | 1.005 [0.97-1.03] | 0.996 | |
| 256 | 10 | 8 | 2 → 2 | 29.39 | 30.14 | 29.41 | 0.992 [0.97-1.04] | 1.026 | |
| 256 | 10 | 14 | 6 → 6 | 25.18 | 24.46 | 25.20 | 1.007 [0.96-1.04] | 0.971 | |
| 256 | 10 | 20 | 5 → 5 | 21.41 | 21.71 | 21.73 | 1.012 [0.96-1.06] | 1.014 | |
| 1024 | 10 | 14 | 3 → 3 | 87.89 | 87.11 | 87.13 | 1.000 [0.99-1.01] | 0.991 | |
| 1024 | 10 | 20 | 2 → 2 | 73.67 | 77.19 | 74.59 | 0.990 [0.95-1.07] | 1.048 | |

## Why it fails

- **k = 1000 (version 1).** Every chunk fills its own k-entry heap per query. Smaller
  chunks mean more heap updates and more partial results to merge.
- **6 threads, k = 100 (version 2: +11% at nq = 16, +23% at nq = 64).** The 200·k limit
  caps S at 10. Ten equal chunks cannot be shared evenly by 6 threads: with equal
  chunk times, some threads do two and others one. This is a flaw of that limit.
- **6 threads, k = 10, nq = 64 (version 2: +10%).** There is little imbalance to remove
  (slowest/fastest chunk 1.32, against 1.84 at 8 threads), and total busy time rose from
  40.3 to 45.6 ms with S = 24, so the extra chunks mostly add cost. This case is not
  stable between runs: version 1 also chose S = 24 and measured 6% *faster* (interval
  14% faster to 6% slower).

These explanations fit the measurements, but only the chunk times and busy times above
were measured directly; heap and merge time were not timed separately.

## Limits

- One laptop and one data family. WSL hides which virtual CPUs are P- or E-cores, and
  Windows can move them, so why threads run at unequal speed was not identified.
- A rule such as "only above 6 threads" would pass these tests, but it would be fitted
  to this machine and these measurements, so it was not adopted.
