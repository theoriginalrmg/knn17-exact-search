# E8: old vs new selector (over-splitting, tried and not kept; see results/i7-13650hx/e8_oversplit.md).
# Needs two knn_bench binaries and the E1 data (scripts/run_experiments.sh E1). From the
# project root, with the normal build in build/:
#   1. cp build/knn_bench /tmp/knn_old
#   2. paste results/i7-13650hx/e8_oversplit_rule.txt into select_params (src/tiled.cpp), just above
#      "while (S > 1 && size_t(S) * nq * k * 12 > mem_budget) S--;"
#   3. cmake --build build -j && ./build/knn_test && cp build/knn_bench /tmp/knn_new
#   4. remove the block again and rebuild
#   5. python3 scripts/e8_oversplit.py /tmp/knn_old /tmp/knn_new [ROUNDS=25] [SEED=99]
import os, shutil, subprocess, statistics as S, collections, random, sys, tempfile
if len(sys.argv) < 3: sys.exit("usage: e8_oversplit.py OLD_BENCH NEW_BENCH [ROUNDS] [SEED]")
OLD, NEW = map(os.path.abspath, sys.argv[1:3])
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
tmp = tempfile.TemporaryDirectory()
CTL = shutil.copy(OLD, os.path.join(tmp.name, "knn_ctl"))  # noise control: copy of OLD
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 25
random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 99)
B = {"old": OLD, "ctl": CTL, "new": NEW}
cells = [(nq, 10, T) for nq in (1, 4, 16, 64) for T in (6, 8, 14, 20)] + [(16, 1000, T) for T in (6, 8, 14, 20)] \
    + [(q, k, T) for q, k in ((16, 100), (64, 100), (4, 1000)) for T in (6, 8, 14, 20)] + [(256, 10, T) for T in (6, 8, 14, 20)] + [(1024, 10, T) for T in (14, 20)]
d = collections.defaultdict(list); par = {}
jobs = [(c, v) for c in cells for v in B]
for r in range(ROUNDS):
    random.shuffle(jobs)
    for c, v in jobs:
        nq, k, T = c
        o = subprocess.run(f"OMP_PROC_BIND=close OMP_PLACES=threads {B[v]} run --db data/u128_db.fbin --q data/u128_q.fbin --nq {nq} --k {k} --threads {T} --reps 5 --check {2 if r == 0 else 0}", shell=True, capture_output=True, text=True).stdout.strip().split(",")
        if r == 0: assert "PASS" in o[15], o
        d[c, v].append(float(o[9]) * 1e3); par[c, v] = o[7].split("S=")[1].split()[0]
def ratio(a, b):  # ratio of medians with bootstrap 90% interval
    f = lambda x, y: S.median(x) / S.median(y)
    n = len(a); bs = sorted(f([random.choice(a) for _ in a], [random.choice(b) for _ in b]) for _ in range(1000))
    return f(a, b), bs[50], bs[949]
print(f"{ROUNDS} rounds x 5 reps; ratio <1 = new faster; [90% bootstrap]")
print("nq k T | S old>new | old ms | ctl ms | new ms | new/old | control/old")
for c in cells:
    o, t, n = d[c, "old"], d[c, "ctl"], d[c, "new"]
    # both old copies are the same program: pool them for the baseline
    r = ratio(n, o + t); q = ratio(t, o)
    print(*c, "|", f"{par[c,'old']}>{par[c,'new']} | {S.median(o):.2f} | {S.median(t):.2f} | {S.median(n):.2f} | {r[0]:.3f} [{r[1]:.2f}-{r[2]:.2f}] | {q[0]:.3f} [{q[1]:.2f}-{q[2]:.2f}]")
