"""Faiss baseline: CPU exact flat L2 search (IndexFlatL2), same data files,
same CSV columns as knn_bench. Only index.search is timed; index.add is the
preprocessing step (copying vectors into the index)."""
import argparse, resource, sys, time
import numpy as np
import faiss

def read_fbin(p):
    with open(p, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)

def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize()

ap = argparse.ArgumentParser()
ap.add_argument("--db"); ap.add_argument("--q"); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--nq", type=int, default=0); ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--label", default="")
ap.add_argument("--save-res", default="")
a = ap.parse_args()

if a.threads: faiss.omp_set_num_threads(a.threads)
T = faiss.omp_get_max_threads()
Y = read_fbin(a.db); X = read_fbin(a.q)
if a.nq: X = np.ascontiguousarray(X[: a.nq])
nq, d = X.shape; nb = Y.shape[0]; k = a.k
t0 = time.perf_counter(); index = faiss.IndexFlatL2(d); index.add(Y); pre = time.perf_counter() - t0
D = np.empty((nq, k), np.float32); I = np.empty((nq, k), np.int64)
rss0 = rss(); peak0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
index.search(X, k, D=D, I=I)                      # warm-up (inside memory window)
ts = []
for _ in range(a.reps):
    s = time.perf_counter(); index.search(X, k, D=D, I=I); ts.append(time.perf_counter() - s)
peak1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
ts.sort(); med = ts[len(ts) // 2]; sd = float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0
path = "blas" if nq * d >= faiss.cvar.distance_compute_blas_threshold else "seq"
work = 4096 * 1024 * 4 if path == "blas" else 0
delta = max(0, peak1 - max(rss0, peak0))
print(f"faiss,{a.label},{nb},{nq},{d},{k},{T},IndexFlatL2 path={path},{pre:.6f},{med:.6f},{ts[0]:.6f},"
      f"{sd:.6f},{nq/med:.1f},{delta/1048576:.2f},{work/1024:.1f},see-check")
if a.save_res:
    with open(a.save_res, "wb") as f:
        np.array([nq, k], np.int32).tofile(f); D.tofile(f); I.tofile(f)
