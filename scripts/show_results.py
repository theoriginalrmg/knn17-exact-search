"""Print the k nearest neighbours from a saved result file; optionally compare with another."""
import argparse
import numpy as np

def load(path):
    with open(path, "rb") as f:
        nq, k = np.fromfile(f, np.int32, 2)
        D = np.fromfile(f, np.float32, nq * k).reshape(nq, k)
        I = np.fromfile(f, np.int64, nq * k).reshape(nq, k)
    return D, I

ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("--queries", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--vs")
a = ap.parse_args()
D, I = load(a.res)
print(f"{a.res}: {D.shape[0]} queries, k = {D.shape[1]}\n")
for q in a.queries:
    print(f"query {q}:")
    print("  rank  id        squared distance")
    for r in range(D.shape[1]):
        print(f"  {r+1:>4}  {I[q, r]:<8}  {D[q, r]:.4f}")
    print()
if a.vs:
    D2, I2 = load(a.vs)
    n = min(len(I), len(I2))
    same = int((I[:n] == I2[:n]).all(axis=1).sum())
    print(f"vs {a.vs}: {same}/{n} queries have identical neighbour ids")
    print(f"largest distance difference: {np.abs(D[:n] - D2[:n]).max():.2e}")
