#!/usr/bin/env bash
# Short commands for the demo. Run from the repo folder.
#   ./knn.sh test                 run all correctness tests
#   ./knn.sh ours  [nq] [k] [t]   our search  (defaults: 5 queries, k=10, 8 threads)
#   ./knn.sh faiss [nq] [k] [t]   Faiss, same search
#   ./knn.sh show  [query]        print the k neighbours of one query (default 0)
#   ./knn.sh compare              do our ids match Faiss and the exact reference?
set -e
DB=data/u128_db.fbin; Q=data/u128_q.fbin
NQ=${2:-5}; K=${3:-10}; T=${4:-8}
export OMP_PROC_BIND=close OMP_PLACES=threads
cols(){ cut -d, -f1,4,6,7,8,10,16; }
case "$1" in
  test)  ./build/knn_test | tail -1 ;;
  ours)  echo "impl,queries,k,threads,selector choice,seconds,check"
         ./build/knn_bench run --db $DB --q $Q --nq $NQ --k $K --threads $T --save-res ours.res | cols ;;
  faiss) echo "impl,queries,k,threads,method,seconds"
         python3 bench/faiss_bench.py --db $DB --q $Q --nq $NQ --k $K --threads $T --save-res faiss.res | cut -d, -f1,4,6,7,8,10 ;;
  show)  [ -f ours.res ] || ./build/knn_bench run --db $DB --q $Q --nq 5 --k 10 --threads 8 --save-res ours.res >/dev/null
         python3 scripts/show_results.py ours.res --queries ${2:-0} ;;
  compare)
         ./build/knn_bench run --db $DB --q $Q --nq 5 --k 10 --threads 8 --reps 1 --save-res ours.res >/dev/null
         python3 bench/faiss_bench.py --db $DB --q $Q --nq 5 --k 10 --reps 1 --save-res faiss.res >/dev/null
         ./build/knn_bench run --impl ref --db $DB --q $Q --nq 5 --k 10 --reps 1 --check 0 --save-res ref.res >/dev/null
         python3 scripts/show_results.py ours.res --queries --vs faiss.res | tail -2
         python3 scripts/show_results.py ours.res --queries --vs ref.res | tail -2 ;;
  *) sed -n 2,7p "$0" ;;
esac
