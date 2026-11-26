#!/usr/bin/env bash
# Reproduces every table/figure. Usage: scripts/run_experiments.sh [-o DIR] [E1 E2 ...]
# Results go to DIR (default results/<hostname>), with a machine.txt describing
# this machine and a log line per run. Then: python3 scripts/analyze.py DIR
# Requires: build/ (cmake), python3 with numpy + faiss-cpu.
set -euo pipefail
cd "$(dirname "$0")/.."
B=./build/knn_bench
F="python3 bench/faiss_bench.py"
R=results/$(hostname)
if [ "${1:-}" = "-o" ]; then R=${2:?usage: scripts/run_experiments.sh [-o DIR] [E1 E2 ...]}; shift 2; fi
mkdir -p data "$R"
[ -f "$R/machine.txt" ] || scripts/machine_info.sh > "$R/machine.txt"
REPS=${REPS:-5}
MAXT=$(nproc)
# E1 thread counts; those above nproc are skipped (i7-13650HX: 20, i9-14900HX: 32)
THREADS="1 2 4 6 8 12 14 20 24 32"
# E2-E4 use 14 threads: with all 20, timings are too noisy
T14=$(( MAXT < 14 ? MAXT : 14 ))
# same pinning for knn_bench and faiss_bench.py (both use OpenMP)
export OMP_PROC_BIND=close OMP_PLACES=threads

gen() {  # family n d seed file
  [ -f "data/$5" ] || $B gen --family "$1" --n "$2" --d "$3" --seed "$4" --out "data/$5"
}
both() {  # csv label db q nq k T
  local out=$1 label=$2 db=$3 q=$4 nq=$5 k=$6 T=$7
  $B run --db "data/$db" --q "data/$q" --nq "$nq" --k "$k" --threads "$T" --reps "$REPS" --check 32 --label "$label" | tee -a "$out"
  # run Faiss first, THEN check its saved result (must be sequential)
  local row chk
  rm -f /tmp/faiss.res
  row=$($F --db "data/$db" --q "data/$q" --nq "$nq" --k "$k" --threads "$T" --reps "$REPS" --label "$label" --save-res /tmp/faiss.res)
  chk=$($B check --db "data/$db" --q "data/$q" --nq "$nq" --k "$k" --res /tmp/faiss.res --check 32 | awk '{print $1}')
  echo "${row/see-check/$chk}" | tee -a "$out"
}
hdr() { $B header > "$1"; }

E=${*:-E1 E2 E3 E4 E5 E6 E7 E9 E10 E11}
echo "run $(date '+%Y-%m-%d %H:%M') experiments: $E; git $(git rev-parse --short HEAD)$(git status --porcelain --untracked-files=no | grep -q . && echo ' + uncommitted changes')" >> "$R/machine.txt"

# E1: query batch size x threads (main result) — uniform, nb=200k, d=128, k=10
if [[ " $E " == *" E1 "* ]]; then
  gen uniform 200000 128 1 u128_db.fbin; gen uniform 4096 128 2 u128_q.fbin
  hdr $R/e1_batch.csv
  for T in $THREADS; do [ "$T" -le "$MAXT" ] || continue; for nq in 1 4 16 64 256 1024 4096; do
    both $R/e1_batch.csv batch u128_db.fbin u128_q.fbin $nq 10 $T; done; done
fi

# E2: dimension — nb=100k, k=10, 14 threads
if [[ " $E " == *" E2 "* ]]; then
  hdr $R/e2_dim.csv
  for d in 32 128 512; do
    gen uniform 100000 $d 3 u${d}_db100k.fbin; gen uniform 1024 $d 4 u${d}_q1k.fbin
    for nq in 1 64 1024; do both $R/e2_dim.csv d$d u${d}_db100k.fbin u${d}_q1k.fbin $nq 10 $T14; done
  done
fi

# E3: k — uniform nb=200k d=128, 14 threads
if [[ " $E " == *" E3 "* ]]; then
  hdr $R/e3_k.csv
  for k in 1 10 100 1000; do for nq in 16 1024; do
    both $R/e3_k.csv k$k u128_db.fbin u128_q.fbin $nq $k $T14; done; done
fi

# E4: input families (clustered, ties) — nb=200k d=128 k=10, 14 threads
if [[ " $E " == *" E4 "* ]]; then
  hdr $R/e4_family.csv
  for fam in clustered ties; do
    gen $fam 200000 128 5 ${fam}_db.fbin; gen $fam 1024 128 6 ${fam}_q.fbin
    for nq in 1 64 1024; do both $R/e4_family.csv $fam ${fam}_db.fbin ${fam}_q.fbin $nq 10 $T14; done
  done
fi

# E5: D3 ablation — every (S, tq, tb) on a grid vs the selector's choice
if [[ " $E " == *" E5 "* ]]; then
  hdr $R/e5_selector.csv
  for nq in 1 4 16 64 256 1024; do
    $B run --db data/u128_db.fbin --q data/u128_q.fbin --nq $nq --k 10 --threads $MAXT --reps $REPS --check 0 --label auto | tee -a $R/e5_selector.csv
    for S in 1 2 4 8 16 20 24 32; do [ "$S" -le "$MAXT" ] || continue; for tq in 4 16 64; do for tb in 256 2048 16384; do
      $B run --db data/u128_db.fbin --q data/u128_q.fbin --nq $nq --k 10 --threads $MAXT --reps 3 --check 0 \
         --S $S --tq $tq --tb $tb --label grid >> $R/e5_selector.csv
    done; done; done
  done
fi

# E6: sequential reference cost (to anchor speedups) on a subset of queries
if [[ " $E " == *" E6 "* ]]; then
  hdr $R/e6_reference.csv
  for nq in 1 16; do
    $B run --impl ref --db data/u128_db.fbin --q data/u128_q.fbin --nq $nq --k 10 --reps 3 --check 0 --label ref | tee -a $R/e6_reference.csv
  done
fi

# E7: is nq=1 limited by memory bandwidth? STREAM-style read vs our nq=1 scan,
# same threads and pinning. 1024 MiB = far above L3; 99 MiB = packed db size.
# "incache": 1 thread on a 2000-row db (1 MiB, fits L2) = speed without RAM. One search takes
# ~13 us, so knn_bench prints times in ns (older CSVs rounded them to whole us, +-7% here).
if [[ " $E " == *" E7 "* ]]; then
  gen uniform 200000 128 1 u128_db.fbin; gen uniform 4096 128 2 u128_q.fbin
  gen uniform 2000 128 1 u128_db2k.fbin
  ./build/bandwidth header > $R/e7_bandwidth.csv
  hdr $R/e7_scan.csv
  Q="--q data/u128_q.fbin --nq 1 --k 10 --check 1"
  for round in 1 2 3 4 5; do
    for T in $THREADS; do [ "$T" -le "$MAXT" ] || continue
      ./build/bandwidth 1024 $T 11 | tee -a $R/e7_bandwidth.csv
      ./build/bandwidth 99 $T 41 | tee -a $R/e7_bandwidth.csv
      $B run --db data/u128_db.fbin $Q --threads $T --reps 41 --label scan | tee -a $R/e7_scan.csv
    done
    $B run --db data/u128_db2k.fbin $Q --threads 1 --reps 501 --label incache | tee -a $R/e7_scan.csv
  done
fi

# E9: SIFT1M (real data: nb=1M, d=128), 14 threads, both libraries checked.
# Needs data/sift/sift_base.fvecs and sift_query.fvecs (http://corpus-texmex.irisa.fr/).
if [[ " $E " == *" E9 "* ]]; then
  if [ -f data/sift/sift_base.fvecs ] && [ -f data/sift/sift_query.fvecs ]; then
    hdr $R/e9_sift.csv
    for nq in 1 64 1024; do
      both $R/e9_sift.csv sift1m sift/sift_base.fvecs sift/sift_query.fvecs $nq 10 $T14
    done
  else
    echo "E9 skipped: data/sift/sift_base.fvecs or data/sift/sift_query.fvecs not found"
  fi
fi

# E10: top-k selection, heap vs buffer (+ nth_element compaction) vs the
# selector's choice; uniform nb=200k d=128, 14 threads.
if [[ " $E " == *" E10 "* ]]; then
  gen uniform 200000 128 1 u128_db.fbin; gen uniform 4096 128 2 u128_q.fbin
  hdr $R/e10_select.csv
  for k in 1 10 100 1000; do for nq in 16 1024; do for sel in heap buf auto; do
    $B run --db data/u128_db.fbin --q data/u128_q.fbin --nq $nq --k $k --threads $T14 --reps $REPS \
       --check 32 --sel $sel --label $sel | tee -a $R/e10_select.csv
  done; done; done
fi

# E11: strict mode (results identical to the double-precision reference) on vs
# off; nb=200k d=128 k=10, 14 threads; the check column also counts rows whose
# ids equal the reference exactly.
if [[ " $E " == *" E11 "* ]]; then
  gen uniform 200000 128 1 u128_db.fbin; gen uniform 4096 128 2 u128_q.fbin
  for fam in clustered ties; do gen $fam 200000 128 5 ${fam}_db.fbin; gen $fam 1024 128 6 ${fam}_q.fbin; done
  hdr $R/e11_strict.csv
  for fam in uniform clustered ties; do
    if [ $fam = uniform ]; then db=u128_db.fbin; q=u128_q.fbin; else db=${fam}_db.fbin; q=${fam}_q.fbin; fi
    for nq in 1 64 1024; do for st in 0 1; do
      $B run --db data/$db --q data/$q --nq $nq --k 10 --threads $T14 --reps $REPS --check 32 \
         --strict $st --label ${fam}_strict$st | tee -a $R/e11_strict.csv
    done; done
  done
fi
