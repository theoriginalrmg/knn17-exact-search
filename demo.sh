#!/usr/bin/env bash
# Live demo: ./demo.sh   (press Enter between steps)
set -e
pause(){ echo; read -rp "--- press Enter for the next step ---"; echo; }
echo "== 1. Correctness tests";                        ./knn.sh test;      pause
echo "== 2. One query (selector: db-parallel)";        ./knn.sh ours 1;    pause
echo "== 3. 4096 queries (selector: query-parallel)";  ./knn.sh ours 4096; pause
echo "== 4. Same 4096 queries in Faiss";               ./knn.sh faiss 4096; pause
echo "== 5. The 10 neighbours of query 0";             ./knn.sh ours 5 >/dev/null; ./knn.sh show 0; pause
echo "== 6. Same ids as Faiss and the exact reference?"; ./knn.sh compare
