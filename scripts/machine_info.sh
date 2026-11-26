#!/usr/bin/env bash
# Prints a description of this machine and the code version, for the
# machine.txt in a results folder. Usage: scripts/machine_info.sh > results/X/machine.txt
cd "$(dirname "$0")/.."
cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
cores=$(lscpu | sed -n 's/^Core(s) per socket:[[:space:]]*//p' | head -1)
sockets=$(lscpu | sed -n 's/^Socket(s):[[:space:]]*//p' | head -1)
tpc=$(lscpu | sed -n 's/^Thread(s) per core:[[:space:]]*//p' | head -1)
l2=$(getconf LEVEL2_CACHE_SIZE 2>/dev/null || echo 0)
ram=$(awk '/MemTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo)
os=$(. /etc/os-release && echo "$PRETTY_NAME")
kernel=$(uname -r)
case "$kernel" in *microsoft*|*WSL*) env="WSL2 (core types and host pinning are hidden)";; *) env="native Linux";; esac
gxx=$(g++ --version 2>/dev/null | head -1)
cmk=$(cmake --version 2>/dev/null | head -1)
py=$(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo "not found")
np=$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "not installed")
fa=$(python3 -c 'import faiss; print(faiss.__version__)' 2>/dev/null || echo "not installed")
commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=$(git status --porcelain --untracked-files=no 2>/dev/null | grep -q . && echo " + uncommitted changes" || true)
cat <<EOF
cpu: $cpu
cores/threads seen by the OS: ${sockets:-?} socket(s) x ${cores:-?} cores x ${tpc:-?} threads = $(nproc) logical CPUs
l2 (getconf): $(( l2 / 1024 )) KiB
ram: $ram visible
os: $os, kernel $kernel, $env
compiler: $gxx; $cmk
python: $py, numpy $np, faiss-cpu $fa
git commit: $commit$dirty
EOF
