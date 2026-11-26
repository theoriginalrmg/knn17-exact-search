// Benchmark / tooling driver.
//
//   knn_bench gen   --family uniform|clustered|ties --n N --d D --seed S --out f.fbin
// --db / --q accept .fbin or .fvecs (SIFT/GIST format, chosen by extension).
//
//   knn_bench run   --db db.fbin --q q.fbin --k K [--impl tiled|ref] [--threads T]
//                   [--S s --tq q --tb b --mr m --sel heap|buf --strict 0|1] [--reps R] [--check M] [--label L]
//                   [--save-res out.res]
//   knn_bench check --db db.fbin --q q.fbin --k K --res result.res [--check M]
//
// `run` prints one CSV row (see HEADER). Peak memory is measured as the growth
// of the process high-water mark (ru_maxrss) during the timed searches, on top
// of the RSS after data load + preprocessing, i.e. the search's working memory.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/data.h"
#include "knn.h"

using namespace knn;

static const char* HEADER =
    "impl,label,nb,nq,d,k,threads,params,preprocess_s,median_s,min_s,std_s,"
    "qps,peak_delta_MiB,working_KiB,check";

static std::map<std::string, std::string> parse(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
    return a;
}
static std::string get(std::map<std::string, std::string>& a, const char* key, const char* def) {
    return a.count(key) ? a[key] : std::string(def);
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// .fvecs by extension, everything else as .fbin
static std::vector<float> load_vectors(const std::string& path, size_t& n, size_t& d) {
    auto v = ends_with(path, ".fvecs") ? read_fvecs(path, n, d) : read_fbin(path, n, d);
    if (n == 0) throw std::runtime_error("no vectors in " + path);
    return v;
}

static void save_res(const std::string& path, size_t nq, size_t k, const float* D, const idx_t* I) {
    FILE* f = std::fopen(path.c_str(), "wb");
    int32_t h[2] = {int32_t(nq), int32_t(k)};
    std::fwrite(h, sizeof h, 1, f);
    std::fwrite(D, sizeof(float), nq * k, f);
    std::fwrite(I, sizeof(idx_t), nq * k, f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: knn_bench gen|run|check|header ...\n");
        return 1;
    }
    std::string cmd = argv[1];
    auto a = parse(argc, argv, 2);

    if (cmd == "header") { std::puts(HEADER); return 0; }

    if (cmd == "gen") {
        size_t n = std::stoul(get(a, "n", "1000")), d = std::stoul(get(a, "d", "128"));
        auto v = make_family(get(a, "family", "uniform"), n, d, std::stoull(get(a, "seed", "1")));
        write_fbin(get(a, "out", "out.fbin"), v.data(), n, d);
        return 0;
    }

    size_t nb, d, nq, d2;
    auto Y = load_vectors(get(a, "db", ""), nb, d);
    auto X = load_vectors(get(a, "q", ""), nq, d2);
    if (d != d2) { std::fprintf(stderr, "dimension mismatch\n"); return 1; }
    if (a.count("nq")) {  // use only the first nq queries
        nq = std::min(nq, size_t(std::stoul(a["nq"])));
        X.resize(nq * d);
    }
    size_t k = std::stoul(get(a, "k", "10"));
    size_t ncheck = std::stoul(get(a, "check", "64"));

    if (cmd == "check") {
        FILE* f = std::fopen(get(a, "res", "").c_str(), "rb");
        int32_t h[2];
        if (!f || std::fread(h, sizeof h, 1, f) != 1) { std::fprintf(stderr, "bad res\n"); return 1; }
        std::vector<float> D(size_t(h[0]) * h[1]);
        std::vector<idx_t> I(D.size());
        if (std::fread(D.data(), 4, D.size(), f) != D.size() ||
            std::fread(I.data(), 8, I.size(), f) != I.size()) { std::fprintf(stderr, "short res\n"); return 1; }
        std::fclose(f);
        CheckOptions o; o.max_queries = ncheck;
        auto rep = check_results(X.data(), std::min<size_t>(nq, h[0]), Y.data(), nb, d, k, D.data(), I.data(), o);
        std::puts(rep.summary().c_str());
        return rep.passed() ? 0 : 2;
    }

    // ------------------------------------------------------------ run ----
    std::string impl = get(a, "impl", "tiled");
    int T = std::stoi(get(a, "threads", "0"));
    if (T > 0) omp_set_num_threads(T);
    T = omp_get_max_threads();
    int reps = std::stoi(get(a, "reps", "5"));

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    std::memset(D.data(), 0, D.size() * 4);
    std::memset(I.data(), 0, I.size() * 8);

    PackedDB db;
    SearchParams p;
    p.nthreads = T;
    p.db_splits = std::stoi(get(a, "S", "0"));
    p.tq = std::stoul(get(a, "tq", "0"));
    p.tb = std::stoul(get(a, "tb", "0"));
    p.mr = std::stoi(get(a, "mr", "0"));
    p.strict = get(a, "strict", "0") == "1";
    {
        std::string sel = get(a, "sel", "auto");
        p.sel = sel == "heap" ? 1 : sel == "buf" ? 2 : 0;
    }
    double t0 = now_sec();
    if (impl == "tiled") pack_database(Y.data(), nb, d, db);
    double pre = now_sec() - t0;

    auto once = [&]() {
        if (impl == "tiled") p = tiled_search(db, X.data(), nq, k, D.data(), I.data(), p);
        else reference_search(X.data(), nq, Y.data(), nb, d, k, D.data(), I.data());
    };
    // memory baseline taken BEFORE the first search so that the first call's
    // allocations (heaps, partial buffers, thread stacks) are visible
    size_t rss0 = current_rss_bytes();
    size_t peak0 = peak_rss_bytes();
    once();  // warm-up (also resolves auto params so the timed runs reuse them)
    std::vector<double> ts;
    for (int r = 0; r < reps; r++) {
        double s = now_sec();
        once();
        ts.push_back(now_sec() - s);
    }
    size_t peak1 = peak_rss_bytes();
    double peak_delta = double(peak1 > std::max(rss0, peak0) ? peak1 - std::max(rss0, peak0) : 0);
    // Also report the analytical working set (what the code allocates), since
    // RSS is page-granular and noisy for small buffers.
    size_t work = impl == "tiled" ? tiled_working_bytes(nq, d, k, p) : nb * sizeof(double);

    std::sort(ts.begin(), ts.end());
    double med = ts[ts.size() / 2], mean = 0, var = 0;
    for (double t : ts) mean += t;
    mean /= ts.size();
    for (double t : ts) var += (t - mean) * (t - mean);
    double sd = ts.size() > 1 ? std::sqrt(var / (ts.size() - 1)) : 0;

    std::string chk = "skipped";
    if (ncheck > 0) {
        CheckOptions o; o.max_queries = ncheck;
        auto rep = check_results(X.data(), nq, Y.data(), nb, d, k, D.data(), I.data(), o);
        chk = std::string(rep.passed() ? "PASS" : "FAIL") + "(" + std::to_string(rep.tolerant_rows_ok) +
              "/" + std::to_string(rep.queries_checked) + ") strict " +
              std::to_string(rep.strict_rows_ok) + "/" + std::to_string(rep.queries_checked);
    }
    if (a.count("save-res")) save_res(a["save-res"], nq, k, D.data(), I.data());

    std::string params = impl == "tiled" ? p.describe() : std::string("sequential");
    std::printf("%s,%s,%zu,%zu,%zu,%zu,%d,%s,%.6f,%.9f,%.9f,%.9f,%.1f,%.2f,%.1f,%s\n", impl.c_str(),
                get(a, "label", "").c_str(), nb, nq, d, k, impl == "ref" ? 1 : T, params.c_str(), pre, med,
                ts.front(), sd, nq / med, peak_delta / 1048576.0, work / 1024.0, chk.c_str());
    return 0;
}
