#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>

#include "knn.h"

namespace knn {

double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

size_t current_rss_bytes() {
    long pages = 0, rss = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
    std::fclose(f);
    return size_t(rss) * size_t(sysconf(_SC_PAGESIZE));
}

size_t peak_rss_bytes() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return size_t(ru.ru_maxrss) * 1024;  // Linux reports KiB
}

}  // namespace knn
