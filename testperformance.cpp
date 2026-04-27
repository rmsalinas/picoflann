#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <numeric>

#include <opencv2/flann.hpp>

#include "picoflann.h"

// ---------------------------------------------------------------------------
// Point type and picoflann adapter
// ---------------------------------------------------------------------------

template<int DIMS>
struct Point {
    float data[DIMS];
};

template<int DIMS>
struct PointAdapter {
    inline float operator()(const Point<DIMS>& p, int d) const { return p.data[d]; }
};

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;
using Msec  = std::chrono::duration<double, std::milli>;

static inline double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration_cast<Msec>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

template<int DIMS>
std::vector<Point<DIMS>> random_points(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
    std::vector<Point<DIMS>> pts(n);
    for (auto& p : pts)
        for (int d = 0; d < DIMS; ++d)
            p.data[d] = dist(rng);
    return pts;
}

template<int DIMS>
cv::Mat points_to_mat(const std::vector<Point<DIMS>>& pts) {
    cv::Mat m(static_cast<int>(pts.size()), DIMS, CV_32F);
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
        std::memcpy(m.ptr<float>(i), pts[i].data, DIMS * sizeof(float));
    return m;
}

// ---------------------------------------------------------------------------
// Result structure
// ---------------------------------------------------------------------------

struct BenchResult {
    double build_ms  = 0.0;
    double search_ms = 0.0;
};

// ---------------------------------------------------------------------------
// picoflann benchmark (exact search)
// ---------------------------------------------------------------------------

template<int DIMS>
BenchResult bench_picoflann(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    int knn, int burn_runs, int measure_iters)
{
    using Tree = picoflann::KdTreeIndex<DIMS, PointAdapter<DIMS>>;

    for (int b = 0; b < burn_runs; ++b) {
        Tree tree;
        tree.build(pts);
        volatile size_t sink = 0;
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); sink += r.size(); }
        (void)sink;
    }

    double total_build = 0.0;
    for (int i = 0; i < measure_iters; ++i) {
        Tree tree;
        auto t0 = Clock::now();
        tree.build(pts);
        total_build += elapsed_ms(t0);
    }

    Tree tree;
    tree.build(pts);

    for (int b = 0; b < burn_runs; ++b) {
        volatile size_t sink = 0;
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); sink += r.size(); }
        (void)sink;
    }

    double total_search = 0.0;
    for (int i = 0; i < measure_iters; ++i) {
        volatile size_t sink = 0;
        auto t0 = Clock::now();
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); sink += r.size(); }
        total_search += elapsed_ms(t0);
        (void)sink;
    }

    return { total_build / measure_iters, total_search / measure_iters };
}

// ---------------------------------------------------------------------------
// OpenCV FLANN benchmark
// checks=-1 means exact (FLANN_CHECKS_UNLIMITED), checks=32 means approximate
// ---------------------------------------------------------------------------

template<int DIMS>
BenchResult bench_opencv(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    int knn, int burn_runs, int measure_iters, int checks)
{
    cv::Mat data  = points_to_mat<DIMS>(pts);
    cv::Mat qmat  = points_to_mat<DIMS>(queries);
    cv::Mat idx_out(static_cast<int>(queries.size()), knn, CV_32S);
    cv::Mat dst_out(static_cast<int>(queries.size()), knn, CV_32F);
    cv::flann::SearchParams sp(checks);

    for (int b = 0; b < burn_runs; ++b) {
        cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));
        fi.knnSearch(qmat, idx_out, dst_out, knn, sp);
    }

    double total_build = 0.0;
    for (int i = 0; i < measure_iters; ++i) {
        auto t0 = Clock::now();
        cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));
        total_build += elapsed_ms(t0);
    }

    cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));

    for (int b = 0; b < burn_runs; ++b)
        fi.knnSearch(qmat, idx_out, dst_out, knn, sp);

    double total_search = 0.0;
    for (int i = 0; i < measure_iters; ++i) {
        auto t0 = Clock::now();
        fi.knnSearch(qmat, idx_out, dst_out, knn, sp);
        total_search += elapsed_ms(t0);
    }

    return { total_build / measure_iters, total_search / measure_iters };
}

// ---------------------------------------------------------------------------
// Pretty-print
// ---------------------------------------------------------------------------

static void print_header(int dims, int nqueries, int knn, int burn, int iters) {
    std::cout << "\n=== " << dims << "D  |  "
              << nqueries << " queries  |  KNN=" << knn
              << "  |  burn=" << burn << "  |  iters=" << iters << " ===\n\n";

    const int W = 14;
    std::cout << std::left
        << std::setw(10) << "N"
        << std::setw(W)  << "pico(ms)"
        << std::setw(W)  << "cv-approx(ms)"
        << std::setw(10) << "vs pico"
        << std::setw(W)  << "cv-exact(ms)"
        << std::setw(10) << "vs pico"
        << "\n"
        << std::string(72, '-') << "\n";
}

static void print_subheader(const char* label) {
    std::cout << " -- " << label << " --\n";
}

static void print_row(size_t n,
    double pico, double cv_approx, double cv_exact)
{
    std::cout << std::left << std::fixed << std::setprecision(2)
        << std::setw(10) << n
        << std::setw(14) << pico
        << std::setw(14) << cv_approx
        << std::setw(10) << cv_approx / pico
        << std::setw(14) << cv_exact
        << std::setw(10) << cv_exact / pico
        << "\n";
}

// ---------------------------------------------------------------------------
// Run benchmark series
// ---------------------------------------------------------------------------

template<int DIMS>
void run_benchmark(
    const std::vector<size_t>& sizes,
    int nqueries, int knn, int burn, int iters)
{
    auto queries = random_points<DIMS>(nqueries, /*seed=*/1234);
    print_header(DIMS, nqueries, knn, burn, iters);

    std::vector<BenchResult> rpico, rapprox, rexact;

    for (size_t n : sizes) {
        auto pts = random_points<DIMS>(n);
        rpico  .push_back(bench_picoflann<DIMS>(pts, queries, knn, burn, iters));
        rapprox.push_back(bench_opencv<DIMS>   (pts, queries, knn, burn, iters, 32));
        rexact .push_back(bench_opencv<DIMS>   (pts, queries, knn, burn, iters, -1));
    }

    print_subheader("BUILD");
    for (size_t i = 0; i < sizes.size(); ++i)
        print_row(sizes[i], rpico[i].build_ms, rapprox[i].build_ms, rexact[i].build_ms);

    std::cout << "\n";
    print_subheader("SEARCH  (ratio = cv / pico, higher = picoflann faster)");
    for (size_t i = 0; i < sizes.size(); ++i)
        print_row(sizes[i], rpico[i].search_ms, rapprox[i].search_ms, rexact[i].search_ms);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const int KNN      = 10;
    const int NQUERIES = 1000;
    const int BURN     = 3;
    const int ITERS    = 5;

    const std::vector<size_t> SIZES = { 10000, 50000, 100000, 500000 };

    std::cout << "picoflann (exact) vs OpenCV FLANN approximate (32 checks) and exact (-1 checks)\n";
    std::cout << "ratio > 1 means picoflann is faster\n";

    run_benchmark<2>(SIZES, NQUERIES, KNN, BURN, ITERS);
    run_benchmark<3>(SIZES, NQUERIES, KNN, BURN, ITERS);
    run_benchmark<4>(SIZES, NQUERIES, KNN, BURN, ITERS);

    return 0;
}
