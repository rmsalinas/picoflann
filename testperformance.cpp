#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <numeric>

#include <opencv2/flann.hpp>

#include "picoflann.h"
#include "picoflann2.h"

// ---------------------------------------------------------------------------
// Shared point type and adapters (same for both libraries)
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
// Result
// ---------------------------------------------------------------------------

struct BenchResult {
    double build_ms  = 0.0;
    double search_ms = 0.0;
};

// ---------------------------------------------------------------------------
// picoflann v1 benchmark
// ---------------------------------------------------------------------------

template<int DIMS>
BenchResult bench_picoflann(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    int knn, int burn, int iters)
{
    using Tree = picoflann::KdTreeIndex<DIMS, PointAdapter<DIMS>>;

    for (int b = 0; b < burn; ++b) {
        Tree tree; tree.build(pts);
        volatile size_t s = 0;
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); s += r.size(); }
        (void)s;
    }

    double tb = 0;
    for (int i = 0; i < iters; ++i) {
        Tree tree; auto t0 = Clock::now(); tree.build(pts); tb += elapsed_ms(t0);
    }

    Tree tree; tree.build(pts);
    for (int b = 0; b < burn; ++b) {
        volatile size_t s = 0;
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); s += r.size(); }
        (void)s;
    }

    double ts = 0;
    for (int i = 0; i < iters; ++i) {
        volatile size_t s = 0; auto t0 = Clock::now();
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); s += r.size(); }
        ts += elapsed_ms(t0); (void)s;
    }
    return { tb / iters, ts / iters };
}

// ---------------------------------------------------------------------------
// picoflann v2 benchmark — Scalar is float or double
// ---------------------------------------------------------------------------

template<int DIMS, typename Scalar>
BenchResult bench_picoflann2_scalar(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    int knn, int burn, int iters)
{
    using Tree = picoflann2::KdTreeIndex<DIMS, PointAdapter<DIMS>, picoflann2::L2, Scalar>;

    for (int b = 0; b < burn; ++b) {
        Tree tree; tree.build(pts);
        volatile size_t s = 0;
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); s += r.size(); }
        (void)s;
    }

    double tb = 0;
    for (int i = 0; i < iters; ++i) {
        Tree tree; auto t0 = Clock::now(); tree.build(pts); tb += elapsed_ms(t0);
    }

    Tree tree; tree.build(pts);
    for (int b = 0; b < burn; ++b) {
        volatile size_t s = 0;
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); s += r.size(); }
        (void)s;
    }

    double ts = 0;
    for (int i = 0; i < iters; ++i) {
        volatile size_t s = 0; auto t0 = Clock::now();
        for (const auto& q : queries) { auto r = tree.searchKnn(pts, q, knn); s += r.size(); }
        ts += elapsed_ms(t0); (void)s;
    }
    return { tb / iters, ts / iters };
}

// ---------------------------------------------------------------------------
// OpenCV FLANN benchmark
// checks=-1 → exact (FLANN_CHECKS_UNLIMITED), checks=32 → approximate
// ---------------------------------------------------------------------------

template<int DIMS>
BenchResult bench_opencv(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    int knn, int burn, int iters, int checks)
{
    cv::Mat data  = points_to_mat<DIMS>(pts);
    cv::Mat qmat  = points_to_mat<DIMS>(queries);
    cv::Mat idx_out(static_cast<int>(queries.size()), knn, CV_32S);
    cv::Mat dst_out(static_cast<int>(queries.size()), knn, CV_32F);
    cv::flann::SearchParams sp(checks);

    for (int b = 0; b < burn; ++b) {
        cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));
        fi.knnSearch(qmat, idx_out, dst_out, knn, sp);
    }

    double tb = 0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));
        tb += elapsed_ms(t0);
    }

    cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));
    for (int b = 0; b < burn; ++b)
        fi.knnSearch(qmat, idx_out, dst_out, knn, sp);

    double ts = 0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        fi.knnSearch(qmat, idx_out, dst_out, knn, sp);
        ts += elapsed_ms(t0);
    }
    return { tb / iters, ts / iters };
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static void print_header(int dims, int nq, int knn, int burn, int iters) {
    std::cout << "\n=== " << dims << "D  |  " << nq << " queries  |  KNN=" << knn
              << "  |  burn=" << burn << "  |  iters=" << iters << " ===\n";
    std::cout << "  cv-approx = 32 checks (approx)   cv-exact = unlimited checks\n";
    std::cout << "  p2f/p1: <1 means pico2-float faster than pico1\n";
    std::cout << "  d/f:    >1 means float faster than double\n\n";
    std::cout << std::left
        << std::setw(8)  << "N"
        << std::setw(12) << "pico1(ms)"
        << std::setw(12) << "p2f(ms)"
        << std::setw(12) << "p2d(ms)"
        << std::setw(8)  << "p2f/p1"
        << std::setw(8)  << "d/f"
        << std::setw(14) << "cv-app(ms)"
        << std::setw(9)  << "vs p2f"
        << std::setw(14) << "cv-ex(ms)"
        << std::setw(9)  << "vs p2f"
        << "\n" << std::string(106, '-') << "\n";
}

static void print_subheader(const char* s) {
    std::cout << " -- " << s << " --\n";
}

static void print_row(size_t n,
    double p1, double p2f, double p2d, double app, double ex)
{
    std::cout << std::left << std::fixed << std::setprecision(2)
        << std::setw(8)  << n
        << std::setw(12) << p1
        << std::setw(12) << p2f
        << std::setw(12) << p2d
        << std::setw(8)  << p2f / p1
        << std::setw(8)  << p2d / p2f
        << std::setw(14) << app
        << std::setw(9)  << app / p2f
        << std::setw(14) << ex
        << std::setw(9)  << ex / p2f
        << "\n";
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

template<int DIMS>
void run_benchmark(const std::vector<size_t>& sizes,
                   int nq, int knn, int burn, int iters)
{
    auto queries = random_points<DIMS>(nq, 1234);
    print_header(DIMS, nq, knn, burn, iters);

    std::vector<BenchResult> rp1, rp2f, rp2d, rapp, rex;
    for (size_t n : sizes) {
        auto pts = random_points<DIMS>(n);
        rp1 .push_back(bench_picoflann                        <DIMS>      (pts, queries, knn, burn, iters));
        rp2f.push_back(bench_picoflann2_scalar<DIMS, float>               (pts, queries, knn, burn, iters));
        rp2d.push_back(bench_picoflann2_scalar<DIMS, double>              (pts, queries, knn, burn, iters));
        rapp.push_back(bench_opencv                           <DIMS>      (pts, queries, knn, burn, iters, 32));
        rex .push_back(bench_opencv                           <DIMS>      (pts, queries, knn, burn, iters, -1));
    }

    print_subheader("BUILD   (p2f/p1<1 means pico2-float faster; d/f>1 means float faster)");
    for (size_t i = 0; i < sizes.size(); ++i)
        print_row(sizes[i], rp1[i].build_ms, rp2f[i].build_ms, rp2d[i].build_ms,
                             rapp[i].build_ms, rex[i].build_ms);

    std::cout << "\n";
    print_subheader("SEARCH  (vs p2f > 1 means pico2-float faster than opencv)");
    for (size_t i = 0; i < sizes.size(); ++i)
        print_row(sizes[i], rp1[i].search_ms, rp2f[i].search_ms, rp2d[i].search_ms,
                             rapp[i].search_ms, rex[i].search_ms);
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

    std::cout << "picoflann v1 vs picoflann v2-float vs picoflann v2-double vs OpenCV FLANN\n";
    std::cout << "p2f/p1 ratio: <1 = pico2-float faster   d/f ratio: >1 = float faster than double\n";

    run_benchmark<2>(SIZES, NQUERIES, KNN, BURN, ITERS);
    run_benchmark<3>(SIZES, NQUERIES, KNN, BURN, ITERS);
    run_benchmark<4>(SIZES, NQUERIES, KNN, BURN, ITERS);

    return 0;
}
