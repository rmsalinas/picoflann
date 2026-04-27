#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <sstream>

#include <opencv2/flann.hpp>

#include "picoflann.h"
#include "picoflann2.h"

// ---------------------------------------------------------------------------
// Point / adapter (shared with testperformance.cpp)
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
// Data helpers
// ---------------------------------------------------------------------------

template<int DIMS>
std::vector<Point<DIMS>> random_points(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-100.f, 100.f);
    std::vector<Point<DIMS>> pts(n);
    for (auto& p : pts)
        for (int d = 0; d < DIMS; ++d)
            p.data[d] = dist(rng);
    return pts;
}

template<int DIMS>
cv::Mat points_to_mat(const std::vector<Point<DIMS>>& pts) {
    cv::Mat m((int)pts.size(), DIMS, CV_32F);
    for (int i = 0; i < (int)pts.size(); ++i)
        std::memcpy(m.ptr<float>(i), pts[i].data, DIMS * sizeof(float));
    return m;
}

// ---------------------------------------------------------------------------
// Brute-force ground truth (double precision)
// ---------------------------------------------------------------------------

template<int DIMS>
std::vector<std::pair<uint32_t, double>> brute_knn(
    const std::vector<Point<DIMS>>& pts,
    const Point<DIMS>& q, int k)
{
    std::vector<std::pair<double, uint32_t>> all;
    all.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        double d = 0;
        for (int dim = 0; dim < DIMS; ++dim) {
            double diff = (double)pts[i].data[dim] - (double)q.data[dim];
            d += diff * diff;
        }
        all.push_back({d, (uint32_t)i});
    }
    std::sort(all.begin(), all.end());
    std::vector<std::pair<uint32_t, double>> res(k);
    for (int i = 0; i < k; ++i)
        res[i] = {all[i].second, all[i].first};
    return res;
}

template<int DIMS>
std::set<uint32_t> brute_radius(
    const std::vector<Point<DIMS>>& pts,
    const Point<DIMS>& q, double radius)
{
    double r2 = radius * radius;
    std::set<uint32_t> res;
    for (size_t i = 0; i < pts.size(); ++i) {
        double d = 0;
        for (int dim = 0; dim < DIMS; ++dim) {
            double diff = (double)pts[i].data[dim] - (double)q.data[dim];
            d += diff * diff;
        }
        if (d <= r2) res.insert((uint32_t)i);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail = "") {
    if (ok) {
        std::cout << "  [PASS] " << name;
        ++g_pass;
    } else {
        std::cout << "  [FAIL] " << name;
        ++g_fail;
    }
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// KNN accuracy checker
// Compares a set of results against double-precision brute-force ground truth.
// Returns {index_match_rate, recall, max_relative_distance_error}.
// ---------------------------------------------------------------------------

struct AccStats {
    double index_match_pct = 0;  // % queries where all K indices match GT (exact order)
    double recall_pct      = 0;  // % of true K neighbors found (set recall, avg over queries)
    double max_rel_dist_err = 0; // max relative error |d_ret - d_gt| / d_gt over all (q,k)
    int    n_queries       = 0;
};

template<int DIMS>
AccStats knn_accuracy(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    int k,
    const std::vector<std::vector<std::pair<uint32_t,double>>>& results)
{
    AccStats s;
    s.n_queries = (int)queries.size();
    int exact_matches = 0;
    double total_recall = 0;

    for (int q = 0; q < (int)queries.size(); ++q) {
        auto gt = brute_knn<DIMS>(pts, queries[q], k);
        const auto& res = results[q];

        // Index match: same ordered list
        bool all_same = ((int)res.size() == k);
        for (int i = 0; all_same && i < k; ++i)
            all_same = (res[i].first == gt[i].first);
        if (all_same) ++exact_matches;

        // Recall: set overlap
        std::set<uint32_t> gt_set;
        for (auto& p : gt) gt_set.insert(p.first);
        int found = 0;
        for (auto& p : res)
            if (gt_set.count(p.first)) ++found;
        total_recall += (double)found / k;

        // Distance error (compare against GT distances at same position in sorted list)
        for (int i = 0; i < (int)std::min((int)res.size(), k); ++i) {
            if (gt[i].second > 1e-10) {
                double err = std::abs(res[i].second - gt[i].second) / gt[i].second;
                s.max_rel_dist_err = std::max(s.max_rel_dist_err, err);
            }
        }
    }

    s.index_match_pct = 100.0 * exact_matches / queries.size();
    s.recall_pct      = 100.0 * total_recall   / queries.size();
    return s;
}

// ---------------------------------------------------------------------------
// Radius accuracy checker
// ---------------------------------------------------------------------------

template<int DIMS>
struct RadiusStats {
    double exact_set_match_pct = 0; // % queries where returned set == GT set exactly
    double avg_precision       = 0; // avg precision = |returned ∩ GT| / |returned|
    double avg_recall          = 0; // avg recall    = |returned ∩ GT| / |GT|
    int    n_queries           = 0;
};

template<int DIMS>
RadiusStats<DIMS> radius_accuracy(
    const std::vector<Point<DIMS>>& pts,
    const std::vector<Point<DIMS>>& queries,
    double radius,
    const std::vector<std::vector<std::pair<uint32_t,double>>>& results)
{
    RadiusStats<DIMS> s;
    s.n_queries = (int)queries.size();
    int exact = 0;
    double total_prec = 0, total_rec = 0;

    for (int q = 0; q < (int)queries.size(); ++q) {
        auto gt_set = brute_radius<DIMS>(pts, queries[q], radius);
        std::set<uint32_t> ret_set;
        for (auto& p : results[q]) ret_set.insert(p.first);

        int isect = 0;
        for (auto x : ret_set) if (gt_set.count(x)) ++isect;

        if (ret_set == gt_set) ++exact;
        total_prec += ret_set.empty() ? 1.0 : (double)isect / ret_set.size();
        total_rec  += gt_set.empty()  ? 1.0 : (double)isect / gt_set.size();
    }

    s.exact_set_match_pct = 100.0 * exact / queries.size();
    s.avg_precision        = 100.0 * total_prec / queries.size();
    s.avg_recall           = 100.0 * total_rec  / queries.size();
    return s;
}

// ---------------------------------------------------------------------------
// Print a row for KNN results table
// ---------------------------------------------------------------------------

static void print_knn_row(const char* name, const AccStats& s, bool is_exact) {
    std::cout << "    " << std::left << std::setw(20) << name
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(9) << s.index_match_pct << "%"
              << std::setw(9) << s.recall_pct      << "%"
              << "   " << std::scientific << std::setprecision(2)
              << s.max_rel_dist_err;
    if (!is_exact) std::cout << "  (approx — low recall expected)";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Run KNN accuracy test for one configuration
// ---------------------------------------------------------------------------

template<int DIMS>
bool run_knn_test(int n, int k, int nq,
                  const char* label, double dist_tol = 1e-5) {
    auto pts     = random_points<DIMS>(n, 42);
    auto queries = random_points<DIMS>(nq, 999);

    // --- picoflann v1 ---
    picoflann::KdTreeIndex<DIMS, PointAdapter<DIMS>> tree1;
    tree1.build(pts);
    std::vector<std::vector<std::pair<uint32_t,double>>> res1(nq), res2(nq);
    for (int q = 0; q < nq; ++q)
        res1[q] = tree1.searchKnn(pts, queries[q], k);

    // --- picoflann v2 (float) ---
    picoflann2::KdTreeIndex<DIMS, PointAdapter<DIMS>> tree2;
    tree2.build(pts);
    for (int q = 0; q < nq; ++q)
        res2[q] = tree2.searchKnn(pts, queries[q], k);

    // --- picoflann v2 (double) ---
    picoflann2::KdTreeIndex<DIMS, PointAdapter<DIMS>, picoflann2::L2, double> tree2d;
    tree2d.build(pts);
    std::vector<std::vector<std::pair<uint32_t,double>>> res2d(nq);
    for (int q = 0; q < nq; ++q)
        res2d[q] = tree2d.searchKnn(pts, queries[q], k);

    // --- OpenCV linear (brute-force) — guaranteed exact ---
    cv::Mat data = points_to_mat<DIMS>(pts);
    cv::Mat qmat = points_to_mat<DIMS>(queries);
    cv::Mat idx_out((int)nq, k, CV_32S);
    cv::Mat dst_out((int)nq, k, CV_32F);
    cv::flann::Index fi_lin(data, cv::flann::LinearIndexParams());
    fi_lin.knnSearch(qmat, idx_out, dst_out, k, cv::flann::SearchParams(-1));
    std::vector<std::vector<std::pair<uint32_t,double>>> res_cv_lin(nq);
    for (int q = 0; q < nq; ++q) {
        res_cv_lin[q].resize(k);
        for (int i = 0; i < k; ++i)
            res_cv_lin[q][i] = {(uint32_t)idx_out.at<int>(q,i),
                                (double)dst_out.at<float>(q,i)};
    }

    // --- OpenCV KD-tree "exact" (SearchParams(-1)) ---
    cv::flann::Index fi(data, cv::flann::KDTreeIndexParams(1));
    fi.knnSearch(qmat, idx_out, dst_out, k, cv::flann::SearchParams(-1));
    std::vector<std::vector<std::pair<uint32_t,double>>> res_cv_exact(nq);
    for (int q = 0; q < nq; ++q) {
        res_cv_exact[q].resize(k);
        for (int i = 0; i < k; ++i)
            res_cv_exact[q][i] = {(uint32_t)idx_out.at<int>(q,i),
                                  (double)dst_out.at<float>(q,i)};
    }

    // --- OpenCV KD-tree approx (32 checks) ---
    fi.knnSearch(qmat, idx_out, dst_out, k, cv::flann::SearchParams(32));
    std::vector<std::vector<std::pair<uint32_t,double>>> res_cv_approx(nq);
    for (int q = 0; q < nq; ++q) {
        res_cv_approx[q].resize(k);
        for (int i = 0; i < k; ++i)
            res_cv_approx[q][i] = {(uint32_t)idx_out.at<int>(q,i),
                                   (double)dst_out.at<float>(q,i)};
    }

    auto s1      = knn_accuracy<DIMS>(pts, queries, k, res1);
    auto s2      = knn_accuracy<DIMS>(pts, queries, k, res2);
    auto s2d     = knn_accuracy<DIMS>(pts, queries, k, res2d);
    auto s_cvlin = knn_accuracy<DIMS>(pts, queries, k, res_cv_lin);
    auto s_cvex  = knn_accuracy<DIMS>(pts, queries, k, res_cv_exact);
    auto s_cvapp = knn_accuracy<DIMS>(pts, queries, k, res_cv_approx);

    std::cout << "\n  KNN  " << label << "\n";
    std::cout << "    " << std::left << std::setw(22) << "method"
              << std::right << std::setw(10) << "idx_match"
              << std::setw(9) << "recall"
              << "   max_dist_err\n";
    std::cout << "    " << std::string(62, '-') << "\n";
    print_knn_row("picoflann v1",        s1,      true);
    print_knn_row("picoflann v2 (float)", s2,      true);
    print_knn_row("picoflann v2 (dbl)",   s2d,     true);
    print_knn_row("opencv linear",        s_cvlin, true);
    print_knn_row("opencv kd exact",      s_cvex,  false);
    print_knn_row("opencv kd approx",     s_cvapp, false);

    bool ok1    = (s1.index_match_pct    == 100.0 && s1.recall_pct    == 100.0 && s1.max_rel_dist_err    < dist_tol);
    bool ok2    = (s2.index_match_pct    == 100.0 && s2.recall_pct    == 100.0 && s2.max_rel_dist_err    < dist_tol);
    bool ok2d   = (s2d.index_match_pct   == 100.0 && s2d.recall_pct   == 100.0 && s2d.max_rel_dist_err   < dist_tol);
    bool ok_lin = (s_cvlin.index_match_pct == 100.0 && s_cvlin.recall_pct == 100.0);

    std::string detail1   = ok1    ? "" : "idx=" + std::to_string((int)s1.index_match_pct)       + "% recall=" + std::to_string((int)s1.recall_pct) + "%";
    std::string detail2   = ok2    ? "" : "idx=" + std::to_string((int)s2.index_match_pct)       + "% recall=" + std::to_string((int)s2.recall_pct) + "%";
    std::string detail2d  = ok2d   ? "" : "idx=" + std::to_string((int)s2d.index_match_pct)      + "% recall=" + std::to_string((int)s2d.recall_pct) + "%";
    std::string detail_ln = ok_lin ? "" : "idx=" + std::to_string((int)s_cvlin.index_match_pct)  + "%";

    std::ostringstream tag;
    tag << label;
    report("picoflann v1      " + tag.str(), ok1,    detail1);
    report("picoflann v2 flt  " + tag.str(), ok2,    detail2);
    report("picoflann v2 dbl  " + tag.str(), ok2d,   detail2d);
    report("opencv linear     " + tag.str(), ok_lin, detail_ln);
    std::cout << "  [INFO] opencv kd exact " << label
              << "  idx=" << std::fixed << std::setprecision(1) << s_cvex.index_match_pct
              << "% recall=" << s_cvex.recall_pct << "%\n";

    return ok1 && ok2 && ok2d && ok_lin;
}

// ---------------------------------------------------------------------------
// Run radius search accuracy test for one configuration
// ---------------------------------------------------------------------------

template<int DIMS>
bool run_radius_test(int n, double radius, int nq, const char* label) {
    auto pts     = random_points<DIMS>(n, 42);
    auto queries = random_points<DIMS>(nq, 999);

    // --- picoflann v1 ---
    picoflann::KdTreeIndex<DIMS, PointAdapter<DIMS>> tree1;
    tree1.build(pts);
    std::vector<std::vector<std::pair<uint32_t,double>>> res1(nq);
    for (int q = 0; q < nq; ++q)
        res1[q] = tree1.radiusSearch(pts, queries[q], radius);

    // --- picoflann v2 (float) ---
    picoflann2::KdTreeIndex<DIMS, PointAdapter<DIMS>> tree2;
    tree2.build(pts);
    std::vector<std::vector<std::pair<uint32_t,double>>> res2(nq);
    for (int q = 0; q < nq; ++q)
        res2[q] = tree2.radiusSearch(pts, queries[q], radius);

    // --- picoflann v2 (double) ---
    picoflann2::KdTreeIndex<DIMS, PointAdapter<DIMS>, picoflann2::L2, double> tree2d;
    tree2d.build(pts);
    std::vector<std::vector<std::pair<uint32_t,double>>> res2d(nq);
    for (int q = 0; q < nq; ++q)
        res2d[q] = tree2d.radiusSearch(pts, queries[q], radius);

    auto s1  = radius_accuracy<DIMS>(pts, queries, radius, res1);
    auto s2  = radius_accuracy<DIMS>(pts, queries, radius, res2);
    auto s2d = radius_accuracy<DIMS>(pts, queries, radius, res2d);

    std::cout << "\n  Radius  " << label << "\n";
    std::cout << "    " << std::left << std::setw(22) << "method"
              << std::right << std::setw(12) << "exact_set%"
              << std::setw(12) << "precision%"
              << std::setw(12) << "recall%\n";
    std::cout << "    " << std::string(58, '-') << "\n";

    auto print_r = [&](const char* name, const RadiusStats<DIMS>& s) {
        std::cout << "    " << std::left << std::setw(22) << name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(11) << s.exact_set_match_pct << "%"
                  << std::setw(11) << s.avg_precision       << "%"
                  << std::setw(11) << s.avg_recall          << "%\n";
    };
    print_r("picoflann v1",        s1);
    print_r("picoflann v2 (float)", s2);
    print_r("picoflann v2 (dbl)",   s2d);

    bool ok1  = (s1.exact_set_match_pct  == 100.0);
    bool ok2  = (s2.exact_set_match_pct  == 100.0);
    bool ok2d = (s2d.exact_set_match_pct == 100.0);

    std::ostringstream tag; tag << label;
    report("picoflann v1      " + tag.str(), ok1);
    report("picoflann v2 flt  " + tag.str(), ok2);
    report("picoflann v2 dbl  " + tag.str(), ok2d);

    return ok1 && ok2 && ok2d;
}

// ---------------------------------------------------------------------------
// Consistency check: pico v1 and pico v2 must return the same indices
// (distances may differ by float vs double rounding)
// ---------------------------------------------------------------------------

template<int DIMS>
bool run_consistency_test(int n, int k, int nq, const char* label) {
    auto pts     = random_points<DIMS>(n, 77);
    auto queries = random_points<DIMS>(nq, 888);

    picoflann::KdTreeIndex<DIMS, PointAdapter<DIMS>>  tree1; tree1.build(pts);
    picoflann2::KdTreeIndex<DIMS, PointAdapter<DIMS>> tree2; tree2.build(pts);

    int mismatch = 0;
    double max_rel_dist_err = 0;

    for (int q = 0; q < nq; ++q) {
        auto r1 = tree1.searchKnn(pts, queries[q], k);
        auto r2 = tree2.searchKnn(pts, queries[q], k);

        for (int i = 0; i < k; ++i) {
            if (r1[i].first != r2[i].first) { ++mismatch; break; }
            if (r1[i].second > 1e-10) {
                double err = std::abs(r2[i].second - r1[i].second) / r1[i].second;
                max_rel_dist_err = std::max(max_rel_dist_err, err);
            }
        }
    }

    bool ok = (mismatch == 0) && (max_rel_dist_err < 1e-5);

    std::cout << "\n  Consistency (pico v1 == pico v2)  " << label << "\n";
    std::cout << "    index mismatches: " << mismatch << "/" << nq
              << "   max_rel_dist_err: " << std::scientific << std::setprecision(2)
              << max_rel_dist_err << "\n";

    std::ostringstream tag; tag << label;
    report("v1 == v2  " + tag.str(), ok);
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== picoflann accuracy tests ===\n";
    std::cout << "Ground truth: brute-force double-precision\n";
    std::cout << "dist_tol for exact methods: 1e-5 relative\n";

    // ---- KNN accuracy ----
    std::cout << "\n--- KNN accuracy (all exact methods must hit 100% recall) ---\n";
    run_knn_test<2>(1000,  10, 200, "2D  N=1000  K=10");
    run_knn_test<3>(1000,  10, 200, "3D  N=1000  K=10");
    run_knn_test<4>(1000,  10, 200, "4D  N=1000  K=10");
    run_knn_test<2>(10000, 10, 200, "2D  N=10000 K=10");
    run_knn_test<3>(10000, 10, 200, "3D  N=10000 K=10");
    run_knn_test<4>(10000, 10, 200, "4D  N=10000 K=10");
    run_knn_test<2>(1000,   1, 200, "2D  N=1000  K=1");
    run_knn_test<3>(1000,   1, 200, "3D  N=1000  K=1");
    run_knn_test<2>(1000,  20, 200, "2D  N=1000  K=20");
    run_knn_test<3>(1000,  20, 200, "3D  N=1000  K=20");

    // ---- Radius search accuracy ----
    std::cout << "\n--- Radius search accuracy (exact set match required) ---\n";
    run_radius_test<2>(1000,  20.0, 200, "2D  N=1000  r=20");
    run_radius_test<3>(1000,  30.0, 200, "3D  N=1000  r=30");
    run_radius_test<4>(1000,  40.0, 200, "4D  N=1000  r=40");
    run_radius_test<2>(10000, 10.0, 200, "2D  N=10000 r=10");
    run_radius_test<3>(10000, 15.0, 200, "3D  N=10000 r=15");

    // ---- pico v1 vs pico v2 consistency ----
    std::cout << "\n--- v1 vs v2 consistency (same indices, near-same distances) ---\n";
    run_consistency_test<2>(5000, 10, 500, "2D  N=5000  K=10");
    run_consistency_test<3>(5000, 10, 500, "3D  N=5000  K=10");
    run_consistency_test<4>(5000, 10, 500, "4D  N=5000  K=10");
    run_consistency_test<2>(5000,  1, 500, "2D  N=5000  K=1");
    run_consistency_test<3>(5000,  1, 500, "3D  N=5000  K=1");

    // ---- Summary ----
    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
