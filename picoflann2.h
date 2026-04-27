#ifndef PicoFlann2_H
#define PicoFlann2_H
// Optimised KD-tree — same public API as picoflann.h.
// Improvements over picoflann.h:
//   1. Node stores leaf indices inline (no heap allocation per leaf)
//   2. BoundingBox is std::array<pair<Scalar,Scalar>,DIMS>  (stack, no heap)
//   3. Distance uses compile-time DIMS (loop unrolling guaranteed)
//   4. Scalar type is a template parameter (float by default, use double for full precision)
//
// Usage:
//   KdTreeIndex<3, MyAdapter>              tree;  // float (fast default)
//   KdTreeIndex<3, MyAdapter, L2, double>  tree;  // double (full precision)
#include <vector>
#include <array>
#include <cassert>
#include <iostream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cstring>

namespace picoflann2 {

// ---------------------------------------------------------------------------
// L2 distance.
// Scalar is passed explicitly so both float and double specialisations share
// the same struct. The adapter output is cast to Scalar before arithmetic,
// so double precision is preserved when Scalar=double even for float data.
// ---------------------------------------------------------------------------
struct L2 {
    template<int NDIMS, typename Scalar, typename E, typename A>
    static inline Scalar compute(const E& a, const E& b, const A& adp, Scalar worst) {
        Scalar sqd = Scalar(0);
        for (int i = 0; i < NDIMS; ++i) {
            Scalar d = Scalar(adp(a, i)) - Scalar(adp(b, i));
            sqd += d * d;
            if (sqd > worst) return sqd;
        }
        return sqd;
    }
};

// ---------------------------------------------------------------------------
// KdTreeIndex
// Scalar: arithmetic type for distances and split values (float or double).
// ---------------------------------------------------------------------------
template<int DIMS,
         typename Adapter,
         typename Distance = L2,
         typename Scalar   = float,
         int MAX_LEAF       = 10>
class KdTreeIndex {

    // ---- Node ---------------------------------------------------------------
    // For Scalar=float : 24 + 4*MAX_LEAF bytes = 64 bytes (one cache line)
    // For Scalar=double: 32 + 4*MAX_LEAF bytes = 72 bytes
    struct Node {
        Scalar   div_val   = Scalar(0);
        Scalar   divhigh   = Scalar(0);
        Scalar   divlow    = Scalar(0);
        uint8_t  col_index = 0;
        uint8_t  idx_count = 0;
        uint16_t _pad      = 0;
        int32_t  _ileft    = -1;
        int32_t  _iright   = -1;
        uint32_t idx[MAX_LEAF];

        bool isLeaf() const { return _ileft == -1; }
    };
    static_assert(MAX_LEAF <= 255, "idx_count is uint8_t");

    // ---- BoundingBox --------------------------------------------------------
    using BBox = std::array<std::pair<Scalar, Scalar>, DIMS>;

    // ---- Index vector -------------------------------------------------------
    struct Index : std::vector<Node> {
        BBox rootBBox{};
        int  nValues = 0;
    };

    // ---- Max-heap result set ------------------------------------------------
    // Worst (largest) distance sits at the heap root.
    // Result pairs are stored as double for API compatibility with picoflann v1.
    class ResultSet {
    public:
        std::vector<std::pair<uint32_t, double>>& array;
        int    maxSize;
        Scalar maxValue    = std::numeric_limits<Scalar>::max();
        bool   radius_mode = false;

        ResultSet(std::vector<std::pair<uint32_t, double>>& data, int ms, Scalar mv)
            : array(data), maxSize(ms) {
            if (mv > Scalar(0)) { maxValue = mv; radius_mode = true; }
        }

        inline void push(uint32_t i, Scalar d) {
            if (radius_mode) {
                if (d < maxValue) array.push_back({i, (double)d});
                return;
            }
            if ((int)array.size() >= maxSize) {
                if (d < (Scalar)array[0].second) {
                    std::swap(array.front(), array.back());
                    array.pop_back();
                    if (array.size() > 1) sift_up(0);
                } else return;
            }
            array.push_back({i, (double)d});
            if (array.size() > 1) sift_down(array.size() - 1);
        }

        inline Scalar worstDist() const {
            if (radius_mode) return maxValue;
            if ((int)array.size() < maxSize) return std::numeric_limits<Scalar>::max();
            return (Scalar)array[0].second;
        }

    private:
        void sift_down(size_t i) {
            if (i == 0) return;
            size_t p = (i - 1) / 2;
            if (array[p].second < array[i].second) {
                std::swap(array[i], array[p]);
                sift_down(p);
            }
        }
        void sift_up(size_t i) {
            size_t l = 2*i+1, r = 2*i+2;
            if (l >= array.size()) return;
            if (r >= array.size()) {
                if (array[i].second < array[l].second) std::swap(array[i], array[l]);
                return;
            }
            if (array[r].second < array[l].second) {
                if (array[i].second < array[l].second) { std::swap(array[i], array[l]); sift_up(l); }
            } else {
                if (array[i].second < array[r].second) { std::swap(array[i], array[r]); sift_up(r); }
            }
        }
    };

    // ---- Members ------------------------------------------------------------
    Index  _index;
    Adapter adapter;
    std::vector<uint32_t> all_indices;

public:
    // -------------------------------------------------------------------------
    // Public API (identical to picoflann.h)
    // -------------------------------------------------------------------------
    template<typename Container>
    void build(const Container& container) {
        _index.clear();
        _index.reserve(container.size() * 2);
        _index.nValues = (int)container.size();
        all_indices.resize(container.size());
        for (size_t i = 0; i < container.size(); ++i) all_indices[i] = (uint32_t)i;
        if (container.size() == 0) return;
        computeBoundingBox(_index.rootBBox, 0, (int)all_indices.size(), container);
        _index.push_back(Node());
        BBox bbox = _index.rootBBox;
        divideTree(0, 0, (int)all_indices.size(), bbox, container);
    }

    void clear() { _index.clear(); all_indices.clear(); }

    template<typename Type, typename Container>
    std::vector<std::pair<uint32_t, double>> searchKnn(
        const Container& container, const Type& val, int nn, bool sorted = true) const {
        std::vector<std::pair<uint32_t, double>> res;
        generalSearch(res, container, val, Scalar(-1), sorted, nn);
        return res;
    }

    template<typename Type, typename Container>
    std::vector<std::pair<uint32_t, double>> radiusSearch(
        const Container& container, const Type& val, double dist,
        bool sorted = true, int maxNN = -1) const {
        std::vector<std::pair<uint32_t, double>> res;
        generalSearch(res, container, val, Scalar(dist), sorted, maxNN);
        return res;
    }

    template<typename Type, typename Container>
    void radiusSearch(std::vector<std::pair<uint32_t, double>>& res,
                      const Container& container, const Type& val, double dist,
                      bool sorted = true, int maxNN = -1) const {
        generalSearch(res, container, val, Scalar(dist), sorted, maxNN);
    }

    void toStream(std::ostream& str) const {
        str.write((char*)&_index.nValues, sizeof(_index.nValues));
        str.write((char*)&_index.rootBBox, sizeof(_index.rootBBox));
        uint64_t sz = _index.size();
        str.write((char*)&sz, sizeof(sz));
        for (size_t i = 0; i < _index.size(); ++i) {
            const Node& n = _index[i];
            str.write((char*)&n.div_val,   sizeof(n.div_val));
            str.write((char*)&n.divhigh,   sizeof(n.divhigh));
            str.write((char*)&n.divlow,    sizeof(n.divlow));
            str.write((char*)&n.col_index, sizeof(n.col_index));
            str.write((char*)&n.idx_count, sizeof(n.idx_count));
            str.write((char*)&n._ileft,    sizeof(n._ileft));
            str.write((char*)&n._iright,   sizeof(n._iright));
            str.write((char*)n.idx, sizeof(uint32_t) * n.idx_count);
        }
    }

    void fromStream(std::istream& str) {
        _index.clear();
        str.read((char*)&_index.nValues, sizeof(_index.nValues));
        str.read((char*)&_index.rootBBox, sizeof(_index.rootBBox));
        uint64_t sz;
        str.read((char*)&sz, sizeof(sz));
        _index.resize(sz);
        for (size_t i = 0; i < sz; ++i) {
            Node& n = _index[i];
            str.read((char*)&n.div_val,   sizeof(n.div_val));
            str.read((char*)&n.divhigh,   sizeof(n.divhigh));
            str.read((char*)&n.divlow,    sizeof(n.divlow));
            str.read((char*)&n.col_index, sizeof(n.col_index));
            str.read((char*)&n.idx_count, sizeof(n.idx_count));
            str.read((char*)&n._ileft,    sizeof(n._ileft));
            str.read((char*)&n._iright,   sizeof(n._iright));
            str.read((char*)n.idx, sizeof(uint32_t) * n.idx_count);
        }
    }

    uint64_t getHash() const {
        // Zero-init x before partial memcpy so float (4 bytes) fills cleanly into uint64.
        auto u64 = [](Scalar v) { uint64_t x = 0; memcpy(&x, &v, sizeof(v)); return x; };
        uint64_t seed = (uint64_t)DIMS;
        seed ^= (uint64_t)_index.nValues + 0x9e3779b9 + (seed<<6) + (seed>>2);
        for (int i = 0; i < DIMS; ++i) {
            seed ^= u64(_index.rootBBox[i].first)  + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= u64(_index.rootBBox[i].second) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        for (size_t i = 0; i < _index.size(); ++i) {
            const Node& n = _index[i];
            seed ^= u64(n.div_val)  + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= u64(n.divhigh)  + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= u64(n.divlow)   + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= (uint64_t)n.col_index + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= (uint64_t)n._ileft    + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= (uint64_t)n._iright   + 0x9e3779b9 + (seed<<6) + (seed>>2);
            for (int j = 0; j < n.idx_count; ++j)
                seed ^= (uint64_t)n.idx[j] + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }

private:
    // -------------------------------------------------------------------------
    // Build helpers
    // -------------------------------------------------------------------------
    template<typename Container>
    void computeBoundingBox(BBox& bbox, int start, int end, const Container& c) {
        for (int i = 0; i < DIMS; ++i)
            bbox[i].first = bbox[i].second = Scalar(adapter(c.at(all_indices[start]), i));
        for (int k = start + 1; k < end; ++k)
            for (int i = 0; i < DIMS; ++i) {
                Scalar v = Scalar(adapter(c.at(all_indices[k]), i));
                if (v < bbox[i].first)  bbox[i].first  = v;
                if (v > bbox[i].second) bbox[i].second = v;
            }
    }

    template<typename Container>
    void mean_var_calculate(int start, int end, Scalar var[], Scalar mean[], const Container& c) {
        constexpr int MAX_ELEM = 100;
        Scalar sum2[DIMS] = {};
        for (int i = 0; i < DIMS; ++i) mean[i] = Scalar(0);
        int incr = (end - start >= 2*MAX_ELEM) ? (end - start) / MAX_ELEM : 1;
        int cnt  = 0;
        for (int i = start; i < end; i += incr) {
            for (int d = 0; d < DIMS; ++d) {
                Scalar v = Scalar(adapter(c.at(all_indices[i]), d));
                mean[d] += v;
                sum2[d] += v * v;
            }
            ++cnt;
        }
        Scalar inv = Scalar(1) / Scalar(cnt);
        for (int d = 0; d < DIMS; ++d) {
            mean[d] *= inv;
            var[d] = sum2[d] * inv - mean[d] * mean[d];
        }
    }

    template<typename Container>
    void planeSplit(uint32_t* ind, int count, int cutfeat, Scalar cutval,
                    int& lim1, int& lim2, const Container& c) {
        int l = 0, r = count - 1;
        for (;;) {
            while (l <= r && Scalar(adapter(c.at(ind[l]), cutfeat)) <  cutval) ++l;
            while (l <= r && Scalar(adapter(c.at(ind[r]), cutfeat)) >= cutval) --r;
            if (l > r) break;
            std::swap(ind[l++], ind[r--]);
        }
        lim1 = l; r = count - 1;
        for (;;) {
            while (l <= r && Scalar(adapter(c.at(ind[l]), cutfeat)) <= cutval) ++l;
            while (l <= r && Scalar(adapter(c.at(ind[r]), cutfeat)) >  cutval) --r;
            if (l > r) break;
            std::swap(ind[l++], ind[r--]);
        }
        lim2 = l;
    }

    template<typename Container>
    void divideTree(int nodeIdx, int start, int end, BBox& bbox, const Container& c) {
        int count = end - start;
        assert(start < end);

        if (count <= MAX_LEAF) {
            Node& leaf     = _index[nodeIdx];
            leaf.idx_count = (uint8_t)count;
            for (int i = 0; i < count; ++i) leaf.idx[i] = all_indices[start + i];
            computeBoundingBox(bbox, start, end, c);
            return;
        }

        Node& curr   = _index[nodeIdx];
        curr._ileft  = (int32_t)_index.size();
        curr._iright = (int32_t)_index.size() + 1;
        _index.push_back(Node());
        int leftNode  = (int)_index.size() - 1;
        _index.push_back(Node());
        int rightNode = (int)_index.size() - 1;

        Scalar var[DIMS], mean[DIMS];
        mean_var_calculate(start, end, var, mean, c);
        curr.col_index = 0;
        for (int i = 1; i < DIMS; ++i)
            if (var[i] > var[curr.col_index]) curr.col_index = i;
        curr.div_val = mean[curr.col_index];

        int lim1, lim2;
        planeSplit(&all_indices[start], count, curr.col_index, curr.div_val, lim1, lim2, c);

        int split;
        if      (lim1 > count/2)  split = lim1;
        else if (lim2 < count/2)  split = lim2;
        else                      split = count/2;
        if (lim1 == count || lim2 == 0) split = count/2;

        if (split < MAX_LEAF || count - split < MAX_LEAF) {
            std::sort(all_indices.begin()+start, all_indices.begin()+end,
                [&](uint32_t a, uint32_t b) {
                    return Scalar(adapter(c.at(a), curr.col_index)) <
                           Scalar(adapter(c.at(b), curr.col_index));
                });
            split        = count / 2;
            curr.div_val = Scalar(adapter(c.at(all_indices[start + split]), curr.col_index));
        }

        BBox left_bbox(bbox);
        left_bbox[curr.col_index].second = curr.div_val;
        divideTree(leftNode, start, start + split, left_bbox, c);
        left_bbox[curr.col_index].second = curr.div_val;

        BBox right_bbox(bbox);
        right_bbox[curr.col_index].first = curr.div_val;
        divideTree(rightNode, start + split, end, right_bbox, c);

        curr.divlow  = left_bbox[curr.col_index].second;
        curr.divhigh = right_bbox[curr.col_index].first;
        assert(curr.divlow <= curr.divhigh);

        for (int i = 0; i < DIMS; ++i) {
            bbox[i].first  = std::min(left_bbox[i].first,  right_bbox[i].first);
            bbox[i].second = std::max(left_bbox[i].second, right_bbox[i].second);
        }
    }

    // -------------------------------------------------------------------------
    // Search
    // -------------------------------------------------------------------------
    template<typename Type>
    Scalar computeInitialDistances(const Type& elem, Scalar dists[], const BBox& bbox) const {
        Scalar distsq = Scalar(0);
        for (int i = 0; i < DIMS; ++i) {
            Scalar ei = Scalar(adapter(elem, i));
            if (ei < bbox[i].first) {
                Scalar d = ei - bbox[i].first;
                dists[i] = d * d;
                distsq  += dists[i];
            } else if (ei > bbox[i].second) {
                Scalar d = ei - bbox[i].second;
                dists[i] = d * d;
                distsq  += dists[i];
            } else {
                dists[i] = Scalar(0);
            }
        }
        return distsq;
    }

    template<typename Type, typename Container>
    void searchRecursive(int32_t nodeIdx, const Type& val, ResultSet& hres,
                         Scalar mindistsq, Scalar dists[],
                         const Container& container) const {
        const Node& node = _index[nodeIdx];
        if (node.isLeaf()) {
            Scalar worst = hres.worstDist();
            for (int i = 0; i < node.idx_count; ++i) {
                Scalar sqd = Distance::template compute<DIMS, Scalar>(
                    val, container.at(node.idx[i]), adapter, worst);
                if (sqd < worst) {
                    hres.push(node.idx[i], sqd);
                    worst = hres.worstDist();
                }
            }
        } else {
            Scalar qval  = Scalar(adapter(val, node.col_index));
            Scalar diff1 = qval - node.divlow;
            Scalar diff2 = qval - node.divhigh;

            int32_t bestChild, otherChild;
            Scalar  cut_dist;
            if (diff1 + diff2 < Scalar(0)) {
                bestChild  = node._ileft;
                otherChild = node._iright;
                cut_dist   = diff2 * diff2;
            } else {
                bestChild  = node._iright;
                otherChild = node._ileft;
                cut_dist   = diff1 * diff1;
            }

            searchRecursive(bestChild, val, hres, mindistsq, dists, container);

            Scalar saved          = dists[node.col_index];
            mindistsq             = mindistsq + cut_dist - saved;
            dists[node.col_index] = cut_dist;
            if (mindistsq <= hres.worstDist())
                searchRecursive(otherChild, val, hres, mindistsq, dists, container);
            dists[node.col_index] = saved;
        }
    }

    template<typename Type, typename Container>
    void generalSearch(std::vector<std::pair<uint32_t, double>>& res,
                       const Container& container, const Type& val,
                       Scalar dist, bool sorted, int maxNn) const {
        res.clear();
        if (_index.empty()) return;
        ResultSet hres(res, maxNn < 0 ? std::numeric_limits<int>::max() : maxNn,
                       dist > Scalar(0) ? dist * dist : Scalar(-1));

        Scalar dists[DIMS] = {};
        Scalar mindistsq   = computeInitialDistances(val, dists, _index.rootBBox);
        searchRecursive(0, val, hres, mindistsq, dists, container);

        if (sorted && res.size() > 1)
            std::sort(res.begin(), res.end(),
                [](const std::pair<uint32_t,double>& a, const std::pair<uint32_t,double>& b) {
                    return a.second < b.second;
                });
    }
};

} // namespace picoflann2
#endif
