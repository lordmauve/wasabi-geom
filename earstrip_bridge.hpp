// Project-owned adapter. Earstrip itself is vendored under its ISC license.
#pragma once
#include "vendor/earstrip/earstrip.hpp"
#include <array>
#include <stdexcept>

namespace wasabi {
template<class T> struct Points {
    const char *data;
    std::ptrdiff_t row, col;
    size_t first, count;
    size_t size() const { return count; }
    std::array<double, 2> operator[](size_t i) const {
        T x, y;
        const char *p = data + static_cast<std::ptrdiff_t>(first + i) * row;
        // memcpy also supports unaligned buffer views without undefined behaviour.
        std::memcpy(&x, p, sizeof(T));
        std::memcpy(&y, p + col, sizeof(T));
        return {static_cast<double>(x), static_cast<double>(y)};
    }
};
template<class T> struct Rings {
    Points<T> points;
    const std::vector<uint32_t> &ends;
    size_t size() const { return ends.empty() ? (points.count ? 1 : 0) : ends.size(); }
    Points<T> operator[](size_t r) const {
        const size_t start = r ? ends[r - 1] : 0;
        const size_t end = ends.empty() ? points.count : ends[r];
        return {points.data, points.row, points.col, start, end - start};
    }
};

template<class T>
std::vector<uint32_t> triangulate(const char *data, size_t count,
                                 std::ptrdiff_t row, std::ptrdiff_t col,
                                 const std::vector<uint32_t> &ends) {
    const Points<T> points{data, row, col, 0, count};
    for (size_t i = 0; i < count; ++i) {
        const auto p = points[i];
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]))
            throw std::invalid_argument("vertices must contain only finite coordinates");
    }
    earstrip::Triangulator<uint32_t> t;
    t.setExtraPoints(false);
    t.setConstrained(false);
    t.setForcedCuts(false);
    const auto status = t.run(Rings<T>{points, ends});
    if (status != earstrip::Status::Exact)
        throw std::invalid_argument("polygon could not be triangulated exactly; use simple rings with non-intersecting holes");
    std::vector<uint32_t> result;
    t.triangles(result);
    return result;
}
inline std::vector<uint32_t> triangulate32(const char *p, size_t n, std::ptrdiff_t row,
                                         std::ptrdiff_t col, const std::vector<uint32_t> &ends) {
    return triangulate<float>(p, n, row, col, ends);
}
inline std::vector<uint32_t> triangulate64(const char *p, size_t n, std::ptrdiff_t row,
                                         std::ptrdiff_t col, const std::vector<uint32_t> &ends) {
    return triangulate<double>(p, n, row, col, ends);
}
} // namespace wasabi
