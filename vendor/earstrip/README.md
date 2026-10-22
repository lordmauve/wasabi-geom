# Vendored Earstrip

`earstrip.hpp` and `LICENSE` are vendored from
https://github.com/GLMap/earstrip at commit
`a145683a71b8a723f1ae9afa20b7f434e9b957e3` (retrieved 2026-10-22).
Earstrip is ISC licensed; its notice is included in source and wheel distributions.
No upstream benchmark datasets are bundled.

Local patch: `detail::lowestBit` uses two 32-bit `_BitScanForward` calls on
32-bit MSVC, where `_BitScanForward64` is unavailable. The 64-bit MSVC and
GCC/Clang paths are unchanged. `LICENSE` is unmodified.

The Python adapter is project code, not an upstream Python binding. It disables
extra vertices, constrained triangulation and forced cuts: returned indices must
refer to the caller's input vertices. Inputs must describe simple polygons with
non-intersecting holes. This is not a validator for arbitrary polygon topology.

Source story: https://news.ycombinator.com/item?id=50193170
Article: https://globus.software/blog/2026/10/16/cutting-map-polygons/
