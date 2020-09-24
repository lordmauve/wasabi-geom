"""Benchmarks for vector classes in Python"""
from timeit import Timer

def timeit(name, stmt):
    defs = dict(
        v=v,
        a=v(1.0, 2.0),
        b=v(-1.0, -1.0),
    )

    loops, time = Timer(
        setup="c = v(0, 0)",
        stmt=stmt,
        globals={**globals(), **defs},
    ).autorange()

    dur = time / loops
    print(f"{name}: {dur * 1e6:0.2f}us per op ({loops} samples)")


from wasabigeom import vec2 as v

print("*** cyvec ***")
timeit("Addition", "a + b")
timeit("In-place addition", "c += b")
timeit("Dot", "a.dot(b)")
timeit("Normalized", "a.normalized()")
print()
