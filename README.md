# cyvec - a Cythonised vector class

I took [wasabi.geom](https://wasabigeom.readthedocs.io/en/latest/), a
pure-Python vector maths library that I wrote a long time ago, and started
Cythonising it.

I've made some changes to the interface; notably, I prefer radians thes days.

This is an experiment to see how effective this approach is compared to a
[Rust implementation I started previously](https://github.com/lordmauve/wvec),
on considerations like developer productivity and the speed of the resulting
library.
