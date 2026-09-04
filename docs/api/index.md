# API reference

Generated from the package's own docstrings, so what you read here is what the
code says rather than a second description of it that can drift.

Two conventions worth knowing before you read a page:

**Everything is documented at the module that *defines* it, not at the package
that re-exports it.** `load_depth_map` appears under
`visbench.data.dense` even though `visbench.data.load_depth_map` also works. Of the 371 cross-references written into the docstrings, 124 name
the defining path against 27 that name the re-export, so this is where the links
already point.

**`#:` comments above a module-level name are its documentation.** There are 462
of them, and they carry things a type annotation cannot — why
`METRIC_DIRECTIONS` is a listed table rather than a heuristic, what each schema
version added, which Taskonomy domains were measured rather than assumed.

```{toctree}
:maxdepth: 2

core
backbones
tasks
tasks-high-level
tasks-mid-level
tasks-low-level
data
heads
metrics
cache
viz
hub
results
utils
```
