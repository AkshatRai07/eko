# EKO Architecture

---

## 0. Repository layout

```text
eko/
├── src/
│   ├── eko/                  # Python package (orchestration + kernels)
│   │   ├── runner/           # user-facing entry points
│   │   ├── evolution_operator/
│   │   ├── kernels/          # Evolution kernels
│   │   ├── scale_variations/
│   │   ├── io/
│   │   └── ...
│   └── ekore/                # Python anomalous dimensions (reference impl.)
├── crates/
│   ├── eko/                  # Glue between Rust and Python (lib.rs)
│   ├── ekore/                # Math Core (mirrors src/ekore/)
│   └── dekoder/
├── extras/                   # Architecture file stored here.
├── ...
└── pyproject.toml
```

> There are other folders and files (for eg. doc, tests, etc), but they aren't much useful here as we focus on math backend
