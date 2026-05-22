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

---

## 1. Build system

The patch file switches the root build backend from Poetry to Maturin. This switch is not needed. Maturin is designed to be the root build tool for projects that are primarily or completely Rust (example you told me, [pineappl](https://github.com/NNPDF/pineappl)). Currently, `eko` is still predominantly Python. Using Maturin as the root builder produces broken wheels and complicates the standard `poetry install` workflow without any benefit.

We shoulld keep Poetry as the single build tool for the `eko` Python package. Maturin's role is narrower - it compiles the Rust extension crate and installs it into the active venv. Poetry then treats that compiled extension as a path dependency. The transition to Maturin as root builder will happen only once the project is heavily or completely Rust.

The necessary changes in the current architecture for this to take place are:

- Change build backend to Poetry.
- Add eko-rs as a dependency in pyproject.toml.
- Add another .github/workflows file which publish the crates to PyPI on commit.
- The file will take care of building, publishing using maturin. The version published will be taken care by `crates/bump-versions.py`.

---

## 2. User entry point

**File:** `src/eko/runner/managed.py` — `solve(theory, operator, path)`

This is the single public function users call. It loads the two cards (`TheoryCard`, `OperatorCard`), builds an `Atlas`, and delegates to `runner/parts.py` for each segment and matching.

### High-level data flow

```text
User
 └─ runner.solve()                         [managed.py]
     └─ for each evolution segment / matching
         ├─ Operator.compute()             [evolution_operator/__init__.py]
         └─ OperatorMatrixElement.compute() [operator_matrix_element.py]
             └─ Operator.integrate()
                 └─ for each target x-grid point
                     └─ run_op_integration()
                         └─ for each source basis function j
                             └─ for each flavor label
                                 └─ scipy.integrate.quad(func, 0.5, 1-ε)
                                     └─ func called O(100) times per quad
                                         └─ quad_ker  (Python/Numba path)
                                            rust_quad_ker (Rust path)
```
