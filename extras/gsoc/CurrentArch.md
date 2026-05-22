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

**File:** `src/eko/runner/managed.py` : `solve(theory, operator, path)`

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

---

## 3. Operator vs OperatorMatrixElement

After the path is decomposed into segments and matchings by the `Atlas`, two
different compute objects handle the integration:

| Class | File | Purpose |
| --- | --- | --- |
| `Operator` | `evolution_operator/__init__.py` | DGLAP evolution between two scales within a fixed-nf region |
| `OperatorMatrixElement` | `evolution_operator/operator_matrix_element.py` | Heavy-quark matching condition at a flavor threshold |

`OperatorMatrixElement` inherits `Operator` and they share the same `integrate()` / `run_op_integration()` machinery; they differ only in which kernel function and labels they use.

---

## 4. Integration loops

### 4.1 Outer loop

`Operator.integrate()` iterates over every point `(k, logx)` in the output x-grid. Each point is independent and hence the problem is embarrassingly parallel (currently via `multiprocessing.Pool`).

```python
# evolution_operator/__init__.py  (line 997)
with pool:
    results = pool.map(self.run_op_integration, log_grid)
```

### 4.2 Inner loop

Inside `run_op_integration`, for each target point the code iterates:

1. **Source basis function `j`** — each `BasisFunction` carries the polynomial coefficients for one source x-node (`areas_representation`).
2. **Flavor label** — a `(mode0, mode1)` pair identifying which element of the operator matrix is being computed (e.g. `(100, 100)` for quark-singlet → quark-singlet).

For each `(j, label)` pair a separate `scipy.integrate.quad` call is made.
I had a thought to parallelize this (i.e. remove the two loops), but the thing is python has GIL, and outer loop is already parallelized, so it may not improve / might deteriorate the performance. Also, we would have to create a separate cfg for each process, thus increasing the memory usage.

---

## 5. scipy.integrate.quad

### 5.1 quad_ker

```text
scipy.integrate.quad(quad_ker_partial, 0.5, 1-ε)
       ↓  calls func at each quadrature node
quad_ker(u, order, mode0, ...) [quad_ker.py, @nb.njit]
       ↓
QuadKerBase  →  integrand
       ↓
quad_ker_qcd / quad_ker_qed  →  anomalous dimensions via ekore
       ↓
kernels/singlet.py | non_singlet.py | ...  →  evolution operator matrix
       ↓
np.real(ker * integrand)  →  returned float
```

`quad_ker` is decorated `@nb.njit`, meaning Numba compiles it to machine code.  
`scipy` calls the resulting function through Python's normal calling convention on every quadrature node this carries Python overhead even with Numba.

### 5.2 rust_quad_ker

```text
scipy.integrate.quad(LowLevelCallable(rust_quad_ker, &cfg), 0.5, 1-ε)
       ↓  scipy's Fortran/C backend calls the C function pointer directly
rust_quad_ker(u, *args)  [crates/eko/src/lib.rs]
       ↓
TalbotPath  →  Mellin contour + Jacobian
       ↓
ekore Rust  →  anomalous dimensions
       ↓
Python callback  →  cb_quad_ker_qcd / cb_quad_ker_qed
       ↓
Rust
       ↓
f64 returned to scipy
```

**Current transitional state:** `scipy → Rust → Python callback → Rust → scipy`

**Target state:** `scipy → Rust → Rust → scipy` (no Python in the hot loop)

### Why LowLevelCallable, not PyO3?

`scipy.integrate.quad` accepts a `LowLevelCallable`, a raw C function pointer paired with a user-data pointer.  
When given an LLC, scipy's underlying Fortran/C QUADPACK routines call the function pointer directly, without ever entering the Python runtime.  
There is no GIL acquisition, no Python frame allocation, and no argument marshalling through Python objects on each of the ~100–10000 evaluations per integral.  

PyO3 wraps Rust functions as ordinary Python callables. If the integration
kernel were exposed via PyO3, the call chain on every point would be:

```text
scipy Fortran  →  acquire GIL  →  Python call dispatch  →  PyO3 wrapper  →  Rust  →  release GIL  →  scipy Fortran
```

This overhead per node dominates at the scale of `O(grid_size²) × O(flavors) × O(100)` calls made in a typical EKO run.

**Conclusion:** PyO3 is the right tool for the Python-Rust API boundary (configuration, result retrieval). LLC is the right tool for the integration loop. The architecture uses both in their appropriate roles.
