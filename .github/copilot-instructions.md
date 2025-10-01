## 🧭 Project context
- **Language:** Python (>=3.10), package: `neuralhydrology`
- **DL stack:** PyTorch
- **Docs:** Sphinx (Read the Docs) from `docs/`
- **Tests:** `pytest` in `test/`
- **Formatting:** `yapf` using repo config at `.style.yapf`
- **CLI tools installed by the package:** `nh-run`, `nh-schedule-runs`, `nh-results-ensemble` (don’t rename) :contentReference[oaicite:0]{index=0}
- **Repo layout highlights:**  
  - `neuralhydrology/` (library code; submodules like `datasetzoo`, `modelzoo`, `training`, `utils`, etc.)  
  - `test/` (unit tests)  
  - `docs/` (Sphinx source)  
  - `examples/`, `environments/` (conda env files) :contentReference[oaicite:1]{index=1}

## ✅ Goals for Copilot
1. Generate Python that follows **.style.yapf** and passes `pytest`. :contentReference[oaicite:2]{index=2}
2. Prefer small, pure utilities; isolate I/O (disk/GPU) behind thin wrappers.
3. Respect the **YAML-config–driven** pattern: add new options via config arguments and validate them; avoid hardcoding run-specific paths. :contentReference[oaicite:3]{index=3}
4. Keep code **device-agnostic** (CPU/GPU/MPS) and avoid assuming CUDA is available. :contentReference[oaicite:4]{index=4}
5. Always prioritize readability and clarity.
6. Write clear and concise comments for each function.
7. Ensure functions have descriptive names and include type hints.

## 🚦 Coding standards
- **Typing:** add `typing` hints; prefer explicit types over `Any`.
- **Style/formatting:** rely on `.style.yapf`; no custom per-file overrides. :contentReference[oaicite:5]{index=5}
- **Errors:** raise informative exceptions including config keys/values involved (no dataset paths with PII).
- **Files & imports:** `snake_case.py`, absolute imports within package.
- **Numerics:** make randomness controllable (seed via caller); avoid global state.
- **Device & performance:** use `torch.no_grad()` where appropriate; move tensors to the right device via helpers; never call `.cuda()` unguarded.

## 🧪 Tests (pytest)
- Put tests in `test/` mirroring module structure.
- Each new public function/class should have at least one fast unit test (no internet, minimal disk).
- Prefer deterministic seeds in tests; skip GPU-only behavior unless truly required.  
- Run locally with `python -m pytest test` before opening a PR. :contentReference[oaicite:6]{index=6}
