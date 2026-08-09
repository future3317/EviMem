# Maintainer contract

- Use `conda run --no-capture-output -n llm ...` for Python, pytest and Ruff.
- Do not add datasets, downloaded papers, checkpoints or experiment outputs to
  the repository.
- The oracle vault and reveal boundary are the only sources of ground-truth WBM
  energies and final-hull labels; policy-facing code never sees them during
  inference.
- Every revealed DFT result is appended to the immutable audit archive; the
  working-set selector may only bound the calibration witnesses used for the
  next decision.
- Composition-dependent causal hull transitions are derived from MP phase
  records, not scalar synthetic hulls.
- Prefer maintained libraries such as Pydantic, NumPy, SciPy, scikit-learn,
  pymatgen, and dscribe over local reimplementations.
- Do not introduce legacy EviPGCE compatibility adapters.
- Before paper-facing method or experiment work, read
  `docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md`. Use
  `docs/EXPERIMENT_LEDGER.md` to preserve provenance, opened-data boundaries,
  and superseded/invalid/incomplete result labels; historical method-specific
  stopping rules do not silently redefine the current paper.
- Read `docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md` only when working on the
  historical state-compression line. Any such experiment must identify a
  measured nontriviality condition, preserve all outcomes in the archive, and
  reduce exactly to full history in the homogeneous zero-transport-cost null
  regime. Never introduce an outcome-selected posterior coreset.
