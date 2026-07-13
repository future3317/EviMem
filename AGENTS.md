# Maintainer contract

- Use `conda run --no-capture-output -n llm ...` for Python, pytest and Ruff.
- Do not add datasets, downloaded papers, checkpoints or experiment outputs to
  the repository.
- The learned controller may emit only structured `CurationAction` values.
- `REQUEST_PUBLICATION` is a request, never publication authority.
- Verification slots may be changed only by deterministic verifier output.
- Memory admission requires evidence, certificate and policy identity.
- Oracle/gold benchmark annotations must remain invisible during inference.
- Prefer maintained libraries such as Pydantic, scikit-learn,
  Transformers/PEFT/TRL and Accelerate over local reimplementations.
- Do not introduce legacy EviPGCE compatibility adapters.

