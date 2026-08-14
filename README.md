<div align="center">
  <h1>Globally Adjudicated Active Search</h1>
  <p><strong>Globally Adjudicated Active Search</strong></p>
  <p>面向全局判定效用与延迟结构化标签的主动搜索研究代码库</p>
  <p>
    <img alt="状态 Status：Research Software 研究软件" src="https://img.shields.io/badge/Status-Research%20Software%20%7C%20研究软件-6f42c1">
    <img alt="许可证 License：Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0%20%7C%20Apache--2.0-b91c1c">
    <img alt="检查 Checks：pytest 与 Ruff" src="https://img.shields.io/badge/Checks-pytest%20%7C%20Ruff%20%7C%20测试--静态检查-2f855a">
    <img alt="基准契约 Benchmark Contract：协议约束 Protocol Bound" src="https://img.shields.io/badge/Benchmark%20Contract-Protocol%20Bound%20%7C%20契约约束-2b6cb0">
  </p>
</div>

---

> **中文定位**：研究“查询立即得到观测、但最终效用由完整候选池的全局判定器延迟决定”的主动搜索问题。项目聚焦 selected-item utility 与 acquisition objective alignment，而不是堆叠更多规划器或追求无条件的通用最优性。
>
> **English positioning**: This repository is a research codebase for active search with immediate observations and delayed, globally adjudicated utility. Its central theme is objective alignment for selected-item utility under a shared posterior and evaluator, not a claim of universally superior planning.

## Research story | 科研主线

The repository supports one coherent scientific story:

1. Formalize active search when a query reveals an immediate target observation,
   while the selected item's final label is assigned only after global
   adjudication over the complete latent pool.
2. Use the adjudicated terminal utility to define the aligned greedy reference
   policy, then compare it with history-local objective controls.
3. Test whether additional planning complexity contributes terminal utility
   after the acquisition objective has already been aligned.

The empirical work is organized around matched controls: adjudicator choice,
optimized-object choice, and planning headroom. Numerical convergence checks
and implementation diagnostics are validation layers; they are not additional
headline methods. Results, data boundaries, and analysis identities are governed
by the current experiment contract rather than by historical exploratory runs.

## Capability overview | 能力概览

- **Oracle-isolated active search**：policy-facing code receives legal
  observations only; unrevealed target outcomes and final labels remain outside
  the policy state.
- **Delayed structured-label evaluation**：selected-item utility is evaluated
  after complete-pool adjudication, with composition-dependent hull geometry
  preserved by the evaluator.
- **Posterior-driven acquisition**：shared posterior, candidate-pool semantics,
  and deterministic policy identities support matched objective and adjudicator
  comparisons.
- **Reproducible protocol execution**：immutable reveal archives, cross-fitting
  manifests, explicit protocol identities, and evaluator-side summaries keep
  data provenance separate from online decision making.
- **Research diagnostics**：the repository contains focused analysis and figure
  tooling for mechanism checks, calibration, numerical convergence, and paper
  artifacts without treating every historical diagnostic as a public method.

## Research status | 当前阶段

This repository is research software accompanying an active-methods study. It
follows open-source project conventions while keeping large or sensitive
artifacts out of the source tree: raw data, oracle vaults, checkpoints,
generated traces, and experiment outputs are managed separately from Git.

The code is intended to be readable, testable, and reproducible within the
data and protocol boundaries documented below. It is a research artifact rather
than a production service or a claim of a universally optimal planner.

## Repository map | 仓库结构

```text
src/matmem/     policy state, posterior, acquisition, and evaluator modules
tools/          protocol runners, audits, summarizers, and paper-figure tools
tests/          unit, protocol, and regression tests
docs/           experiment contract, provenance ledger, and protocol notes
```

The primary implementation boundaries are:

- `src/matmem/protocol_closed_loop.py` — legal policy state, append-only
  reveals, oracle separation, and closed-loop execution;
- `src/matmem/posterior.py` and `src/matmem/transport.py` — the registered
  working posterior and transport structure;
- `src/matmem/hull_geometry.py` and `src/matmem/hull_engine.py` — fixed-
  composition hull primitives and evaluator logic;
- `src/matmem/protocol_acquisition.py` and
  `src/matmem/policy_registry.py` — acquisition policies and policy identity;
- `tools/` — controlled analysis and publication-artifact entry points.

## Checks | 检查

Use the project-specific `llm` environment. Do not use Conda `base` for project
execution, tests, or dependency installation.

```powershell
conda run --no-capture-output -n llm pytest -q
conda run --no-capture-output -n llm ruff check src tests tools
```

The checks above validate the tracked source and tests only. They do not create
or fetch datasets, oracle vaults, checkpoints, or experiment outputs.

## Benchmark Contract | 基准契约

All paper-facing experiments must follow the active contract and preserve its
opened-data boundaries, policy identities, evaluator semantics, and provenance
labels:

- [Current paper and experiment contract](docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md)
- [Experiment ledger](docs/EXPERIMENT_LEDGER.md)
- [Protocol index](docs/PROTOCOL_INDEX.md)

In particular, a MatPES result is development evidence unless the contract
explicitly authorizes a separate held-out rerun. A held-out rerun is not called
external or untouched unless its systems and pair IDs are genuinely disjoint
and the analysis was frozen before outcomes were opened. This repository does
not advertise a public benchmark score.

## Data and provenance | 数据与溯源

- Policy-facing code never receives unrevealed target energies or final labels.
- Every legal reveal is retained in the immutable audit archive and conditions
  the registered posterior.
- Composition-dependent hull transitions are derived from registered phase
  records, not scalar synthetic hulls.
- Raw datasets, vaults, checkpoints, generated traces, and summaries remain
  outside Git by project policy.

## License | 许可证

This project is licensed under the [Apache License 2.0](LICENSE). You may use,
modify, and redistribute the code under its terms. The license includes a
patent grant and an explicit warranty disclaimer; see the full text in
[`LICENSE`](LICENSE).

## Citation and contact | 引用与联系

Citation information will be added with the associated manuscript release.
When reusing the code or reporting results, cite the released paper and keep
the benchmark-contract version with the reported experiment identity.
