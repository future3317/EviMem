**把新论文迁移到独立的 `E:\CODE\EviMem-RL` 是正确决定**，比继续在旧 `Piepaper` 上重构更合理。

旧项目已经形成大量历史模块、实验数据、兼容路径和论文专属逻辑。新论文虽然继承 EviPGCE 的安全发布思想，但研究对象已经从“静态科学数据发布治理”变成了“带证据记忆的序列决策智能体”。两者强行共用同一个仓库，后面会不断出现：

* 为兼容旧论文而保留无用接口；
* 新旧 schema 相互转换；
* 旧实验被新代码意外污染；
* 无法分清某个组件究竟属于旧论文还是新方法；
* reviewer 很难根据开源代码复现新论文的最小闭环。

旧项目本来就存在 V1/V2 共存、多个 observation schema 和 fallback 逻辑，独立建仓正好避免继续背这些历史负担。

不过，需要非常明确地说：

> **Claude 当前完成的是“新项目骨架迁移”，不是 EviMem-RL Methods 已经实现。**

“28 passed”只能说明这套小型骨架内部暂时自洽，不能说明完整科学策展 pipeline 已经正确。

---

# 一、这次迁移做对了什么

## 1. 新旧论文已经物理隔离

现在：

* `E:\CODE\Piepaper`：旧论文 EviPGCE、旧实验和数据；
* `E:\CODE\EviMem-RL`：新论文代码；
* 没有复制数据、checkpoint 和实验结果；
* 新项目不依赖 `src.evipgce`；
* 旧项目没有被继续修改。

这是最关键的一步。

旧论文的核心价值是：LLM 只能提出 candidate，只有 evidence binding、tuple verification、DomainPack validation、conflict resolution 和 publication gate 才能决定是否发布。

新项目应当继承这个**原则和经过验证的最小实现**，而不是继承旧仓库的所有代码。

## 2. 没有伪造 GRPO

Claude 明确说：

> 当前只是可供 Transformers、PEFT、TRL 接入的安全底座，没有假装已经实现和训练 GRPO。

这是正确的。现在直接写一个看似完整的 `grpo_trainer.py`，却没有 episode、真实 reward、轨迹数据和模型训练，只会制造“代码已经做完”的假象。

## 3. 优先建立 canonical contracts

先确定：

* `EvidenceRef`
* `CandidateObservation`
* `ClaimState`
* `VerificationCertificate`
* `WarrantedMemory`
* `CurationTrajectory`

再写 controller 和训练，是正确顺序。

特别是 memory 不能只是字符串摘要。它必须绑定 evidence、certificate、decision 和 policy version，这正是新论文最重要的研究对象。

## 4. controller 不拥有 publication 权限

这条必须永远保留。旧论文的消融已经表明，取消严格 gate 虽然可能提高表面 F1，却会允许大量不可验证记录直接释放。

---

# 二、当前新仓库大概率还缺什么

根据 Claude 的描述，新仓库目前迁移了：

* contracts；
* controller；
* memory；
* RL substrate；
* benchmark；
* human review；
* runtime。

但回复中没有明确说已经完整迁移以下关键部分：

| 必需组件                                      | 当前状态判断            |
| ----------------------------------------- | ----------------- |
| Immutable Evidence Release builder        | 不确定，可能只有 contract |
| Evidence block store                      | 不确定               |
| DomainPack loader 和 validator             | 未明确迁移             |
| Evidence binding cascade                  | 未明确迁移             |
| Tuple-level verifier                      | 未明确迁移             |
| Multi-block distributed evidence verifier | 大概率未实现            |
| Conflict resolver                         | 未明确迁移             |
| Publication gate                          | 声称保留原则，但新仓库实现不明   |
| Atomic publication commit                 | 新仓库是否真正迁移不明       |
| Audit/review store                        | 不明确               |
| 真实 LLM proposer adapter                   | 大概率没有             |
| 旧 Gold 数据只读导入器                            | 没有说明              |
| 真实 sequential episodes                    | 大概率只有结构           |
| Oracle trajectory builder                 | 不明确               |
| Heuristic baseline                        | 不明确               |
| SFT controller                            | 未实现               |
| GRPO training                             | 明确未实现             |
| 三随机种子实验                                   | 未进行               |

所以目前更准确的阶段应当叫：

> **Phase 0A：独立仓库与方法接口初始化**

还不能说 Phase 0/1 已全部完成。

---

# 三、最需要警惕的一点：不要把“迁移接口”误认为“实现方法”

一个典型风险是：

```python
class VerificationCertificate:
    ...
```

已经存在，但真实运行中 certificate 可能只是测试里手工构造出来的，而不是由：

```text
EvidenceRef
→ evidence binding
→ slot verification
→ DomainPack validation
→ conflict resolution
→ publication gate
```

自动生成。

同样：

```python
class WarrantedMemoryStore:
    ...
```

存在，也不意味着它真的实现了：

* certificate-based admission；
* policy-version compatibility；
* supersession；
* conflict memory；
* rejected memory；
* stale memory filtering；
* memory-to-action retrieval。

新仓库下一阶段不是继续增加文件，而是验证：

> **每一个 Methods 名词是否都对应一条真实、可运行、不可伪造的数据路径。**

---

# 四、28 个测试远远不够证明完整 pipeline

28 个测试作为新仓库第一版是正常的，但测试数量与旧项目的 4205 个不能直接比较。

当前至少应补齐以下两个端到端场景。

## 场景 A：正确发布

输入一篇极小 fixture：

```text
BaTiO3 exhibits a d33 value of 190 pC/N at room temperature.
```

系统必须真实完成：

```text
document
→ immutable evidence release
→ candidate
→ ClaimState
→ retrieve/inspect action
→ deterministic slot verification
→ DomainPack validation
→ Verified-Strong certificate
→ publication request
→ atomic publication commit
→ verified memory consolidation
```

最终断言：

* publication store 恰好有一条记录；
* certificate 引用真实 immutable EvidenceRef；
* memory 引用同一 certificate；
* controller 没有调用任何数据库 writer；
* 同一 run 重试不会重复写入。

## 场景 B：错误拒绝

输入：

```text
The predicted d33 may reach 190 pC/N in future optimized samples.
```

或者数值同时邻近多个材料。

系统必须完成：

```text
candidate proposed
→ evidence checked
→ prediction/review/ambiguity detected
→ gate rejected
→ no publication
→ rejection certificate
→ governed rejected memory
```

最终断言：

* publication store 为零；
* audit store 有拒绝记录；
* rejection reason 是确定性产生的；
* rejected memory 可以被后续 episode 检索；
* LLM 自己声称“已验证”不会改变结果。

在这两个场景跑通以前，不要开始 SFT 或 GRPO。

---

# 五、新项目的推荐最终目录

建议新仓库收敛为以下结构，而不是继续增加大量 manager 和 wrapper：

```text
EviMem-RL/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── configs/
│   ├── domains/
│   ├── experiments/
│   └── models/
├── docs/
│   ├── METHODS.md
│   ├── ARCHITECTURE.md
│   └── IMPLEMENTATION_STATUS.md
├── src/evimem/
│   ├── contracts/
│   │   ├── evidence.py
│   │   ├── candidate.py
│   │   ├── claim.py
│   │   ├── certificate.py
│   │   ├── memory.py
│   │   └── trajectory.py
│   ├── evidence/
│   │   ├── release.py
│   │   ├── store.py
│   │   └── retrieval.py
│   ├── domains/
│   │   ├── models.py
│   │   ├── loader.py
│   │   └── validation.py
│   ├── verification/
│   │   ├── binding.py
│   │   ├── tuple_verifier.py
│   │   ├── multi_block.py
│   │   ├── conflicts.py
│   │   └── gate.py
│   ├── publication/
│   │   ├── store.py
│   │   └── commit.py
│   ├── memory/
│   │   ├── store.py
│   │   ├── admission.py
│   │   ├── retrieval.py
│   │   ├── consolidation.py
│   │   └── supersession.py
│   ├── controller/
│   │   ├── actions.py
│   │   ├── state.py
│   │   ├── policy.py
│   │   ├── executor.py
│   │   └── engine.py
│   ├── benchmark/
│   │   ├── episodes.py
│   │   ├── stream.py
│   │   ├── oracle.py
│   │   ├── baselines.py
│   │   └── metrics.py
│   ├── training/
│   │   ├── datasets.py
│   │   ├── imitation.py
│   │   ├── rewards.py
│   │   └── grpo.py
│   └── runtime.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
└── scripts/
    ├── export_legacy_assets.py
    ├── build_benchmark.py
    ├── run_heuristic.py
    ├── train_sft.py
    └── train_grpo.py
```

不建议再建立：

```text
managers/
services/
roles/
handlers/
processors/
coordinators/
```

除非其中确实存在多个可替换实现。否则这些层只会重新制造旧项目的过度设计。

---

# 六、“能用库完成的不自己写”应该如何执行

这个原则是对的，但需要区分**基础设施**和**论文方法**。

## 应当优先使用成熟库

| 需求            | 建议                                     |
| ------------- | -------------------------------------- |
| Schema 与验证    | Pydantic v2                            |
| 配置            | Hydra / OmegaConf，或者简单 YAML + Pydantic |
| 数据表           | PyArrow / Polars                       |
| 事务存储          | SQLAlchemy + SQLite/PostgreSQL         |
| 数据集封装         | Hugging Face Datasets                  |
| 向量检索 baseline | FAISS                                  |
| 模型            | Transformers                           |
| LoRA/QLoRA    | PEFT                                   |
| SFT/GRPO      | TRL                                    |
| 指标            | scikit-learn                           |
| 日志            | Python logging / structlog             |
| 实验记录          | MLflow 或 W&B                           |
| 单元测试          | pytest                                 |
| 代码质量          | Ruff + mypy/pyright                    |

## 不应交给通用框架隐藏的部分

以下是论文方法本身，应明确实现：

* warranted-memory admission；
* memory authority 与 policy compatibility；
* state definition；
* action space；
* action masking；
* verifier-shaped reward；
* conflict-aware memory retrieval；
* termination logic；
* certificate-driven consolidation；
* memory supersession；
* publication safety boundary。

不建议用 LangChain 或 LangGraph 把这些核心过程包进黑盒图节点。那会让：

* 方法不透明；
* reward 难以重放；
* action 定义不严格；
* reviewer 无法判断创新到底在哪里；
* 依赖版本升级后行为发生变化。

---

# 七、旧项目中应该迁移什么

不要整目录复制，只迁移经过审计的最小代码。

## 应迁移

* `EvidenceReleaseManager` 的核心逻辑；
* typed `EvidenceRef`；
* DomainPack schema 和三个 domain config；
* evidence binding 的经过验证部分；
* tuple-level verification；
* conflict resolution；
* strict publication gate；
* atomic idempotent commit；
* Gold benchmark 的只读导出脚本；
* 必要的 negative-control fixtures。

旧项目已经实现了不可变 evidence release、typed locator、原子 commit 和版本化 DomainPack，这些属于新方法的安全基础。

## 不应迁移

* V1/V2 adapters；
* old orchestrator；
* multi-agent roles；
* legacy wrappers；
* 跨模块透传 manager；
* 旧 pipeline CLI；
* 历史兼容 schema；
* 旧论文生成脚本；
* 旧结果数据库；
* 全局可变状态；
* 旧 fallback 常量。

## 数据处理方式

新仓库不复制大数据是正确的，但需要建立：

```text
external_assets.yaml
```

例如：

```yaml
legacy_project:
  root: "E:/CODE/Piepaper"
  evidence_release: "E:/CODE/Piepaper/..."
  gold_benchmark: "E:/CODE/Piepaper/..."
  domain_packs: "E:/CODE/Piepaper/domains"
  read_only: true
```

实验时使用只读路径或导出后的 versioned manifest，不要让新项目修改旧数据。

---

# 八、现在必须立刻做的三件事

## 1. 初始化 Git

新项目还没有 Git，这是当前最危险的问题。

应立即：

```bash
cd E:\CODE\EviMem-RL
git init
git add .
git commit -m "chore: initialize standalone EviMem-RL scaffold"
```

然后创建私有远程仓库并 push。

否则下一次 Claude Code 大规模删除或修改后，很难判断哪些代码被破坏。

## 2. 建立 Methods—Implementation 对照表

新增：

```text
docs/IMPLEMENTATION_STATUS.md
```

格式应为：

| Methods 部分               | 代码位置                  | 状态          | 是否真实运行 | 测试        |
| ------------------------ | --------------------- | ----------- | ------ | --------- |
| EvidenceRelease          | `evidence/release.py` | partial     | 否      | unit only |
| Warranted admission      | `memory/admission.py` | implemented | 是      | 6 tests   |
| Multi-block verification | 无                     | missing     | 否      | 无         |
| GRPO                     | 无                     | missing     | 否      | 无         |

禁止用“底座完成”这种模糊表述。

## 3. 跑通两条真实端到端路径

先实现：

```text
one publishable candidate
one rejected candidate
```

再考虑 benchmark、SFT 和 RL。

---

# 九、可以直接发给 Claude Code 的下一轮任务

```text
请继续处理 E:\CODE\EviMem-RL，但本轮不要实现 SFT、GRPO 或任何模型训练。

目标：把当前 scaffold 升级为真正可运行的 deterministic EviMem-RL Phase 0，而不是继续增加接口或空壳模块。

第一步：审计当前仓库
1. 阅读 docs/METHODS.md 和 docs/ARCHITECTURE.md。
2. 枚举全部源码和测试，查找 pass、TODO、NotImplementedError、mock-only implementation、dummy return、手工构造 certificate、无法执行的 protocol。
3. 新建 docs/IMPLEMENTATION_STATUS.md，逐项对照 METHODS：
   - immutable EvidenceRelease
   - EvidenceRef
   - CandidateObservation
   - ClaimState
   - VerificationCertificate
   - DomainPack
   - evidence binding
   - tuple-level verification
   - conflict resolution
   - publication gate
   - atomic commit
   - warranted-memory admission
   - retrieval
   - consolidation
   - supersession
   - action controller
   - executor
   - trajectory
   - reward
   - benchmark
   - oracle isolation
   对每项标记 IMPLEMENTED / PARTIAL / STUB / MISSING，并给出文件和测试位置。
4. 不得把 dataclass 存在视为功能已实现。

第二步：建立 deterministic 最小闭环
1. 从 E:\CODE\Piepaper 只迁移经过验证且必要的核心逻辑：
   - EvidenceReleaseManager
   - DomainPack schema/config
   - evidence binding
   - tuple verification
   - conflict resolution
   - publication gate
   - atomic idempotent publication commit
2. 迁移后必须改为 src.evimem 原生实现，不允许运行时 import src.evipgce、Piepaper 或 compat adapter。
3. 不复制数据、DB、论文、结果或 checkpoint。
4. 可以创建只读 legacy asset manifest/export script。

第三步：实现两个真实 E2E 测试
A. publishable fixture：
   document → evidence release → candidate → claim state →
   deterministic verification → publication request →
   gate pass → atomic commit → warranted memory。
B. rejected fixture：
   prediction/ambiguous evidence → gate reject →
   zero published records → rejection certificate →
   rejected memory。
必须验证：
- controller 无数据库写权限；
- 只有 commit service 可写 publication store；
- retry 幂等；
- certificate 引用真实 immutable EvidenceRef；
- memory 引用 certificate 和 DomainPack version；
- LLM 自报 verified/published 不影响结果。

第四步：工程要求
1. 优先使用 Pydantic v2、SQLAlchemy、PyArrow/Polars 等成熟库。
2. 不使用 LangChain/LangGraph 隐藏核心 state/action/reward/memory 逻辑。
3. 不新增 manager、wrapper、role 或兼容层，除非存在至少两个真实实现。
4. 删除确认无引用的空壳和重复代码。
5. 初始化 Git，提交当前 scaffold 基线，再分阶段提交本轮修改。
6. 运行 pytest、ruff、compileall、git diff --check。
7. 最终报告必须区分：
   - 已真实运行的功能
   - 仅定义 contract 的功能
   - 尚未实现的训练/实验功能
8. 本轮不宣称 Phase 1、SFT 或 GRPO 完成。
```

---

# 最终评价

Claude 这次做出的**仓库隔离决策是正确的，甚至是必要的**。但是目前的新仓库应被看作：

> 一个干净、方向正确的 EviMem-RL scaffold。

而不是：

> 已经完成 Phase 0/1 的新论文方法实现。

接下来不要继续扩展抽象层，也不要马上训练。先证明下面这条链是真实运行的：

[
\text{Evidence}
\rightarrow
\text{Action}
\rightarrow
\text{Verification}
\rightarrow
\text{Certificate}
\rightarrow
\text{Publication/Reject}
\rightarrow
\text{Governed Memory}.
]

这条链跑通以后，才有资格进入 heuristic baseline、oracle trajectory、SFT controller 和 GRPO。
