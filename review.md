## 结论

你的方向应该从 **EviMem-RL** 正式调整为：

# **EviMem：Evidence-Certified Memory for Continual Scientific Curation**

不再做 RL，也不再把“训练材料领域大模型”作为核心。新的研究问题是：

> **智能体能否将经过证据验证的成功、拒绝和冲突经验写入长期记忆，并利用这些记忆提升后续科学信息抽取，同时避免错误记忆污染数据库？**

现有 150 DOI 不再作为主训练集或主要 benchmark，只保留为最终的真实材料数据库案例研究。主训练和主实验全部基于公开数据。

这是一次合理的方向收缩：你保留了当前代码最有价值的证据验证、certificate 和 publication gate，同时把论文创新集中在 **memory admission、retrieval、update、conflict 与 continual learning** 上。

---

# 一、最终论文不再研究什么

不建议继续研究：

* 材料领域大模型预训练；
* GRPO 或其他强化学习；
* 自由行动的 scientific agent；
* 多智能体角色堆叠；
* 单纯的向量数据库或 GraphRAG；
* 只在 150 DOI 上验证的材料抽取系统。

这些方向或者资源要求高，或者容易被认为是工程组合。

新的论文应该研究三个可学习问题：

[
\text{Should Write}
\rightarrow
\text{What to Retrieve}
\rightarrow
\text{How to Update}
]

也就是：

1. 哪些历史经验值得写入长期记忆；
2. 当前论文应该检索哪些历史记忆；
3. 新证据到来后，旧记忆应当新增、合并、冲突、降权还是失效。

---

# 二、修改后的完整方法

## 2.1 Memory item

每条记忆不再是普通自然语言摘要，而是：

[
m_i =
(c_i,e_i,z_i,d_i,t_i,v_i)
]

其中：

* (c_i)：结构化科学 claim；
* (e_i)：不可变 EvidenceRef；
* (z_i)：VerificationCertificate；
* (d_i)：published、rejected、conflict、ambiguous 等决策；
* (t_i)：时间和来源论文；
* (v_i)：DomainPack 或 schema 版本。

例如：

```json
{
  "memory_type": "rejected",
  "claim": {
    "material": "PZT",
    "property": "d33",
    "value": 190,
    "unit": "pC/N",
    "condition": "room temperature"
  },
  "evidence_refs": ["ev_xxx"],
  "certificate": {
    "status": "rejected",
    "reason": "prediction_not_measurement"
  },
  "source_document": "paper_xxx",
  "policy_version": "piezoelectric@1.3.0"
}
```

这使你的方法与 A-Mem、Mem0、普通 RAG 产生根本区别：已有方法主要组织笔记、对话事实或推理经验，而你的 memory 带有**证据、验证状态和发布权限**。A-Mem 会动态建立笔记属性与链接，Mem0 会从持续交互中抽取、合并和检索重要信息，ReasoningBank 则从成功和失败轨迹中提炼推理策略；这些工作说明“可演化记忆”是重要方向，但没有直接解决科学记录的证据资格和数据库发布安全问题。([NeurIPS 会议录][1])

---

## 2.2 四类记忆

### Verified Memory

已经通过 evidence binding、tuple verification 和 publication gate 的科学记录。

作用：

* 提供历史实体别名；
* 提供常见证据位置；
* 帮助识别重复结果；
* 检测新结果是否真的新颖。

### Rejected Memory

被确定性验证器拒绝的候选，例如：

* 数值属于预测而非实验；
* 属性与材料绑定错误；
* 缺失必要条件；
* 同一数值附近存在多个材料；
* 单位不兼容。

它是论文很重要的创新，因为大部分 memory 方法只保存成功经验，而你的系统利用失败记忆防止智能体重复犯错。

### Conflict Memory

当两个记录具有相同的：

```text
entity + property + condition + measurement setting
```

但值或结论不兼容时建立冲突边，而不是直接覆盖旧记录。

### Superseded Memory

后续证据或人工审查确认旧记录过期后，旧 memory 不删除，而标记：

```text
active → superseded
```

保留完整 lineage。

---

## 2.3 三个可学习模块

### 模块一：Memory Admission

判断当前 episode 是否应写入长期记忆：

[
p_{\text{write}}
================

g_\phi(c,e,z,d)
]

输出：

```text
WRITE_VERIFIED
WRITE_REJECTED
WRITE_CONFLICT
EPHEMERAL_ONLY
IGNORE
```

Certificate 是硬约束：没有证据和证书的 LLM 自我反思不能进入高权限长期记忆。

---

### 模块二：Evidence-Certified Retriever

检索分数不只看文本相似度：

[
S(q,m)=
\alpha S_{\text{semantic}}
+\beta S_{\text{structure}}
+\gamma S_{\text{authority}}
+\eta S_{\text{temporal}}
-\delta S_{\text{conflict}}
-\xi S_{\text{stale}}.
]

其中：

* semantic：语义相似；
* structure：entity、property、condition 是否匹配；
* authority：human-confirmed、verified、rejected 的权限；
* temporal：是否符合当前时间；
* conflict：是否存在未解决冲突；
* stale：是否已经 superseded。

检索结果必须返回 memory 内容以及它的证据和决策状态，而不是只返回一段文本。

---

### 模块三：Typed Memory Update

对于新证据与已有 memory，模型预测：

```text
ADD
MERGE
LINK
CONFLICT
SUPERSEDE
IGNORE
```

例如：

* 完全相同的 tuple：`MERGE`
* 相同材料和属性，不同温度：`LINK`
* 相同条件但数值明显冲突：`CONFLICT`
* 新证据纠正旧错误：`SUPERSEDE`
* 仅主题相似但无关系：`IGNORE`

这一模块比“将新信息 append 到向量库”更有学术价值。

---

# 三、推荐使用的公开数据集

我建议建立一个新的统一 benchmark：

# **SciMem-Curate**

它不是重新人工标注几千篇论文，而是把多个公开科学数据集统一转换成：

```text
document stream
+ scientific claims
+ evidence
+ memory operation
+ final decision
```

## 3.1 核心训练数据

| 数据集                        | 规模和内容                                                                                                    | 在项目中的用途                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **SciREX**                 | 1,170 篇完整科学论文，包含 Method、Metric、Task、Material、Score 等文档级 N-ary relations；代码和数据采用 Apache-2.0。([GitHub][2]) | 主要的结构化科学记录与跨论文 memory 训练                        |
| **QASPER**                 | 1,585 篇 NLP 论文、5,049 个问题，每个答案带 supporting evidence，部分问题需要跨多个章节取证。([arXiv][3])                            | 训练 evidence retrieval 和跨区域证据记忆                  |
| **SciFact**                | 1,409 个科学 claim、5,183 篇候选摘要，带 support/refute 和 rationale。([Hugging Face][4])                             | 训练 verified、rejected 和 conflict memory          |
| **Evidence Inference 2.0** | 3,346 篇临床试验文章、12,616 个 intervention–comparator–outcome prompts，并带 evidence。([ACL Anthology][5])          | 训练条件敏感关系、支持/无差异/反向结论与证据                         |
| **MeasEval**               | 科学文本中的 quantity、measured entity、property、unit、qualifier 和 context 关系。([GitHub][6])                       | 辅助训练 value–unit–entity–property slot extraction |

这几组公开数据合起来已经包含：

* 数千篇科学文章；
* 一万多条有证据的监督任务；
* 支持与反驳；
* 文档级 N-ary relations；
* 跨章节 evidence；
* 数值、单位、属性和条件。

这比使用 150 DOI 训练稳健得多，而且不是依赖一个材料子领域。

---

## 3.2 主要域外测试集

### POLYIE：材料领域域外测试

POLYIE 包含 146 篇完整的 polymer solar cell 和 lithium battery 论文，由专家标注：

```text
Compound Name
Property Name
Property Value
Condition
```

以及它们组成的 N-ary relation，数据与代码采用 Apache-2.0。它与当前 `CandidateObservation` 的 schema 非常接近，因此特别适合做**未在材料数据上训练的 OOD 测试**。([ACL Anthology][7])

不建议将 POLYIE 全部用于训练。最好设置：

* `POLYIE zero-shot OOD`
* `POLYIE few-shot adaptation`
* `POLYIE full supervised upper bound`

这样可以证明你的 memory 方法不是只对 NLP 或生物医学论文有效。

---

### BioRED：生物医学域外测试

BioRED 包含 600 篇 PubMed abstracts，具有多类实体、文档级关系，而且将关系标记为 novel finding 或 background knowledge。这个 novel/background 标注非常适合验证 memory 是否能区分“新发现”和“历史知识”。([arXiv][8])

---

### SciFact-Open：大规模检索压力测试

SciFact-Open 在约 50 万篇研究摘要上评估开放域 scientific claim verification，原论文发现从小规模语料训练的系统迁移到开放域时至少下降 15 F1。它很适合检验：

* memory store 扩展到大规模时的检索性能；
* 错误相似记忆是否污染结果；
* evidence-certified retrieval 是否优于普通向量检索。([arXiv][9])

---

## 3.3 不建议作为主数据集的资源

### SuperMat

虽然主题非常匹配，但其仓库明确说明完整 annotation 因版权原因并不公开，因此不适合作为论文的主要可复现训练集。([GitHub][10])

### MatSci-NLP

MatSci-NLP 提供 NER、relation classification、event extraction、slot filling、synthesis retrieval 等七种材料 NLP 任务，适合作为辅助预训练或补充实验。([GitHub][11])

但仓库说明其子数据来自互联网，并要求分别参考原数据论文，因此各子集的许可证需要逐项审计。它不适合在没有 license manifest 的情况下直接打包成你的主要训练集。([GitHub][11])

---

# 四、如何把这些独立数据集变成 Memory 数据

现有数据集本身大多不是 memory benchmark，因此需要构建 **continual episodes**，但应尽量从真实 annotation 推导，而不是让 LLM 随机生成。

## 4.1 Episode 基本形式

每个 episode：

```yaml
episode:
  history:
    - previous verified/rejected/conflict memories
  current_document:
    text: ...
    timestamp: ...
  query:
    claim_or_candidate: ...
  gold:
    relevant_memories: [...]
    evidence: [...]
    final_record: ...
    memory_operation: ADD | MERGE | LINK | CONFLICT | IGNORE
```

---

## 4.2 SciREX 的转换

每个 N-ary relation：

```text
Method + Task + Material + Metric + Score
```

转成一条 verified record。

按照论文发表年份或固定文献顺序组成 stream。

例如：

```text
Paper 1:
BERT + NER + CoNLL03 + F1 + 91.2

Paper 2:
SciBERT + NER + CoNLL03 + F1 + 92.1
```

第二篇到达时，第一篇 memory：

* 对 task、dataset 和 metric 有帮助；
* 但不能把旧 method 和 score 错误复制过来；
* 两条记录应当 `LINK`，而不是 `MERGE`。

SciREX 有一个已知问题：约一半关系中至少一个实体只出现在被丢弃的表格里，因此端到端实验需要使用官方 filtered relation protocol，并把 table-missing 样本单独作为困难集，而不是误当作普通模型错误。([GitHub][12])

---

## 4.3 SciFact 的转换

SciFact 天然提供：

```text
claim
support evidence
refute evidence
rationale
```

可以转换为：

* support → verified memory；
* refute → rejected/conflict memory；
* 无 evidence → ignore/ambiguous；
* 同一主题的 support/refute pair → conflict episode。

它是训练负向 memory 最重要的数据来源。

---

## 4.4 QASPER 的转换

每个 question 是当前 query，已标注的 supporting paragraphs 是相关 memory/evidence。

构造：

* relevant evidence memory；
* same-paper hard negatives；
* same-topic but non-answer paragraphs；
* multi-section evidence episodes。

这样可以训练 retriever，而不用人工制造相关性标签。

---

## 4.5 Evidence Inference 的转换

其结构：

```text
Intervention
Comparator
Outcome
Result direction
Evidence
```

适合构造条件敏感 memory：

```text
Drug A > Drug B on Outcome X
Drug A = Drug B on Outcome Y
Drug A < Drug B under Population Z
```

模型必须保留 outcome、population 和 condition，不能只因药物名称相同就合并。

---

## 4.6 Rejected memory 的生成原则

优先使用真实标注：

* SciFact 的 refute；
* Evidence Inference 的方向标签；
* 数据集中明确的 no-evidence；
* deterministic gate 对模型候选的真实拒绝。

可以生成 hard negative，但只能做以下受控变换：

* 同一 value，替换 entity；
* 同一 entity，替换 condition；
* 同一 property，替换 unit；
* 相同主题但错误 evidence。

这些必须标记为 `controlled corruption`，不能伪装成天然论文冲突。

---

# 五、训练方案：不做 RL

## Stage 1：训练 Memory Retriever

使用 bi-encoder 对 query 和 memory 编码：

[
h_q=f_\psi(q), \qquad h_m=f_\psi(m)
]

采用 contrastive loss：

[
\mathcal{L}_{ret}
=================

-\log
\frac{\exp(s(q,m^+)/\tau)}
{\exp(s(q,m^+)/\tau)+
\sum_j \exp(s(q,m_j^-)/\tau)}.
]

正样本：

* gold supporting evidence；
* gold relevant relation；
* 相同实体与正确上下文的历史记录。

难负样本：

* 相同实体但不同属性；
* 相同属性但不同条件；
* 相同数值但不同材料；
* superseded 或 conflict memory。

这一阶段可使用 `sentence-transformers` 和 FAISS 完成，单张 4090 足够。

---

## Stage 2：监督训练 Memory Manager

输入：

```text
current candidate
+ verification certificate
+ retrieved memories
+ current evidence
```

输出：

```json
{
  "admission": "WRITE_REJECTED",
  "update_operation": "CONFLICT",
  "target_memory_ids": ["mem_123"],
  "reason_code": "same_context_incompatible_value"
}
```

监督损失：

[
\mathcal{L}_{mem}
=================

\mathcal{L}*{admission}
+
\lambda_u\mathcal{L}*{update}
+
\lambda_t\mathcal{L}*{type}
+
\lambda_r\mathcal{L}*{reason}.
]

可以使用 3B–7B 开源 instruct 模型做 QLoRA，不训练完整领域大模型。

---

## Stage 3：Memory-Conditioned Scientific Curation

固定同一个 proposer，在不同 memory 方法下运行：

[
\hat{y}
=======

f_\theta(D,\operatorname{Retrieve}(D,\mathcal M)).
]

检索到的 memory 不作为“绝对事实”直接复制，而被组织成：

```yaml
verified_precedents:
  - ...
known_failure_patterns:
  - ...
possible_conflicts:
  - ...
required_checks:
  - ...
```

之后仍由当前证据和 deterministic gate 决定是否发布。

总训练目标可以写为：

[
\mathcal{L}
===========

\mathcal{L}*{extract}
+
\lambda_1\mathcal{L}*{ret}
+
\lambda_2\mathcal{L}*{admission}
+
\lambda_3\mathcal{L}*{update}
+
\lambda_4\mathcal{L}_{conflict}.
]

---

# 六、必须对比的方法

## 6.1 Memory 对比方法

| 方法                      | 作用                                                        |
| ----------------------- | --------------------------------------------------------- |
| **No Memory**           | 每篇论文独立处理                                                  |
| **Full History**        | 将全部历史直接塞进上下文                                              |
| **BM25 Memory**         | 关键词检索                                                     |
| **Dense Vector Memory** | embedding + top-k                                         |
| **Summary Memory**      | 将历史压缩成摘要                                                  |
| **Mem0**                | 动态抽取与合并长期记忆                                               |
| **HippoRAG**            | 知识图和 PageRank 式关联检索                                       |
| **A-Mem**               | 动态 note、tag、link 和 memory evolution                       |
| **ReasoningBank**       | 保存成功与失败中提炼的策略                                             |
| **EviMem**              | evidence + certificate + typed decision + governed update |

LongMemEval 表明长期记忆不仅需要事实检索，还涉及多会话推理、时间信息、知识更新和拒答；MemoryAgentBench进一步将 memory 能力概括为准确检索、test-time learning、长程理解和冲突处理。这说明只比较一个向量 RAG baseline 不足以支撑 ICLR 投稿。([OpenReview][13])

---

## 6.2 科学信息抽取对比方法

保持同一数据输入，还应比较：

* SciBERT；
* MatSciBERT；
* DyGIE++；
* PURE；
* document-level IE baseline；
* zero-shot LLM；
* few-shot LLM；
* full-text LLM；
* evidence-RAG LLM。

POLYIE 官方仓库本身提供了 BERT NER、DyGIE++、PURE 和 GPT 类 baseline，可以优先复用其公开实现与官方评测协议。([GitHub][14])

最公平的比较方式是：

> **固定相同的 base LLM、candidate proposer、token budget 和 publication gate，只替换 memory 模块。**

否则 reviewer 会认为提升来自不同模型，而不是 memory。

---

# 七、最终实验划分

## 7.1 训练

```text
SciREX train
QASPER train
SciFact train
Evidence Inference train
MeasEval train
```

## 7.2 ID 验证与测试

使用各数据集官方 dev/test，不打乱原始划分。

## 7.3 OOD 测试

```text
POLYIE
BioRED
```

不在这两个数据集上训练主 memory 模型。

## 7.4 大规模检索测试

```text
SciFact-Open 500K corpus
```

## 7.5 真实世界案例

```text
原来的 150 DOI Gold benchmark
```

它只作为：

* real-world material database case study；
* deterministic publication safety 验证；
* 与旧 EviPGCE 的连接。

不要再用它来训练，也不要把它作为唯一主结果。

---

# 八、需要报告的指标

## 科学记录质量

* tuple precision / recall / F1；
* evidence span F1；
* Published Observation F1；
* Verified-Strong recall；
* unsupported publication rate；
* negative-control false publication。

## Memory 本身

* Recall@1 / Recall@5 / Recall@10；
* MRR / nDCG；
* memory admission precision；
* ADD/MERGE/LINK/CONFLICT/IGNORE accuracy；
* conflict resolution accuracy；
* repeated-error reduction；
* stale-memory error rate；
* memory pollution robustness。

## 持续学习

定义：

[
\Delta_{\text{memory}}
======================

## F1_{\text{with memory}}

F1_{\text{without memory}}.
]

还要报告：

* 随着 stream 增长，性能是否提高；
* memory size；
* 每篇论文检索 token 数；
* 历史错误是否持续传播；
* 新 DomainPack/schema 下旧 memory 是否失效。

---

# 九、ICLR 论文应当怎样重新表述

## 新标题

**EviMem: Evidence-Certified Memory for Continual Scientific Curation**

或者：

**Learning What to Remember: Evidence-Governed Memory for Continual Scientific Information Extraction**

## 新的四个主要贡献

### 贡献一：新问题

把 scientific information extraction 从独立文档预测，重新定义为持续文献流中的 memory-based curation。

### 贡献二：新 memory 对象

提出 evidence-certified memory，每条记忆携带：

* structured claim；
* immutable evidence；
* verification certificate；
* decision status；
* time/version。

### 贡献三：新 memory 学习机制

学习：

* admission；
* retrieval；
* typed update；
* conflict resolution。

但 publication 仍由确定性 gate 控制。

### 贡献四：公开多领域 benchmark

将 SciREX、QASPER、SciFact、Evidence Inference、MeasEval、POLYIE 和 BioRED 转换为统一的 continual scientific memory benchmark。

---

# 十、这个方向的关键风险

## 风险一：只做规则型 memory store

假如最终只是：

```text
certificate pass → store
certificate fail → reject
```

这仍然更像系统工程，不够 ICLR。

必须至少有两个可学习组件：

* memory retriever；
* memory admission/update model。

---

## 风险二：人为构造的 stream 太强

主结果应尽量来自自然 annotations 和真实 publication ordering。

人工 corruption 只能作为 memory stress test，不能成为全部训练和测试来源。

---

## 风险三：memory 只提升 retrieval，不提升最终科学记录

必须证明 memory 能够：

* 提高 tuple F1 或 Published F1；
* 减少重复错误；
* 更准确地区分新记录、重复记录和冲突记录；
* 在不提高错误发布率的情况下提高召回率。

---

## 风险四：跨数据集 schema 不统一

不要粗暴把所有数据变成一段 instruction 文本。

应先建立统一的最小 schema：

```text
subject
relation/property
object/value
unit
condition
evidence
decision
timestamp
source
```

不同数据集缺失的字段允许为 `null`，但不能伪造。

---

# 十一、推荐的下一阶段

当前 deterministic Phase 0 已经足够，不要继续扩展 publication 工程。

接下来的正确顺序是：

1. 将项目从 `EviMem-RL` 重命名或在论文层面改为 `EviMem`；
2. 增加公开数据下载与 license audit；
3. 建立统一 `ScientificMemoryRecord`；
4. 实现 SciREX、QASPER、SciFact、Evidence Inference、MeasEval adapter；
5. 构建无 future leakage 的 stream；
6. 先跑 NoMem、BM25、Dense Memory；
7. 再实现 certificate-aware retriever；
8. 再训练 supervised admission/update model；
9. 最后接 POLYIE、BioRED 和 150 DOI 做 OOD/真实案例。

最小可发表版本不需要 RL，也不需要训练领域大模型：

[
\boxed{
\text{Public Scientific Data}
+
\text{Learned Certified Memory}
+
\text{Continual Evaluation}
+
\text{Deterministic Safety Gate}
}
]

这会比原来的 EviMem-RL 更聚焦，也更适合单张 4090，并且更容易回答 ICLR reviewer 最核心的问题：

> **你的 memory 方法究竟学会了什么，以及它为什么比普通 RAG、A-Mem 和 Mem0 更可靠？**

可以设置每周检索 ICLR、ICML 和 NeurIPS 新出现的 agent-memory 工作，持续更新 baseline 列表和 related work。

[1]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html?utm_source=chatgpt.com "A-Mem: Agentic Memory for LLM Agents"
[2]: https://github.com/allenai/SciREX/blob/master/Statistics.md "SciREX/Statistics.md at master · allenai/SciREX · GitHub"
[3]: https://arxiv.org/abs/2105.03011?utm_source=chatgpt.com "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers"
[4]: https://huggingface.co/datasets/allenai/scifact?utm_source=chatgpt.com "allenai/scifact · Datasets at Hugging Face"
[5]: https://aclanthology.org/anthology-files/pdf/bionlp/2020.bionlp-1.13.pdf?utm_source=chatgpt.com "Evidence Inference 2.0: More Data, Better Models"
[6]: https://github.com/harperco/MeasEval "GitHub - harperco/MeasEval: SemEval-2021 Task 8: MeasEval data and other bits · GitHub"
[7]: https://aclanthology.org/2024.naacl-long.131/ "POLYIE: A Dataset of Information Extraction from Polymer Material Scientific Literature - ACL Anthology"
[8]: https://arxiv.org/abs/2204.04263?utm_source=chatgpt.com "BioRED: A Rich Biomedical Relation Extraction Dataset"
[9]: https://arxiv.org/abs/2210.13777?utm_source=chatgpt.com "SciFact-Open: Towards open-domain scientific claim verification"
[10]: https://github.com/lfoppiano/SuperMat "GitHub - lfoppiano/SuperMat: Superconductors material dataset · GitHub"
[11]: https://github.com/BangLab-UdeM-Mila/NLP4MatSci-ACL23/tree/main/dataset "NLP4MatSci-ACL23/dataset at main · BangLab-UdeM-Mila/NLP4MatSci-ACL23 · GitHub"
[12]: https://github.com/allenai/SciREX "GitHub - allenai/SciREX: Data/Code Repository for https://api.semanticscholar.org/CorpusID:218470122 · GitHub"
[13]: https://openreview.net/pdf?id=pZiyCaVuti&utm_source=chatgpt.com "LONGMEMEVAL: BENCHMARKING CHAT ASSIST"
[14]: https://github.com/jerry3027/PolyIE "GitHub - jerry3027/PolyIE · GitHub"
