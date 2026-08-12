## 总体判断

**这个 idea 是对的，而且比当前 MATMEM 主线明显提升了一个层级。**

当前论文虽然称为 closed-loop，但实际流程是：候选已经进入筛选流程，oracle 返回结果后，系统才决定是否保留该残差；方法并不决定下一次计算哪个材料。因此它更接近“在线残差校正与有界记忆”，而不是完整的闭环发现。

你提出的：

> **Dual-Budget Boundary Memory：在 oracle 预算 (B) 与 memory 容量 (K) 下，联合决定下一次查询什么，以及查询后保留什么。**

确实修复了这个根本问题。我会给它：

* **研究问题价值：8.5/10**
* **目前形式化完整度：6.5/10**
* **修正后成为顶会主线的潜力：8/10**

但现在这句话里存在一个必须先解决的关键漏洞：

> “对剩余候选边界决策的反事实 regret 降低”在在线阶段通常不可直接计算，因为剩余候选的真实 oracle 结果还不知道。

---

# 一、这个 idea 真正有价值的地方

## 1. 它终于让 closed-loop 名副其实

新的闭环是：

[
\text{当前记忆}
\rightarrow
\text{选择下一次 DFT}
\rightarrow
\text{获得真实结果}
\rightarrow
\text{更新有限记忆}
\rightarrow
\text{影响后续 DFT 选择}.
]

这形成了真正的反馈链：

> 记住某类 near-miss，不只是让同一个预测器的 MAE 更低，而是防止系统以后继续把有限的 DFT 预算浪费在相似的假稳定材料上。

因此，memory 的价值最终落在：

* 相同 DFT 预算下发现更多稳定材料；
* 相同稳定发现数量下进行更少的无效计算；
* 相同容量下获得更低的累计决策 regret。

这比“残差检索提升预测精度”更有科学意义。

## 2. 双预算是一个清楚的学术约束

你同时研究：

[
\sum_t C(x_t)\leq B,
\qquad
|M_t|\leq K.
]

其中：

* (B) 限制能够进行多少次昂贵 oracle/DFT；
* (K) 限制能够长期保留多少历史科学观测。

这两个预算不是简单相加，而是互相影响：

* 查询哪个材料，决定未来有哪些信息可以进入 memory；
* memory 保留什么，决定以后会查询哪些材料。

这种耦合关系才是论文真正可能形成新问题的地方。

## 3. Boundary Memory 比普通 coreset 更有材料特色

普通 coreset 追求：

* 整体分布覆盖；
* 重构误差；
* 平均预测误差；
* 梯度匹配。

但材料发现中的核心是：

[
\hat h(x)=\hat E(x)-H_t(x)
]

是否跨过稳定性阈值 (\tau)。

因此，你不是要保存能够重构全部历史的数据，而是保存能够维持**稳定性决策边界**的观测。这一定位比“decision-aware memory”更集中，也更容易发展出理论。

---

# 二、现在最危险的问题：反事实 regret 会泄露未来信息

你现在的表述是：

> oracle 返回后，retention 用“对剩余候选边界决策的反事实 regret 降低”重选记忆。

问题在于，假设剩余候选为 (U_{t+1})，真正的决策 regret 是：

[
R(M)
====

\sum_{x\in U_{t+1}}
\ell\bigl(\hat s_M(x),s(x)\bigr),
]

但 (s(x)) 是剩余候选的真实稳定性，在时间 (t) 尚未经过 DFT，因此在线算法不能知道。

所以必须严格区分两个量。

## 1. Oracle counterfactual utility

在整条历史流全部完成后，离线回放时可以计算：

[
\Delta^{\mathrm{oracle}}_i
==========================

## R_{\mathrm{future}}(M)

R_{\mathrm{future}}(M\cup{i}).
]

它可以用作：

* oracle upper bound；
* 离线监督标签；
* 评估 retention policy 学得好不好；
* 机制诊断。

但它**不能直接用于在线记忆更新**。

## 2. Deployable estimated utility

真正部署时只能计算：

[
\widehat{\Delta}_i
==================

## \widehat R_t(M)

\widehat R_t(M\cup{i}),
]

其中 (\widehat R_t) 只能使用当时可观察的信息：

* 剩余候选结构；
* frozen predictor 输出；
* 当前因果凸包 (H_t)；
* 当前 memory；
* 协议兼容关系；
* 已校准的不确定性；
* 已经观测到的历史 residual。

因此建议不要把在线量直接称为“counterfactual regret reduction”，而改成：

> **observability-safe expected downstream boundary-risk reduction**

中文可以写成：

> **可观测约束下的预期未来边界风险降低。**

ICML 2025 已经存在以 counterfactual covering radius 推导 active-learning 风险上界的工作，所以“counterfactual”一词最好保留给有严格潜在结果定义的量，否则容易引起概念混淆。([Proceedings of Machine Learning Research][1])

---

# 三、“联合 acquisition–retention”不能只是两个模块拼接

这是第二个必须注意的问题。

假设你做成：

* acquisition：普通 uncertainty sampling 或 UCB；
* retention：现有 MATMEM greedy；
* 然后把两者放在同一个循环里。

审稿人仍然会说：

> 这是 active learning 加一个 bounded replay buffer，并非真正的联合优化。

要体现联合性，**acquisition 必须预见到查询结果最终只能在容量 (K) 的 memory 中被使用**。

---

# 四、推荐的正式定义

## 1. 状态

第 (t) 轮状态定义为：

[
S_t =
\left(
U_t,,
M_{t-1},,
H_t,,
b_t
\right),
]

其中：

* (U_t)：尚未查询的候选池；
* (M_{t-1})：当前容量不超过 (K) 的 memory；
* (H_t)：当前因果凸包；
* (b_t)：剩余 oracle 预算。

## 2. 边界风险势函数

定义：

[
\Phi_t(M)
=========

\sum_{x\in U_t}
w_t(x),
\overline{\ell}_t(x;M).
]

其中：

* (w_t(x)) 衡量候选离稳定性边界有多近；
* (\overline{\ell}_t(x;M)) 是 memory 对该候选决策错误的可计算风险上界；
* 上界可以包含结构覆盖距离、残差不确定性、协议迁移误差和校准半径。

例如：

[
w_t(x)
======

\exp\left(
-\frac{
|\hat h_M(x)-\tau|
}{\sigma}
\right).
]

越靠近凸包阈值的候选，权重越高。

更理想的理论方向是证明：

> 在同协议 residual 局部 Lipschitz、校准有效和一定 margin 条件下，未来 false-stable/false-unstable 风险可以由 (\Phi_t(M)) 上界。

这样你优化的就不再是人为设计的 surrogate，而是与最终科学决策直接联系的风险界。

## 3. Retention

oracle 返回新观测 (m_t) 后：

[
M_t^*
=====

\arg\min_{
M'\subseteq M_{t-1}\cup{m_t},
,|M'|\leq K
}
\Phi_{t+1}(M').
]

这表示只保留最能维持剩余候选边界决策的 (K) 条观测。

## 4. Retention-aware acquisition

下一次查询不应只选择“当前最不确定”的候选，而应选择查询后最可能降低未来边界风险的候选：

[
x_t^*
=====

\arg\max_{x\in U_t}
\frac{
\lambda_{\mathrm{disc}}
P_t(s_x=1)
+
\mathbb E_{r_x\sim p_t(r\mid x)}
\left[
\Phi_t(M_{t-1})
---------------

\Phi_{t+1}\bigl(M_t^*(x,r_x)\bigr)
\right]
}{
C(x)
}.
]

它包含两个部分：

1. **发现价值**：候选本身有多大概率稳定；
2. **信息价值**：查询它以后，在容量 (K) 约束下能减少多少后续边界风险。

第二项非常关键。它意味着 acquisition 在选择候选时已经考虑：

> 即使这个候选的信息有价值，有限 memory 是否有空间保留它？它是否会替换掉一条更重要的旧观测？

这才是真正的 acquisition–retention coupling。

---

# 五、你需要避免“泛化的子模优化”陷阱

最近已经存在在线两阶段子模优化工作：先维护一个有限集合，再在后续目标揭示后从中选择子集，并给出在线 regret 和近似保证。因此，仅仅把你的目标写成 submodular function，再给一个 greedy 的 (1-1/e) 保证，创新性仍然不够。

你真正需要的理论不是：

> 我们提出了一个新的子模函数。

而应当是：

> 为什么动态凸包附近的有限 residual memory 能够控制未来材料筛选错误？

最有价值的理论链条应当是：

[
\text{memory 覆盖半径}
\Longrightarrow
\text{residual estimation error}
\Longrightarrow
\text{hull-margin decision error}
\Longrightarrow
\text{discovery regret bound}.
]

另外，NeurIPS 2025 已经研究了 coreset 随系统状态变化而变陈旧，并根据当前 coreset 质量触发重新选择。因此，“动态重选 memory”本身也不是足够独立的创新点；你的差异必须落在**因果凸包、边界风险和双预算耦合**上。

---

# 六、还必须明确：这是有限候选池还是开放世界

“对剩余候选进行优化”默认是一个 **transductive finite-pool setting**：

* 一开始就知道全部候选结构；
* 但不知道它们的 DFT 结果；
* 每轮从剩余候选中选一个查询。

这完全可以接受，而且非常适合作为第一篇论文的设定。

但论文必须诚实地写成：

> bounded-memory pool-based materials discovery

而不能直接声称它解决了未来所有未知材料的开放世界发现。

如果未来候选也会不断到达，则需要把：

[
\sum_{x\in U_t}
]

改成对未来候选分布 (P_{t+1:T}(x)) 的期望，这会引入分布预测和 non-stationarity，第一版不建议同时解决。

---

# 七、最关键的基础实验

## Phase 0：合成闭环 falsification

至少构造五类流：

1. near-boundary residual 有局部相关性；
2. 相同化学族周期性复现；
3. residual 完全随机、memory 理论上无用；
4. 新相进入导致 causal hull revision；
5. 协议变化，直接 residual reuse 会产生负迁移。

在第 3 类中，你的方法应自动退化，而不是无条件优于 baseline。

## 必须比较的基线

* frozen predictor top-ranked acquisition；
* uncertainty acquisition；
* compatible kNN；
* full history；
* FIFO；
* reservoir；
* diversity/k-center；
* residual-priority；
* 固定 acquisition + boundary retention；
* retention-aware acquisition + 简单 memory；
* decoupled acquisition + retention；
* 完整 joint dual-budget 方法；
* 使用真实未来标签计算的 oracle counterfactual retention。

ICML 2025 的科学 active learning 已经使用“预期方差降低”设计 acquisition；多步 BO 也已经将序贯查询写成动态决策问题。因此，单独增加 acquisition function 或 multi-step lookahead 并不足以证明贡献，你需要显示**有限 memory 改变了最优查询策略**。([Proceedings of Machine Learning Research][2])

## 最重要的 GO/NO-GO 条件

在相同 (B,K) 下，完整联合方法必须显著超过：

[
\text{best acquisition}
+
\text{best independently designed retention}.
]

也就是：

[
J_{\mathrm{joint}}

>

J_{\mathrm{decoupled}}.
]

否则实验只能证明：

> active acquisition 有效，或者 boundary memory 有效。

不能证明“联合 acquisition–retention”是必要的。

---

# 八、建议改写后的核心 idea

你现在这段可以改成下面这样：

> **Dual-Budget Boundary Memory** studies pool-based materials discovery under a finite oracle budget (B) and a finite memory capacity (K). At each round, the acquisition policy selects a candidate using only the remaining candidate pool, the current causal hull, and previously observed memory. After the oracle result is revealed, the retention policy reconstructs a capacity-(K) memory by minimizing an observability-safe upper bound on future boundary decision risk. Crucially, acquisition is retention-aware: it estimates not only the immediate probability of discovering a stable material, but also the expected reduction in downstream boundary risk after the queried observation competes for limited memory capacity. Actual future counterfactual regret is used only as an offline oracle and supervision signal, never as an online feature.

中文概括为：

> 在 DFT 预算和记忆容量的双重约束下，系统联合决定“下一次算什么”和“算完后记住什么”。查询策略不仅考虑当前材料被发现为稳定相的概率，还预估该观测在有限 memory 中经过最优保留后，能够降低多少未来凸包边界决策风险；真实未来 regret 只用于离线评估和监督，不进入在线决策。

## 最终结论

**这个方向值得继续，而且是目前最合理的主线。**

不过核心贡献不能停留在：

> acquisition + retention + 两个预算。

必须进一步收敛为：

> **用同一个具有材料物理含义的 boundary-risk potential，同时驱动 acquisition 和 retention，并证明该 potential 与最终 discovery regret 存在联系。**

做到这一点以后，论文才会从“严谨的材料 memory 系统”真正升级为一个新的**资源受限序贯科学发现问题**。

[1]: https://proceedings.mlr.press/v267/wen25b.html "Enhancing Treatment Effect Estimation via Active Learning: A Counterfactual Covering Perspective"
[2]: https://proceedings.mlr.press/v267/kim25m.html "Active Learning with Selective Time-Step Acquisition for PDEs"
