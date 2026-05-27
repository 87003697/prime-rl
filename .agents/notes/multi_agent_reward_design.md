# Multi-Agent Reward Design: Generator-Verifier + Tool-use Chain

> 场景：Generator 通过 tool-use chain 生成解答，Verifier 判断对错，双方对抗训练。

## 核心挑战

1. **Verifier 退化** — 全 accept/reject 都是 trivial equilibrium
2. **Tool-chain credit assignment** — 多步 tool call，哪步该拿分？
3. **对抗稳定性** — live verifier 做 reward 会导致 non-stationary + exploit

---

## 综述与 Paper Lists

| 资源 | 链接 | 说明 |
|------|------|------|
| **The Landscape of Agentic RL for LLMs** | [arXiv:2509.02547](https://arxiv.org/abs/2509.02547) | 100 页全景综述，Oxford+上海AI Lab，覆盖 Multi-Agent / Tool Use / Credit Assignment |
| **ToolRL: Reward is All Tool Learning Needs** | [arXiv:2504.13958](https://arxiv.org/abs/2504.13958) | Tool-use reward 设计专项，format+correctness 分层，[代码](https://github.com/qiancheng0/ToolRL) |
| **A Survey on Post-Training of LLMs** | [arXiv:2503.06072](https://arxiv.org/abs/2503.06072) | RLHF→RLAIF→DPO 演进，Self-Evolved Reward Learning |
| **Awesome-Agent-RL** | [GitHub](https://github.com/tongjingqi/Awesome-Agent-RL) | 按 5 类组织的 reward construction paper list |
| **Awesome-AgenticLLM-RL-Papers** | [GitHub](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers) | 配套大综述的 500+ 论文列表 |

### 🆕 2026 年综述（4 月以后）

| 资源 | 链接 | 说明 |
|------|------|------|
| **Beyond Individual Intelligence (LIFE)** | [arXiv:2605.14892](https://arxiv.org/abs/2605.14892) | Multi-Agent LLM 最全面综述：LIFE 框架（Lay→Integrate→Find faults→Evolve），含 Who&When failure benchmark |
| **The End of Reward Engineering** | [arXiv:2601.08237](https://arxiv.org/abs/2601.08237) | 范式转变论文：从手工数值 reward → LLM 生成 semantic reward specification |
| **MARL for LLM Teams** | [arXiv:2605.02801](https://arxiv.org/abs/2605.02801) | LLM 团队的 MARL 综述：6 种组织架构 + 8 种 reward 类型分类 + orchestration trace |
| **Externalization as Design Principle** | [arXiv:2604.08224](https://arxiv.org/abs/2604.08224) | 54 页，上海交大。Agent 可靠性靠外部认知基础设施（memory/skills/protocols）而非模型本身 |
| **Unified View of Post-Training** | (Huawei/Nankai, 2604) | 统一视角看 SFT/DPO/RL：support set expansion + policy reshaping |
| **SHARP: Shapley Credit for MAS** | [arXiv:2602.08335](https://arxiv.org/abs/2602.08335) | 博弈论 Shapley value 做 multi-agent credit assignment，global + marginal + tool-process reward |
| **A Survey of RL for Large Reasoning Models** | [alphaXiv:2509.08827](https://www.alphaxiv.org/resources/2509.08827) | RLVR 三支柱：Reward Design / Policy Optimization / Sampling Strategies（持续更新至 2026） |

### 2026 权威机构论文（Agent RL 相关）

| 论文 | 链接 | 机构 | 说明 |
|------|------|------|------|
| **SWiRL** | [arXiv:2504.04736](https://arxiv.org/abs/2504.04736) | **Google DeepMind** | Step-Wise RL：合成数据 + Gemini 1.5 Pro 做 process judge + step-wise reward。+21.5% GSM8K，跨任务零样本迁移 +16.9% |
| **SRL (Supervised RL)** | ICLR 2026 | **Google** | 分解 expert trajectory 为 step-wise dense similarity reward，超 SFT+RLVR。解决 sparse reward |
| **ARTIST** | [arXiv:2505.01441](https://arxiv.org/abs/2505.01441) | — | 统一 Agentic Reasoning + Tool Integration via RL，自主决定 when/how/which tool，+22%。"Deliberative tool calls"：少而精优于频繁调用 |
| **CodeGym** | [arXiv:2509.17325](https://arxiv.org/abs/2509.17325) | — | 合成 POMDP 环境做 generalizable tool-use RL，80K+ instances，τ-Bench +8.7%。[代码](https://github.com/StigLidu/CodeGym) |
| **Agent-R1** | [arXiv:2511.14460](https://arxiv.org/abs/2511.14460) | USTC | End-to-end RL for multi-turn agent：Extended MDP + **action masking**。GRPO/PPO 3× baseline。[代码](https://github.com/0russwest0/Agent-R1) |
| **ResRL** | [arXiv:2605.00380](https://arxiv.org/abs/2605.00380) | — | "Precision penalization"：只惩罚 error-specific tokens（via semantic residual），保留 creativity |
| **SCALELOGIC** | [arXiv:2605.06638](https://arxiv.org/abs/2605.06638) | — | Power-law scaling：训练成本 vs 推理深度的定量关系 |
| **Demystifying Agentic RL** | [2510.11701](https://emergentmind.com/articles/2510.11701) | — | 实践指南：data/algorithm/reasoning-mode 三维度优化 |

### 2026 综述中的关键发现

**MARL for LLM Teams (2605.02801)** 分类了 8 种 reward 家族：

```
1. Shared team reward（全组共享最终结果）
2. Role-specific reward（按角色分别定义）
3. Process reward（过程步骤打分）
4. Communication reward（沟通质量）
5. Contribution-based reward（Shapley/marginal）
6. Adversarial reward（对抗信号）
7. Curriculum reward（动态难度调节）
8. Semantic reward（NL-based，LLM 生成）
```

**LIFE 框架 (2605.14892)** 的核心数据：
- Failure attribution 现状：agent 定位 ~53.5%，step 定位 ~14.2%（远未解决）
- Self-evolution 的 alignment 风险：模型在自我进化中偏离人类价值观

**SHARP (2602.08335)** 的三层 reward 分解：
```
R_total = R_global（team 结果）
        + R_shapley（每个 agent 的边际贡献，via Shapley value）
        + R_tool-process（工具执行效率）
```

**Open Challenges (2026 综述共识)**：
1. Message-level credit（对话中单条消息的贡献归因）
2. Anti-factual ambiguity（"不做某事"的贡献如何衡量）
3. Automated reward weighting（超越手动调权重）
4. 100+ agent 的 scalability

---

### 2026 新算法（从综述中提取）

#### Tool-Integrated Reasoning (TIR)

| 系统 | 链接 | 核心贡献 |
|------|------|---------|
| **MatchTIR** | [GitHub](https://github.com/quchangle1/MatchTIR) | **Bipartite matching** 做 turn-level reward：区分有效 tool call vs 冗余 call。4B 超 8B baseline |
| **ArenaRL** | [arXiv:2601.06487](https://arxiv.org/abs/2601.06487) | **Tournament-based relative ranking** 替代 scalar reward，解决 open-ended "discrimination collapse"。O(N) |
| **ReTool** | [arXiv:2504.11536](https://arxiv.org/abs/2504.11536) | Strategic tool use via PPO，outcome-only reward 涌现 code self-correction。AIME 72.5%。[代码](https://github.com/ReTool-RL/ReTool) |
| **CoScale-RL** | [arXiv:2601.14695](https://arxiv.org/abs/2601.14695) | Co-scale data（per-problem 多解）和 computation（rollout samples），稳定 hard task 训练 |
| **Search-R2** | [arXiv:2602.03647](https://arxiv.org/abs/2602.03647) | Actor-Refiner 协作：actor 生成 trajectory + refiner 修正，hybrid reward 做 fine-grained credit |
| **MATTRL** | [arXiv:2601.09667](https://arxiv.org/abs/2601.09667) | Multi-agent test-time RL：turn-level credit + textual experience pool injection |
| **SRL (Google, ICLR 2026)** | — | Supervised RL：分解 expert trajectory 为 step-wise dense similarity reward，超 SFT 和 RLVR |

#### Multi-Agent 协作

| 系统 | 链接 | 核心贡献 |
|------|------|---------|
| **LCA** | [arXiv:2502.03723](https://arxiv.org/abs/2502.03723) | LLM 分析 team goal + individual action → 生成 dense agent-specific reward。超越 value decomposition |
| **MAC-SPGG** | [arXiv:2508.02076](https://arxiv.org/abs/2508.02076) | 公共物品博弈：sequential contribution + reward redesign 消除 free-riding |
| **CoMAS** | [arXiv:2510.08529](https://arxiv.org/abs/2510.08529) | 纯靠 agent 间交互产生的 intrinsic reward 做 co-evolution |

---

## 相关系统 Landscape

### Self-Play / Co-evolution（自我博弈）

| 系统 | 架构 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[Absolute Zero](https://arxiv.org/abs/2505.03335)** | Proposer + Solver（同一模型两角色） | Proposer: learnability reward（中等难度最高分）；Solver: 环境验证的 binary accuracy | **零人类数据**，靠代码执行器提供 ground truth，7B +15.2% on math。[代码](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) |
| **[R-Zero](https://arxiv.org/abs/2508.05004)** | Challenger + Solver（共进化） | Challenger: uncertainty reward `1-2|p̂-0.5|`（Solver 50% 正确率时最高）+ 重复惩罚(BLEU)；Solver: majority vote pseudo-label | 无外部验证器，用 self-consistency 做 pseudo-label。[代码](https://github.com/Chengsong-Huang/R-Zero) |
| **[MAE](https://arxiv.org/abs/2510.23595)** | Proposer + Solver + Judge（单 LLM 三角色） | Proposer: quality(Judge) + difficulty(1-Solver成功率)；Solver: Judge评分；Judge: 格式合规 | 三方共同进化，无需外部标注，+4.54% avg |
| **[RL Tango](https://arxiv.org/abs/2505.15034)** (NeurIPS 2025) | Generator + Verifier（交替训练） | Generator: outcome accuracy；Verifier: balanced accuracy on (correct, incorrect) pairs | Verifier 只需 outcome reward 即可涌现 step-level 判断。[代码](https://github.com/kaiwenzha/rl-tango) |
| **[SPIRAL](https://arxiv.org/abs/2506.24119)** | Multi-agent self-play on zero-sum games | Win/loss reward from game environment | 用棋类等零和博弈的自然验证环境做 multi-turn RL |
| **[PasoDoble](https://arxiv.org/abs/2511.11881)** | Proposer + Solver（GAN-like） | Proposer: 生成 Solver 做不出的难题；Solver: 正确率 | 数学推理的 GAN，对抗驱动难度递增 |

### Adversarial / Game-theoretic（对抗博弈）

| 系统 | 架构 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[Prover-Verifier Games](https://arxiv.org/abs/2407.13692)** (OpenAI 2024) | Prover（helpful/sneaky 两模式）+ Verifier | Prover-helpful: correct + verifier accepts；Prover-sneaky: incorrect + verifier accepts；Verifier: 区分 helpful vs sneaky | 迭代博弈提升 legibility，verifier 不会被骗 |
| **[RLAC](https://arxiv.org/abs/2511.01758)** | Generator + Adversarial Critic + External Validator | Critic: 找到 generator 的 failure mode 得分；Generator: 通过 external validator 验证 | Critic 主动攻击 + 外部锚定，防止 critic 瞎编 |
| **[Self-Questioning](https://arxiv.org/abs/2508.03682)** | Asymmetric self-play（问题生成 + 解答） | Asymmetric reward for questioner vs solver | 非对称自博弈做推理提升 |

### Collaborative（协作）

| 系统 | 架构 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[MAPoRL](https://arxiv.org/abs/2502.18439)** (ACL 2025) | 多个 LLM agent 讨论 → Verifier 评分 | Verifier 评估最终答案 + 讨论质量；奖励 corrective/persuasive 互动 | 博弈论建模 agent 间协作，joint training >> 单独 finetune |
| **[DPSDP](https://arxiv.org/abs/2506.08379)** | Actor + Critic（多轮 refine） | Actor: 答案正确性；Critic: 反馈质量（能否帮 Actor 改对）；联合 KL-regularized DP 优化 | 5 轮 refine，MATH +5%，OOD 仍有效 |
| **[SWEET-RL](https://arxiv.org/abs/2503.15478)** (Tian Ye & Levine) | Multi-turn collaborative agents | Per-step advantage via Bradley-Terry preference model；asymmetric actor-critic（利用 training-time hidden info） | 解决多轮协作的 credit assignment |
| **[FlowReasoner](https://arxiv.org/abs/2504.15257)** | Query-level meta-agent 动态生成 multi-agent 系统 | Multi-purpose reward: performance + complexity + efficiency | GRPO 训练，每个 query 动态组 agent 系统。[代码](https://github.com/sail-sg/FlowReasoner) |

### Tool-use RL（工具使用强化学习）

| 系统 | 架构 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[ToolRL](https://arxiv.org/abs/2504.13958)** | LLM + multi-tool chain | Format reward (binary) + Correctness reward (tool name/param name/param content Jaccard)，总分[-3,4] | 分层 reward 比 binary 好 17%，no KL penalty 鼓励探索。[代码](https://github.com/qiancheng0/ToolRL) |
| **[Tool-Star](https://arxiv.org/abs/2505.16410)** | LLM + multi-tool (search+python+...) | Accuracy + format penalty(-1) + multi-tool bonus(+0.1)；Cold-start SFT → Self-Critic RL (GRPO+DPO) | 鼓励组合使用多种工具，difficulty-aware curriculum。[代码](https://github.com/dongguanting/Tool-Star) |
| **[Search-R1](https://arxiv.org/abs/2503.09516)** | LLM + search tool chain | Outcome accuracy；**retrieved token masking**（tool 返回不计入 policy loss） | Tool-use RL 的 canonical reference，masking 防梯度噪声 |
| **[WebThinker](https://arxiv.org/abs/2504.21776)** | LLM + web research tools | Iterative DPO on tool utilization trajectories | 用 DPO 替代 RL 做 tool-use 优化 |
| **[WebRL](https://arxiv.org/abs/2411.02337)** | LLM web agent + ORM | Outcome-Supervised Reward Model (binary 0/1) + self-evolving curriculum + KL-constrained updates | 8B 模型超 GPT-4-Turbo，curriculum 自动生成。[代码](https://github.com/THUDM/WebRL) |

### Credit Assignment（信用分配）

| 系统 | 粒度 | 方法 | 亮点 |
|------|------|------|------|
| **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)** (OpenAI) | Step-level | Process Reward Model (PRM)，人工标注每步对/错 | PRM 显著优于 ORM，但标注成本高 |
| **[Math-Shepherd](https://arxiv.org/abs/2312.08935)** | Step-level | 自动 step-level supervision：从每步 MC rollout 估计 value | 无需人工标注的 PRM 训练 |
| **[KTAE](https://arxiv.org/abs/2505.16826)** | **Token-level** | Fisher exact test + Information Gain 统计 token 与正确性的关联 → token-level advantage | Model-free，不需额外模型，比 GRPO/DAPO 更细粒度。[代码](https://github.com/ZNLP/KTAE) |
| **[Step-GRPO](https://arxiv.org/abs/2503.12937)** | Step-level | Rule-based reasoning rewards，per-step credit | 透明的 step-wise reward 分配 |
| **[R³](https://arxiv.org/abs/2402.05808)** | Step-level | Reverse curriculum：从最后一步往前逐步扩展训练 | 把 sparse reward 转化为 step-level signal |
| **[SWEET-RL](https://arxiv.org/abs/2503.15478)** | Step-level | Bradley-Terry preference + asymmetric actor-critic | 利用 training-time hidden info 做更好的 advantage 估计 |

### Long-horizon Agent RL（长链路智能体）

| 系统 | 特点 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[RAGEN](https://arxiv.org/abs/2504.20073)** (StarPO) | Multi-turn agent-env interaction | Trajectory-level；发现需要 reasoning-aware reward | "Echo Trap"：reward variance cliff → policy 坍缩。StarPO-S 用 trajectory filtering + decoupled clipping 解决。[代码](https://github.com/RAGEN-AI/RAGEN) |
| **[ASearcher](https://arxiv.org/abs/2508.07976)** | 40+ tool calls，150k+ tokens per trajectory | Sparse reward + LLM-as-Judge；fully async RL training | 突破 10-turn 限制，async 架构近 100% GPU 利用率。[代码](https://github.com/inclusionAI/ASearcher) |
| **[AgentRL](https://arxiv.org/abs/2510.04206)** (清华) | 多任务异步 RL | **Task advantage normalization**：per-task normalize reward | 5.8× throughput，防止某类任务主导梯度 |
| **[ARPO](https://arxiv.org/abs/2505.16282)** | GUI agent，长 horizon sparse reward | GRPO + experience replay buffer + task selection | Replay buffer 复用成功 trajectory，OSWorld 29.9% SOTA。[代码](https://github.com/dvlab-research/ARPO) |
| **[WebSailor](https://arxiv.org/abs/2507.02592)** | Web search agent | High-uncertainty task generation + efficient RL | 自动生成高不确定性任务做 curriculum |

### Unsupervised / Self-Rewarding（无监督奖励）

| 系统 | 方法 | 亮点 |
|------|------|------|
| **[TTRL](https://arxiv.org/abs/2504.16084)** | Majority voting 作为 test-time reward | 无需任何标注，多次采样投票做 self-reward |
| **[EMPO](https://arxiv.org/abs/2504.05812)** | Entropy minimization on unlabeled questions | 最小化预测熵作为 reward signal |
| **[Intuitor](https://arxiv.org/abs/2505.19590)** | Self-certainty as intrinsic reward | 模型自身确定性作为内在奖励 |
| **[Spurious Rewards](https://arxiv.org/abs/2506.10947)** | 研究 random reward 的效果 | 发现随机 reward 也能有部分效果（entropy regularization） |

### Reward Model / Process Supervision（奖励模型）

| 系统 | 方法 | 亮点 |
|------|------|------|
| **[GenRM (DeepMind)](https://arxiv.org/abs/2408.15240)** | Next-token prediction 做 verification | Generative verifier，CoT + majority vote |
| **[GenRM (SynthLabs)](https://arxiv.org/abs/2410.12832)** | Self-generated reasoning traces 做 preference labels | 自生成训练数据训 reward model |
| **[CRM](https://arxiv.org/abs/2511.16202)** | 多个 specialized evaluator + aggregator | 每个维度独立评估，模块化、可解释 |
| **[Two Minds Better Than One](https://arxiv.org/abs/2505.10597)** | 两个 RM 互相 peer-review | RewardBench +9.94 under noise |
| **[WorldPM](https://arxiv.org/abs/2505.10527)** | 15M forum data 预训练 reward model | Scaling laws for preference modeling |

### 🆕 2026 新进展：Multi-Agent 训练稳定性

| 系统 | 架构 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[Dr. MAS](https://arxiv.org/abs/2602.08847)** | 多 agent GRPO | **Per-agent advantage normalization**：每个 agent 用自己的 (μ_k, σ_k) normalize，而非全局 baseline | 解决异构 agent reward 分布不一致导致的梯度爆炸，+15.2% |
| **[Graph-GRPO](https://arxiv.org/abs/2603.02701)** | Multi-agent 通信拓扑优化 | 采样多个 DAG 通信图 → group-relative 比较 → edge-specific advantage | 用 GRPO 优化 agent 间**通信拓扑结构**本身 |
| **[M-GRPO](https://arxiv.org/abs/2511.13288)** | 垂直 multi-agent（主+子 agent） | 分层 GRPO：主 agent 和 sub-agent 解耦训练 | 解决 sub-agent 调用频率不一致 + 跨服务器训练，GAIA/XBench SOTA |

### 🆕 2026 新进展：Per-Action / Step-Level Reward

| 系统 | 方法 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[MAPPA](https://arxiv.org/abs/2601.23228)** | LLM-as-a-Coach | 每个 agent 的每个 action 由 AI coach 实时打分(0-10)，dense per-action reward | 精准 fault attribution，Data analysis +12.5%, Math +17.5% |
| **[Agent-RRM / Reagent](https://arxiv.org/abs/2601.22154)** | 三层结构化反馈 | **Reasoning Trace + Critique + Score**（不只给分，还给原因和改进建议） | GAIA 43.7%，减少 30-50% 训练步数。[代码](https://github.com/kxfan2002/Reagent) |
| **[FineStep](https://arxiv.org/abs/2605.04719)** | Step-level tool credit | R_exec(执行正确性) + R_proc(过程质量)，discounted backprop 到每步 | Tool-integrated tasks 专用 step-level credit，Text-to-SQL +3.25% |
| **[RetroAgent](https://arxiv.org/abs/2603.08561)** (上海 AI Lab) | Dual intrinsic feedback | (1) 数值型 capability-evolution reward（超越历史最佳才奖励）；(2) 语言型 reflection memory（NL lesson + SimUtil-UCB retrieval） | ALFWorld +18.3%, WebShop +15.4%。双反馈防止 local optima |
| **[ReTool](https://arxiv.org/abs/2504.11536)** (ByteDance) | Strategic tool use via RL | Outcome-driven reward + multi-turn code execution feedback loop | Cold-start SFT → PPO，AIME 72.5%，涌现 code self-correction。[代码](https://github.com/ReTool-RL/ReTool) |

### 🆕 2026 新进展：Orchestration-Level RL（编排层）

| 系统 | 方法 | Reward 设计 | 亮点 |
|------|------|------------|------|
| **[FlowSteer](https://arxiv.org/abs/2602.01664)** | End-to-end workflow orchestration RL | CWRPO 算法：diversity-constrained reward + conditional release（抑制 shortcut） | Policy model 分析执行状态 → 选 editing action → 迭代 refine workflow |
| **[RL via Orchestration Traces](https://www.chatpaper.ai/zh/dashboard/paper/787aa80e-781f-496a-b2cd-dd6fddc18159)** (Zhang et al. 2026) | Temporal interaction graph | Reward 从 token 到 team-level 多粒度；parallelism speedup + split correctness + aggregation quality | 把 multi-agent 协作形式化为时间戳事件图（spawn/delegate/communicate/terminate） |
| **[MARS](https://arxiv.org/abs/2510.04935)** | Dual-system co-evolution | Shared trajectory reward for System 1 (fast) + System 2 (slow) | Co-evolving 快慢思考系统 |

### 🆕 2026 新进展：Failure Diagnosis & Feedback

| 系统 | 方法 | 亮点 |
|------|------|------|
| **[GraphTracer](https://arxiv.org/abs/2510.10581)** | Information Dependency Graph (IDG) | 不按时间序列，按**信息依赖关系**追溯 root cause。+18.18% attribution accuracy |
| **[LEAFE](https://arxiv.org/abs/2603.16843)** | Reflective exploration | Agent 总结 env feedback → 回溯错误决策点 → 探索替代 action → SFT 蒸馏。+14% Pass@128 vs GRPO |
| **[MAFBench](https://arxiv.org/abs/2602.03128)** | Multi-agent framework benchmark | 揭示框架间 100x latency 差异和 30% accuracy drops |

---

## 2026 vs 2025：演进趋势

| 2025 的做法 | 2026 的进化 | 代表 |
|------------|------------|------|
| Global advantage normalization | **Per-agent** advantage normalization | Dr. MAS |
| Trajectory-level reward | **Per-action** dense reward from AI coach | MAPPA |
| Binary outcome reward | 三层结构化反馈（trace + critique + score） | Agent-RRM |
| 固定 multi-agent 架构 | RL 优化 **agent 通信拓扑** | Graph-GRPO |
| 单层 GRPO | 垂直 hierarchical M-GRPO（主/子 agent 分层训练） | M-GRPO |
| 结果导向 reward | Capability-evolution reward（超越历史才奖励）+ linguistic reflection | RetroAgent |
| Agent 内部优化 | **Workflow 编排层** end-to-end RL | FlowSteer |
| Temporal credit assignment | **Information dependency graph** 做 causal attribution | GraphTracer |
| Tool-use 需要显式 reward shaping | Outcome-driven RL 涌现 tool self-correction | ReTool |

---

## 跨系统共性 Pattern

```
┌─────────────────────────────────────────────────────────┐
│  Pattern 1: 角色分离 + 各自独立 reward                    │
│  (MAE, RL Tango, PVG, DPSDP)                            │
│  → 每个角色有明确、可独立计算的 reward signal              │
│  → 防止一个角色的 reward 依赖另一个角色的 live output      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 2: 外部锚定 (ground truth / env executor)       │
│  (AbsoluteZero, RLAC, RL Tango, ToolRL)                 │
│  → 至少一个 reward 信号来自不可被 game 的外部 oracle       │
│  → 防止多 agent 共谋                                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 3: Difficulty curriculum 自动涌现               │
│  (AbsoluteZero, R-Zero, MAE, PasoDoble, WebRL)          │
│  → Proposer 的 reward 包含 "Solver 做不出" 这一项         │
│  → 随着 Solver 变强，Proposer 自动生成更难的题            │
│  → R-Zero: uncertainty reward 让难度锚定在 50% 正确率     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 4: Task advantage normalization                 │
│  (AgentRL, GRPO, KTAE)                                  │
│  → 多任务/多维度时，per-task/per-dim normalize            │
│  → 防止某一类 reward 信号 dominate gradient               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 5: Experience replay for sparse reward          │
│  (ARPO, WebRL, R-Zero)                                  │
│  → 长 horizon + sparse reward 时复用成功 trajectory       │
│  → 配合 task selection 聚焦 informative interactions     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 6: Token/Step masking                           │
│  (Search-R1, ToolRL)                                    │
│  → Tool 返回的 observation token 不计入 policy loss       │
│  → 只对 agent 自己生成的 token 计算梯度                   │
│  → 防止 tool output 的噪声污染 policy gradient           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 7 [2026]: Per-agent normalization               │
│  (Dr. MAS, M-GRPO)                                     │
│  → 异构 agent reward 分布不同，各自 normalize             │
│  → 替代全局 baseline，防止某 agent reward scale 主导      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 8 [2026]: Structured feedback > scalar reward   │
│  (Agent-RRM, RetroAgent, LEAFE)                         │
│  → Reward 不只是数字，还包含 reasoning trace + critique   │
│  → 语言反馈可作为 intrinsic reward 或 context 注入        │
│  → 减少 30-50% 训练步数（Agent-RRM 实测）                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pattern 9 [2026]: Causal (not temporal) attribution     │
│  (GraphTracer, FineStep)                                │
│  → 按信息依赖关系追溯 root cause，而非按时间顺序          │
│  → 多步 tool-use 中第 2 步的错可能在第 18 步才暴露        │
│  → IDG (Information Dependency Graph) 做精准归因          │
└─────────────────────────────────────────────────────────┘
```

---

## 我们的方案设计

### Verifier Reward：锚定 calibration

```python
def reward_verifier(judgement, ground_truth):
    # 核心：balanced accuracy，对的说对，错的说错
    correct = (judgement.accept == ground_truth.is_correct)
    r_calibration = +1.0 if correct else -1.0

    # 可选：rationale 质量（防 lazy judge）
    r_rationale = rationale_specificity(judgement.rationale, generator_output)

    return r_calibration + 0.1 * r_rationale
```

**关键**：训练 batch 里正负样本必须平衡（既有 generator 做对的，也有做错的），否则 verifier 学到偏的 prior。

### Generator Reward：分层组合

```python
def reward_generator(trajectory, verifier_judgement, ground_truth=None):
    # Layer 1: Outcome（最终结果）
    if ground_truth:
        r_outcome = exact_match(trajectory.final_answer, ground_truth)
    else:
        r_outcome = ema_verifier(trajectory)  # EMA verifier，不是 live

    # Layer 2: Tool-use process（参考 ToolRL 的分层设计）
    r_process = sum(
        gamma**i * (
            tool_call_valid(step)             # format reward (binary)
            + tool_name_match(step)           # tool selection correctness
            + param_match(step)               # parameter correctness
            + observation_used(step, later_steps)  # observation 被后续利用
            - redundant_call(step, earlier_steps)  # 重复调用惩罚
        )
        for i, step in enumerate(trajectory.steps)
    )

    # Layer 3: Efficiency（步数惩罚）
    r_efficiency = -len(trajectory.steps) / max_steps

    # Layer 4: Multi-tool bonus（参考 Tool-Star）
    r_diversity = 0.1 if len(unique_tools(trajectory)) > 1 else 0.0

    return 1.0 * r_outcome + 0.05 * r_process + 0.02 * r_efficiency + r_diversity
```

### 对抗信号处理

```python
# ❌ 不稳定：generator exploit verifier 的 bug
r_gen = live_verifier(gen_output)

# ✅ EMA verifier 提供稳定评估面（参考 PVG）
r_gen = ema_verifier(gen_output)  # θ_ema = β·θ_ema + (1-β)·θ_V

# ✅ 混合信号（有 ground truth 时）
r_gen = 0.7 * ground_truth_reward + 0.3 * ema_verifier_reward
```

---

## Tool-use Credit Assignment 策略

| 策略 | 复杂度 | 适用场景 | 参考 |
|------|--------|---------|------|
| **A: PRM** | 高（需训练额外模型或 MC rollout） | 有大量标注数据或计算预算充足 | Let's Verify Step by Step, Math-Shepherd |
| **B: KTAE (token-level)** | 中（统计方法，不需额外模型） | 想要比 step 更细的粒度 | KTAE |
| **C: Hindsight shaping** | 中（rule-based 回溯） | 中等长度 chain，有明确的 "错误步" 信号 | R³ |
| **D: Trajectory-level GRPO** | 低（不做 per-step 分解） | 起步阶段，chain 不太长 | DeepSeek-R1, GRPO |

**建议路径**：D → C → B → A（从简单开始，variance 大了再加复杂度）

### Strategy D: GRPO (推荐起步)

同一问题采样 N 条 trajectory，group-relative advantage：

```python
rewards = [R(traj) for traj in trajectories]
baseline = mean(rewards)  # 或 leave-one-out
advantages = [r - baseline for r in rewards]
# 注意：tool observation tokens 要 mask 掉不计入 loss (Search-R1)
```

### Strategy C: Hindsight (第二步)

```python
def hindsight_step_reward(trajectory, step_i, final_correct):
    if not final_correct:
        if is_obviously_wrong(step_i):
            return -1.0  # 明显错误步重罚
        return -0.1      # 均匀轻罚
    else:
        return +1.0 / len(trajectory.steps)  # 均匀正奖励
```

### Strategy B: KTAE (第三步，token-level)

```python
# 统计每个 token 在 correct vs incorrect rollouts 中的出现频率
# Fisher exact test → p-value → association score
# 调整 rollout-level advantage → token-level advantage
# 无需额外模型，纯统计方法
```

---

## 训练节奏

```
Phase 1: Warmup（分别预训练）
├── Generator: SFT on tool-use demonstrations（参考 Tool-Star cold-start）
├── Verifier: SFT on (solution, judgement) pairs
└── 目的：双方有基础能力再开始对抗

Phase 2: Alternating RL（参考 RL Tango）
├── Fix Verifier → train Generator (N steps)
│   └── 用 GRPO + token masking (Search-R1)
├── Fix Generator → train Verifier (M steps)
│   └── Verifier 用 replay buffer（不只看当前 generator 输出，参考 ARPO）
├── 比例 N:M ≈ 3:1
└── 监控 Echo Trap（参考 RAGEN）：reward variance cliff → 回退或加 filtering

Phase 3: Joint（可选，风险高）
├── 同时更新，generator 用 EMA verifier 信号
└── 监控 verifier accuracy，< 55% 则回退到 Phase 2
```

### 稳定性技巧（来自各论文）

| 技巧 | 来源 | 作用 |
|------|------|------|
| Trajectory filtering（保留 top-25% variance） | RAGEN/StarPO-S | 防 Echo Trap |
| Experience replay buffer | ARPO | Sparse reward 下提升 sample efficiency |
| Task advantage normalization | AgentRL | 多任务时防止 reward domination |
| Token masking (tool output) | Search-R1 | 防梯度噪声 |
| Decoupled clipping (ε_low≠ε_high) | StarPO-S | 平衡 exploration 和 stability |
| No KL penalty | ToolRL | 鼓励 tool-use 空间的探索 |
| Self-evolving curriculum | WebRL, R-Zero | 自动生成适当难度的训练任务 |
| Per-agent advantage normalization | Dr. MAS [2026] | 异构 agent reward 分布不同时稳定训练 |
| Structured feedback (trace+critique+score) | Agent-RRM [2026] | 比 scalar reward 收敛快 30-50% |
| Capability-evolution reward | RetroAgent [2026] | 只有超越历史最佳才奖励，防 local optima |
| Diversity-constrained reward | FlowSteer [2026] | 抑制 workflow shortcut behavior |
| IDG-based fault attribution | GraphTracer [2026] | 精准定位 multi-step 错误根因 |

---

## 稳定性监控

| 指标 | 健康范围 | 报警信号 |
|------|---------|---------|
| Verifier balanced accuracy | > 65% | < 55% = 退化 |
| Generator win rate vs EMA verifier | 40-70% | > 85% = exploit |
| Tool call 平均步数 | 缓慢下降 | 突然 = 1 → 退化到不用工具 |
| Verifier accept rate | 30-70% | < 10% 或 > 90% = trivial |
| Ground truth accuracy | 上升 | 停滞 + accept 上升 = 共谋 |
| **Reward variance** | 稳定或缓降 | **骤降 = Echo Trap**（RAGEN） |
| Unique tools per trajectory | > 1 | = 1 → 退化到单工具 |

---

## 精读推荐（按优先级）

### Tier 1：直接对应场景

1. **[RL Tango](https://arxiv.org/abs/2505.15034)** — Generator-Verifier 交替 RL，outcome reward 涌现 step-level 判断
2. **[ToolRL](https://arxiv.org/abs/2504.13958)** — Tool-use reward 分层设计（format + correctness），GRPO 训练
3. **[Search-R1](https://arxiv.org/abs/2503.09516)** — Tool-use chain RL：token masking + outcome reward
4. **[Agent-RRM / Reagent](https://arxiv.org/abs/2601.22154)** 🆕 — 三层结构化反馈替代 scalar reward，训练效率大幅提升
5. **[SWiRL](https://arxiv.org/abs/2504.04736)** 🆕 (DeepMind) — Step-Wise RL + process judge，跨任务泛化最强

### Tier 2：关键组件

5. **[Absolute Zero](https://arxiv.org/abs/2505.03335)** — 零数据 self-play，learnability reward
6. **[RAGEN](https://arxiv.org/abs/2504.20073)** — Long-horizon agent RL 的 Echo Trap 问题 + StarPO-S 解法
7. **[KTAE](https://arxiv.org/abs/2505.16826)** — Token-level credit assignment，model-free 统计方法
8. **[SWEET-RL](https://arxiv.org/abs/2503.15478)** — 多轮协作 credit assignment (Bradley-Terry + asymmetric AC)
9. **[Dr. MAS](https://arxiv.org/abs/2602.08847)** 🆕 — Per-agent normalization，multi-agent GRPO 稳定训练
10. **[MAPPA](https://arxiv.org/abs/2601.23228)** 🆕 — Per-action dense reward from LLM-as-Coach
11. **[RetroAgent](https://arxiv.org/abs/2603.08561)** 🆕 — Dual feedback (数值+语言) 防 local optima
12. **[ARTIST](https://arxiv.org/abs/2505.01441)** 🆕 — Agentic Reasoning + Tool Integration，deliberative tool calls
13. **[Agent-R1](https://arxiv.org/abs/2511.14460)** 🆕 — End-to-end multi-turn RL + action masking，3× baseline
14. **[SHARP](https://arxiv.org/abs/2602.08335)** 🆕 — Shapley value credit assignment for MAS

### Tier 3：扩展视野

12. **[Prover-Verifier Games](https://arxiv.org/abs/2407.13692)** — 对抗训练稳定性（helpful/sneaky 双模式）
13. **[R-Zero](https://arxiv.org/abs/2508.05004)** — Challenger-Solver 共进化，uncertainty-driven difficulty
14. **[ASearcher](https://arxiv.org/abs/2508.07976)** — 40+ turn 长链路 async RL
15. **[Tool-Star](https://arxiv.org/abs/2505.16410)** — Multi-tool 组合 reward + difficulty curriculum
16. **[ARPO](https://arxiv.org/abs/2505.16282)** — Experience replay for sparse-reward agent RL
17. **[ReTool](https://arxiv.org/abs/2504.11536)** — Strategic tool use RL，AIME 72.5%，涌现 self-correction
18. **[FlowSteer](https://arxiv.org/abs/2602.01664)** 🆕 — Workflow 编排层 RL (CWRPO)
19. **[GraphTracer](https://arxiv.org/abs/2510.10581)** 🆕 — IDG failure tracing for multi-agent
20. **[LEAFE](https://arxiv.org/abs/2603.16843)** 🆕 — Reflective exploration + SFT 蒸馏

---

## 设计决策备忘

- **有 ground truth oracle → verifier reward 简单（对/错 ±1），generator 用混合信号**
- **无 ground truth → 必须防止 generator-verifier 共谋，需 consensus / 多 verifier 投票**
- **RL Tango 的 insight：verifier 不需要 process-level 标注，outcome reward 就够**
- **Tool masking (Search-R1)：tool 返回的 observation token 不计入 policy loss，避免梯度噪声**
- **AbsoluteZero 的 insight：learnability reward（中等难度 > 太难/太简单）自动生成 curriculum**
- **AgentRL 的 insight：多任务时 per-task normalize reward，防止某类任务主导梯度**
- **RAGEN 的 warning："Echo Trap" — reward variance 突然消失意味着 policy 坍缩到 shallow strategy**
- **ToolRL 的 insight：fine-grained tool reward (name+param+content) 比 binary 好 17%**
- **KTAE 的 insight：token-level credit assignment 可以纯统计做，不需要额外模型**
- **R-Zero 的 insight：pseudo-label 会 noise accumulate（79%→63%），长期需要外部锚定**
- **ARPO 的 insight：sparse reward 下 replay buffer 极其重要，OSWorld +3.9% over no-replay**

### 🆕 2026 新 Insights

- **Dr. MAS：per-agent normalize 比 global normalize 在异构 multi-agent 中好 15%+**
- **Agent-RRM：structured feedback (trace+critique+score) 比 scalar reward 减少 30-50% 训练步数**
- **RetroAgent：capability-evolution reward（超越历史才奖励）+ linguistic reflection 双管齐下防 local optima**
- **MAPPA：LLM-as-Coach 给每个 action 实时打分(0-10)，解决 dense reward 的人力成本问题**
- **FlowSteer：workflow 编排层也可以 end-to-end RL，不只是 agent 内部策略**
- **GraphTracer：multi-step 失败归因应按 information dependency 而非时间顺序**
- **ReTool：outcome-only reward 可以涌现 tool self-correction（code 写错 → 自动改），不需要显式 reward shaping**
- **LEAFE：reflective exploration（回溯+替代探索+蒸馏）比 GRPO 在 interactive 任务上好 14%**
- **FineStep：tool-integrated credit assignment 需要 R_exec + R_proc 两个维度，单独 outcome 不够**
- **SHARP：Shapley value 做 multi-agent credit assignment，理论上最公平但计算开销大（需近似）**
- **LIFE 综述：现有 step-level failure attribution 只有 14.2% 准确率——这是未解决的核心难题**
- **"End of Reward Engineering"：LLM 自动从 NL spec 合成 reward function，是 2026 的方向性趋势**
- **MARL for LLM Teams：8 种 reward 类型可组合使用，不必只选一种**
- **ArenaRL：没有 ground truth 时用 tournament ranking 做相对评价，比绝对打分更稳定**
- **MatchTIR：bipartite matching 精准对齐 "哪个 tool call 贡献了正确性"，比均匀 credit 好**
- **Search-R2：Actor-Refiner 分工比单 agent self-refine 效果好（专门的 refiner 学修正策略）**
