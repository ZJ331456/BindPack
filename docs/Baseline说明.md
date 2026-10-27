# Baseline 说明

本文件固定论文中的方法分组、比较目的与复现口径。下列清单是论文的对比范围；是否已有可用结果，以同一评测协议下的实际输出为准，不能把列入清单写成已经完成复现。文献信息核验日期：2026-10-18。实验安排见[实验图表](实验图表.md)，评测定义见[指标说明](指标说明.md)。

## Standard Retrieval

这一组直接对原始 passages 排序，用于判断结构建模相对词项匹配和稠密检索的收益。报告 QA 时，在检索结果后接相同的 reader；embedding 模型本身不等于完整的 RAG 方法。

| 方法 | 版本与出处 | 机制与比较目的 |
| --- | --- | --- |
| BM25 | 经典稀疏检索；[Robertson 与 Zaragoza 的 BM25 原作者综述](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) | 根据词频、逆文档频率和长度归一化打分，提供词项匹配基线；固定分词器、实现版本及 `k1/b`。 |
| Contriever | **TMLR 2022**；[正式论文](https://openreview.net/notes/edits/attachment?id=fZ9JdaTZYD&name=pdf)、[作者代码](https://github.com/facebookresearch/contriever) | 通过无监督对比学习训练双编码器。使用 `facebook/contriever`；若采用 MS MARCO 微调版，应单独标为 Contriever-MSMARCO，不能混用。 |
| BGE-M3 | [BAAI 官方模型卡](https://huggingface.co/BAAI/bge-m3)、[论文](https://arxiv.org/abs/2402.03216) | 固定 `BAAI/bge-m3`，主表采用 dense-only passage 检索，控制为标准稠密基线。若使用 sparse/multi-vector 或融合模式，另行标注，避免与 dense-only 结果混写。 |
| Qwen3-Emb | [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)、[论文](https://arxiv.org/abs/2506.05176) | 明确指 **Qwen3-Embedding-8B** 的直接 passage 向量检索，作为与 BindPack 使用相同 embedding 的强基线；保留查询 instruction、pooling、归一化和截断设置。 |

## Structure-Aware RAG

这里的结构包括推理过程、摘要树、语料图和查询依赖图。IRCoT 属于迭代推理检索；LogicRAG 构建查询结构；不能把这一组统一描述为“都预建语料知识图谱”。

| 方法 | 核验后的发表信息与来源 | 核心机制与比较目的 |
| --- | --- | --- |
| IRCoT | **ACL 2023**；[论文](https://aclanthology.org/2023.acl-long.557/)、[作者代码](https://github.com/StonyBrookNLP/ircot) | 交替生成推理步骤与检索，观察 BindPack 相对多轮检索推理的效果和在线成本。 |
| RAPTOR | **ICLR 2024**；[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)、[作者代码](https://github.com/parthsarthi03/raptor) | 递归聚类并摘要，形成多层检索树，比较层次摘要结构与依赖证据组织。 |
| LightRAG | **Findings of EMNLP 2025**；[论文](https://aclanthology.org/2025.findings-emnlp.568/)、[作者代码](https://github.com/HKUDS/LightRAG) | 联合图结构与向量表示进行双层检索，覆盖局部实体信息与较高层关系；是语料图检索对照。 |
| PropRAG | **EMNLP 2025**；[论文](https://aclanthology.org/2025.emnlp-main.317/) | 离线抽取带上下文的 propositions，在线通过无 LLM 的 beam search 发现 proposition paths，比较显式路径搜索与 BindPack 的 Route–Plan–Pack 配合。 |
| HippoRAG2 | **ICML 2025**；[正式论文](https://proceedings.mlr.press/v267/gutierrez25a.html)、[作者代码](https://github.com/OSU-NLP-Group/HippoRAG) | 结合实体与 passage 图、查询到事实的检索过滤及 Personalized PageRank，是本项目最直接的图检索基线。论文题名为 *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*。 |
| PathRAG | **AAAI 2026**；[正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/40268)、[作者代码](https://github.com/BUPT-GAMMA/PathRAG) | 使用流量剪枝选取关系路径并组织为路径提示，比较路径去冗余与证据包选择。 |
| LogicRAG | **AAAI 2026**；[正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/40278)、[作者代码](https://github.com/chensyCN/LogicRAG) | 将问题分解成逻辑依赖 DAG，按拓扑顺序执行检索并剪枝，是 Plan 的关键机制对照。正式题名为 *You Don’t Need Pre-Built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures*。 |
| CatRAG | **Findings of ACL 2026**；[论文](https://aclanthology.org/2026.findings-acl.290/)、[作者代码](https://github.com/kwunhang/CatRAG) | 在 HippoRAG2 上加入实体锚定、查询相关边权调整和关键事实 passage 增强，比较查询自适应图遍历与完整证据恢复。论文题名为 *Breaking the Static Graph: Context-Aware Traversal for Graph-Based RAG*。 |

LightRAG 和 CatRAG 均保留在指定的方法清单中；论文引用需写明 **Findings**，与会议主会论文准确区分。

## BindPack

BindPack 是本文方法：Route 发现并精炼候选路径，Plan 构建与执行依赖计划、绑定中间答案及证据来源，Pack 在预算内平衡覆盖、冗余与依赖闭合，形成最终证据包。

消融使用 `full`、`wo_route`、`wo_plan`、`wo_pack` 四种阶段模式。独立 HippoRAG2 是额外方法基线；`wo_route` 使用本项目的基础候选发现适配器，不能替代独立 HippoRAG2 复现。

## 公平比较与结果口径

- **相同数据。** 固定三个 QA 数据集及 HoVer 扩展实验的问题 ID、顺序、语料与划分；HoVer 单独采用分类评测。本地 1000 题子集应写明规模，不能表述为上游完整官方 split。不得把 `*_v2.json` 或旧实验的不同样本直接并入同一主表。
- **相同 reader。** 主比较使用 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，对齐最终回答阶段的提示、证据预算、生成与思考模式。保留各方法必要的检索、规划和摘要提示，并计入这些额外调用的成本。若另报各方法原生生成流程的 QA，需说明差异。
- **明确 embedding。** BindPack 默认使用 [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)。结构方法在兼容且经过适配验证时统一为该 embedding，记录与原论文的差异；保留各方法必需的专用模块。BM25 不使用 embedding，Contriever/BGE-M3 的编码器定义了各自基线，不替换成 Qwen。原配置复现结果与统一配置结果应分开标识。
- **相同输入，匹配索引。** 各方法可以按自身设计建立图、树或向量索引，建库成本一并报告。当前冻结的 HippoRAG2 图索引只在兼容实现之间共享，不能直接充当 RAPTOR、PropRAG、PathRAG 等全部方法的原生索引；换 embedding 必须重建匹配的向量资产。
- **相同指标定义。** 统一输出 QA EM/F1；有可验证的原始 passage 排名时计算 Recall@K、FCR@K，有可信 gold 依赖图及证据映射时再计算 DCC@K。摘要或生成内容若不能追溯原始 passage，不强行当作命中；缺失条件写 `null` 和原因。具体定义见[指标说明](指标说明.md)。
- **可追溯复现。** 结果至少记录方法代码 commit、模型版本、有效配置、语料/索引摘要、预算和是否使用 gold hop 信息。论文原表数值与本机复现数值分开；不要只替换模型名后仍称完全复现原设置。

## 当前仓库中的执行范围

Standard Retrieval 已添加独立入口 `/root/baseline/scripts/run_standard_retrieval.sh`，配置位于 `/root/baseline/configs/standard_retrieval.json`，读取 `/root/BindPack/datasets` 的 HotpotQA、2WikiMultiHopQA、MuSiQue、HoVer。完整参数、来源和操作见 [Standard Retrieval 运行说明](../../baseline/scripts/STANDARD_RETRIEVAL.md)。

Standard Retrieval 已在完整 corpus 上为每个方法/数据集挑 3 题完成建库、检索、reader、独立指标的真实预检；Top3/5/8/10 分别实际生成答案。预检结果目录为 `/root/baseline/outputs/standard_review_20261019`，不能把小样本分数当作主实验结论。

全部 reader 固定 `/root/models/Qwen3-8B`，Qwen dense 使用 `/root/models/Qwen3-Embedding-8B`；Contriever、BGE-M3 保留原生编码器，BM25 不使用 embedding。全部方法使用共同 canonical passages 和独立指标。MuSiQue 可评 DCC/NodeCov/ClosureGap；其他数据集缺可信 gold DAG 时保留 null，表格显示 N/A。

每组保存 `report.json` 和逐题诊断，并记录真实建库、索引/模型加载和查询时间、平均/P50/P95、LLM 调用与 tokens、实际 reader 证据篇数/tokens、裁剪及索引大小。无计费单价时货币费用为 N/A，不把共享服务 wall time 当作独占 GPU 计算时间。CSV 与预算曲线由 `/root/baseline/scripts/summarize_standard_retrieval.py` 从 validated 结果导出。

本轮实际预检已完成：64 个 case、192 次真实 reader 调用，零重试，token 全部可对账。审核入口见 [小样本结果](../../baseline/outputs/standard_review_20261019/REVIEW.md)，[效率成本](../../baseline/outputs/standard_review_20261019/efficiency_cost.csv) 与 [方法耗时](../../baseline/outputs/standard_review_20261019/method_time_cost.csv) 单独保存。新增 12 项单元测试及原项目 231 项回归测试通过；已验证服务关闭时可离线恢复全部完成组，原结果和计时不变。

全量采用 reader@5，四方法 × 四数据集 × 每集全部 1000 题，共 16 组、16000 次问题评测；完整排名保留至 200，计算各 K 的证据指标。入口为 `/root/baseline/scripts/run_standard_retrieval_full.sh`，会在全部评测结束后自动审计与导出指标、效率成本及方法/整批耗时表。新的结果目录为 `/root/baseline/scripts/outputs/standard_full_20261019`，索引、服务及评测日志同样位于 `/root/baseline/scripts/outputs` 下；原预检结果和索引保留。

本批全量已于 2026-10-19 完成并通过审计：16组、16000题、零请求重试，实际总耗时6小时24分15秒。[@5 HTML结果汇总](../../baseline/scripts/outputs/standard_full_20261019/results_at5.html) 采用分组三线表，提供小数/百分比切换、原值CSV导出和打印；HoVer单列原生分类指标，结构不可评项保留N/A。

八个 Structure-Aware RAG 使用独立环境`/root/baseline/.venvs/structure`与共同Qwen模型，公共入口为`/root/baseline/scripts/run_structure_retrieval_matrix.sh`。原生机制及统一配置差异见[运行说明](../../baseline/scripts/STRUCTURE_RETRIEVAL.md)。按最新指令，后续小样本仅测MuSiQue/HoVer，每方法每集3题，16组48题；已完成的其他测试保留。32篇开发夹具，只用于真实建库和指标验通，不能填论文主表；全量使用相同冻结的完整canonical corpus，每集1000题。

**RAPTOR已在2026-10-20修复passage评测口径。** [HippoRAG2论文](https://arxiv.org/pdf/2502.14802)表3和9确实有其Recall。新适配保留完整原文叶子并使用原生tree-traversal到叶子，输出原文排名与共同reader@5；不把摘要所有后代免费算作召回，也不宣称与作者未公开的适配细节完全一致。八方法均要求Recall/FCR@3/5/8/10/20/200，MuSiQue另要求gold DAG下的DCC/NodeCov/ClosureGap；其余数据集缺gold DAG的结构指标仍为N/A。

按用户授权，旧队列PID1212859已停止，vLLM保留。IRCoT四组4000题审计通过并原样保留；旧RAPTOR两组完成结果和MuSiQue部分预测/成本完整归档到`outputs/cancelled_runs/raptor_summary_units_20261020_130235`。新预检为[passage预检目录](../../baseline/scripts/outputs/structure_smoke_passages_20261020)，按最新指令要求MuSiQue/HoVer的16组通过后，在原全量目录补跑RAPTOR并续跑四数据集全量；源码、模型、配置、冻结数据与既有结果的SHA都需核验。实际通过/完成以该批`smoke_passed.ok`、`audit.json`、`completed.ok`为准。

LightRAG保留原生建库与hybrid检索，最终回答按用户确认只用召回文本；图关系不送入reader。LightRAG/PathRAG的长度恢复采用历史压缩、原文分段重抽取与完成标记核验，不增加上下文或输出上限。启动记录、日志、进度条和实际PID见[全量启动记录](../../baseline/scripts/outputs/structure_full_20261019_v2/RUN.md)。

`scripts/run_bindpack_four_cases_reuse.sh` 复用三个已经核验的 Qwen 索引，默认运行四组 BindPack 和外部 HippoRAG2，并启用 reader QA。该入口没有实现上述所有 baseline 的统一调度。

当前这轮独立 HippoRAG2 的输出目录为：

```text
outputs/3multi_hop_datasets_results_10_5/cases/<dataset>/hippporag2_reproduce/
```

目录名中的三个 `p` 沿用现有脚本。它调用仓库外 `/root/baseline/HippoRAG/main.py` 的独立实现，不能仅凭本仓库就假定外部依赖已安装。历史 `hipporag2/`、`pathcondrag_original/` 和 `exp4_*` 目录保留原实验身份；同属 HippoRAG2 或旧完整方法，不代表样本、索引、配置和 reader 与本轮一致。论文主表以核验匹配协议后的结果为准。

2026-10-20 13:51:13，本轮MuSiQue/HoVer的16组48题预检与独立审计均通过，四数据集全量已恢复，PID1534055；IRCoT原4000题跳过，当前RAPTOR重建HotpotQA完整索引。45项回归测试通过；既有额外两集测试与取消RAPTOR成本独立保留。实际状态见全量运行记录及status.json。
