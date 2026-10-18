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

- **相同数据。** 固定三个数据集的问题 ID、顺序、语料与划分。本地 1000 题子集应写明规模，不能表述为上游完整官方 split。不得把 `*_v2.json` 或旧实验的不同样本直接并入同一主表。
- **相同 reader。** 主比较使用 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，对齐最终回答阶段的提示、证据预算、生成与思考模式。保留各方法必要的检索、规划和摘要提示，并计入这些额外调用的成本。若另报各方法原生生成流程的 QA，需说明差异。
- **明确 embedding。** BindPack 默认使用 [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)。结构方法在兼容且经过适配验证时统一为该 embedding，记录与原论文的差异；保留各方法必需的专用模块。BM25 不使用 embedding，Contriever/BGE-M3 的编码器定义了各自基线，不替换成 Qwen。原配置复现结果与统一配置结果应分开标识。
- **相同输入，匹配索引。** 各方法可以按自身设计建立图、树或向量索引，建库成本一并报告。当前冻结的 HippoRAG2 图索引只在兼容实现之间共享，不能直接充当 RAPTOR、PropRAG、PathRAG 等全部方法的原生索引；换 embedding 必须重建匹配的向量资产。
- **相同指标定义。** 统一输出 QA EM/F1；有可验证的原始 passage 排名时计算 Recall@K、FCR@K，有可信 gold 依赖图及证据映射时再计算 DCC@K。摘要或生成内容若不能追溯原始 passage，不强行当作命中；缺失条件写 `null` 和原因。具体定义见[指标说明](指标说明.md)。
- **可追溯复现。** 结果至少记录方法代码 commit、模型版本、有效配置、语料/索引摘要、预算和是否使用 gold hop 信息。论文原表数值与本机复现数值分开；不要只替换模型名后仍称完全复现原设置。

## 当前仓库中的执行范围

`scripts/run_bindpack_four_cases_reuse.sh` 复用三个已经核验的 Qwen 索引，默认运行四组 BindPack 和外部 HippoRAG2，并启用 reader QA。该入口没有实现上述所有 baseline 的统一调度。

当前这轮独立 HippoRAG2 的输出目录为：

```text
outputs/3multi_hop_datasets_results_10_5/cases/<dataset>/hippporag2_reproduce/
```

目录名中的三个 `p` 沿用现有脚本。它调用仓库外 `/root/baseline/HippoRAG/main.py` 的独立实现，不能仅凭本仓库就假定外部依赖已安装。历史 `hipporag2/`、`pathcondrag_original/` 和 `exp4_*` 目录保留原实验身份；同属 HippoRAG2 或旧完整方法，不代表样本、索引、配置和 reader 与本轮一致。论文主表以核验匹配协议后的结果为准。
