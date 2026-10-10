# BindPack

在 HippoRAG2 图检索基础上，执行带来源绑定的依赖子目标，并在有限检索前缀中选择
祖先闭合证据包。已有支撑保护和独立语义否决用于控制末端交换的风险。

当前方法对应此前 PathCondRAG 的已验证 `legacy` 组合：stage 4 的依赖绑定检索，
加上 `planning,plan_prune,dag_package,support_semantic_veto`。
`dependency` / `dependency_joint` 属于后续评分尝试，不是此项目的默认主方法。

## 阅读入口

- [实验说明](docs/实验说明.md)：当前算法边界、论文贡献与实验顺序。
- [源码导航](src/bindpack/README.md)：各目录职责和兼容入口。
- [证据模块导航](src/bindpack/evidence/README.md)：主线与历史实验的关系。
- [索引构建说明](src/bindpack/index/README.md)：NER/OpenIE、校验、重试和失败策略。
- [迁移与重构核验](docs/代码迁移与结构重构核验_2026-10-11.md)：实际验证结果及限制。

## 代码结构

```text
src/bindpack/
├── config.py
├── index/               # 抽取、图构建、检查点与索引发布
│   ├── ner/
│   ├── openie/
│   ├── graph_index.py
│   └── artifacts.py
├── retrieval/           # 查询编排、图检索、排序与路径选择
├── evidence/
│   ├── planning/        # 依赖计划、绑定与关系工具
│   ├── selection/       # 证据包、覆盖与集合选择
│   ├── verification/    # 支撑保护与语义核验
│   └── experimental/    # 保留的历史尝试和消融
├── core/                # 公共运行时、基础检索和 QA 原语
├── storage/             # 持久化向量与文本
├── embedding_model/
├── llm/
├── prompts/
├── evaluation/
└── utils/
```

根目录少数旧模块只重导出公开类，实际实现各保留一份。

## 环境与运行入口

算法代码使用 `rag` 环境；本地 vLLM 服务使用独立的 `vllm` 环境。
安装本项目或将 `src` 加入 `PYTHONPATH` 后可以使用：

```python
from bindpack import BindPack, BindPackConfig
```

裸 `BindPackConfig()` 保留 stage 0 默认配置。完整方法由正式脚本传入 stage、
依赖规划、PCQD、迭代检索和核验参数；不要把包的默认配置当成已完成实验配置。

现有正式入口：

| 脚本 | 用途 |
| --- | --- |
| `scripts/build_shared_index.py` | 使用本项目抽取器构建共享索引 |
| `scripts/eval_dataset.py` | 单个数据集的检索 / QA 评估 |
| `scripts/run_multi_dataset_retrieval_nv2.sh` | NV2 三个索引与五种方法的既有流程 |
| `scripts/run_multi_dataset_retrieval_plan_prune_dag_package.sh` | 复用共享索引运行 DAG 包版本 |
| `scripts/run_multi_dataset_retrieval_support_semantic_veto.sh` | 复用共享索引运行当前完整方法 |

可先查看 Python 入口的 `--help`；shell 入口会创建日志并执行工作流。
输出根目录通过 `OUT_ROOT` 或 `--out-root` 指定。既有结果目录保留历史 case 名称。
外部 HippoRAG2 默认位于 `/root/baseline/HippoRAG`，相应 runner 支持指定其位置。

共享索引应冻结语料、模型、抽取配置与身份信息，同一数据集的各方法复用同一个索引。
`openie_strict=false` 允许最终抽取失败的 chunk 保留文本与向量、排除未通过检查的关系，
继续发布索引；它不能绕过语料身份冲突等其他错误。详见索引说明。

LLM 并发、embedding batch 和 hop 来源以每次实验保存的配置为准。
当前正式三数据集流程使用 benchmark hop 条件；论文应披露这一结构信息条件，
并另做 estimated 对照。MuSiQue 逐题标签和 HotpotQA / 2Wiki 的数据集级 prior 有区别。

## CPU 检查

```bash
PYTHONPATH="$PWD/src" /root/anaconda3/envs/rag/bin/python -B -m unittest discover -s tests -v
```

这些检查使用 mock 与小型数据验证算法、导入和脚本边界，不启动模型或后台任务。
本地完整查询集上的多轮探索结果不等于未参与调参的官方完整测试集。
