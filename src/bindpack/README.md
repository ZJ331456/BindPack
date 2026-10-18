# BindPack 源码导航

公开入口只有 `from bindpack import BindPack, BindPackConfig`；用 `BindPack(config=cfg)` 创建检索器，运行配置为 `rag.config`。旧模块包装、类名别名、历史 flags 和实验 API 已移除。

```text
bindpack/
├── config.py                  BindPackConfig 配置组合与读取
├── configuration/
│   ├── defaults.json          runtime/candidates/route/plan/pack 默认数值唯一来源
│   └── *.py                   各组 schema、类型与范围校验
├── retrieval/engine.py        Route 窗口调度与在线检索
├── pipeline.py                共享候选、Plan/Pack 显式组合
├── stages/
│   ├── route/                 seeds、bridge、iterative、QD/PCQD、path-set
│   ├── plan/                  structure、planner、canonicalizer、validator、executor、binder、verifier
│   └── pack/                  scorer、coverage、redundancy、DAG package、final safeguard
├── utils/evidence.py          共享文本与来源分支工具
├── index/                     图/向量/OpenIE 索引、IDF 与 bridge cache
├── core/                      模型初始化、dense retrieval 与 reader
├── storage/                   向量/文本存储
├── embedding_model/、llm/     模型适配与请求缓存
└── evaluation/                检索、QA、独立 gold 图与 FCR/DCC 报告
```

Plan 使用初始 base 和 QD/PCQD 的独立 DPR 候选，在最终 Route fusion/path-set 以前执行；Pack 结合最终 Route 排名和 Plan state。目录整理保留这条原 Full 执行关系。

`EvidencePipeline` 组合 `PlanStage`、`PackStage`；Plan 显式组合 planner/executor/verifier，Pack 组合 scorer/coverage/DAG selector。算法不再依靠旧 EvidenceRetrieval/Mixin 多层继承。

默认模型为 Qwen3-8B + Qwen3-Embedding-8B。嵌套参数通过 `cfg.candidates`、`cfg.route`、`cfg.plan`、`cfg.pack` 访问；默认值只存于 `configuration/defaults.json`。`configs/ablations_qwen3.json` 管理资源与实验编排，并引用方法参数文件。

```python
from bindpack import BindPack, BindPackConfig

cfg = BindPackConfig.from_file(stage_ablation="full", save_dir="outputs/my_run/index")
rag = BindPack(config=cfg)
```

| 模式 | 行为 |
| --- | --- |
| `full` | 完整三阶段和最终 safeguard |
| `wo_route` | 基础图候选适配器，移除增强 Route；保留 Plan/Pack |
| `wo_plan` | 不生成/执行依赖 DAG、绑定或 proof；Route + flat coverage/redundancy Pack |
| `wo_pack` | Route/Plan 保持，仅 base/goal pointwise；无 greedy、DAG 和 final safeguard |

`wo_route` 是项目内的基础候选适配器，外部 HippoRAG2 使用独立入口；`wo_plan` 跳过依赖 DAG package。Qwen embedding 必须使用与模型、语料和建库身份匹配的索引，可以复用已核验的 Qwen 索引。`retrieve` 输出检索指标，`rag_qa` 同时执行 reader；FCR/DCC 在结果保存后独立计算。

阶段定义、实验设计、配置和运行命令见 [实验图表](../../docs/实验图表.md)，FCR/DCC 与报告协议见 [指标说明](../../docs/指标说明.md)，各对照方法及论文出处见 [Baseline 说明](../../docs/Baseline说明.md)。
