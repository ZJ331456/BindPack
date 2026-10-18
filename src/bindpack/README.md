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
└── evaluation/                检索与 QA 指标
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

`wo_route` 不是外部 HippoRAG2 的独立复现，`wo_plan` 不具有完整 DAG Pack。Qwen embedding 使用独立新索引。`retrieve` 只有检索指标，`rag_qa` 才有 reader QA 指标。

完整机制与边界见 [模块说明](../../docs/模块组合-消融说明.md)，运行方法见 [四组实验](../../docs/四组实验-Qwen3.md)。

本轮验证见 [第二轮重构验证（2026-10-18）](../../docs/三阶段重构验证_2026-10-18.md)。
