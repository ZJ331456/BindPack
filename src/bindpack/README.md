# BindPack 源码导航

## 按职责阅读

| 目录 | 职责 | 首先阅读 |
| --- | --- | --- |
| `index/` | NER、OpenIE、图构建、检查点、索引发布与实体缓存 | `shared_index_builder.py`、`graph_index.py`、`artifacts.py` |
| `retrieval/` | 查询预取、跳数、图检索、QD/PCQD、排序与结果编排 | `engine.py`、`path_optimizer.py` |
| `evidence/` | 子目标依赖执行、来源绑定、证据包选择和结果核验 | `README.md` |
| `core/` | 公共运行时初始化、HippoRAG 风格检索与 QA 原语 | `base.py` |
| `storage/` | 持久化向量与文本存储 | `embedding_store.py` |
| `embedding_model/`、`llm/` | 模型适配、请求缓存与并发控制 | 对应模型适配器 |
| `prompts/` | 索引及公共任务提示词 | `templates/` |
| `evaluation/` | Recall、QA 指标和逐题导出 | `result_export.py` |
| `utils/` | 共用配置、文本、日志及类型工具 | `config_utils.py` |

`config.py` 定义算法配置。包顶层保持 `from bindpack import BindPack, BindPackConfig`。

## 当前正式算法

正式 runner 使用 stage 4，并启用：

```text
planning,plan_prune,dag_package,support_semantic_veto
```

`BindPackConfig()` 默认就是上述完整方法，并开启原有 PCQD、迭代检索和路径优化。
正式脚本还会固定模型、语料、并发与预算。仅修改方法名称不等于相同实验条件。
历史 stage、恢复类扩展和未使用的评分/选择实验已删除。

调用关系：

```text
BindPack（retrieval/engine.py）
  ├─ IndexArtifactsMixin → 实体 IDF / bridge 缓存
  ├─ BaseRAG → GraphIndexMixin → NER / OpenIE / 图与检查点
  └─ ImprovedEvidenceRetrieval（Planner → DAGPackage → EvidenceRetrieval）
       → 依赖规划与绑定 → 候选检索 → 证据包选择 → 支撑保护与语义否决
```

索引方法从旧类移入 mixin，原方法正文保持一致，`super()` 次序保留原来的覆盖关系。
索引文件名、已有向量/图结构和 `QuerySolution` 位置保持兼容。

## 保留的兼容入口

根目录的 `BaseRAG.py`、`bindpack.py`、`embedding_store.py`、`rerank.py` 仅重导出
原有公开类，实际实现只保留一份。顶层 `PathCondRAG` / `PCRAG` 及配置别名仍可用。
内部证据模块按新目录引用；不为每个旧内部路径复制一个包装文件。

当前仅保留完整方法及其实际调用的工具；现有四个组件可按 flags 关闭。
完整清单及验证见 [最终方法精简说明](../../docs/最终全量方法代码精简_2026-10-11.md)。
