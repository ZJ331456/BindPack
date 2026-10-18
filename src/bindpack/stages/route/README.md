# RouteStage

Route 将通过 fact filter 的种子、桥扩展、有限第二轮检索、静态 QD/PCQD 和轻量 path-set 汇总为候选排名。实现位于 `query_seed_prior.py`、`bridge_expansion.py`、`iterative_refinement.py`、`route_refinement.py`、`path_set_consolidation.py` 与 `path_set_selection.py`。

参数按 `config.route.seed/bridge/iterative/refinement/paths/index` 分组，数值仅定义在 `configuration/defaults.json`。Stage 的内部组件共享检索器和图索引，保留原 Full 的先验、融合、prompt 与稳定排序规则。

`wo_route` 调用 `baseline.py` 的固定适配器：fact-frequency seeds、DPR restart 与固定 PPR damping，关闭增强 Route 的候选通路。该适配器使用本项目公共索引与 fact filter，不代表外部 HippoRAG2 的独立 benchmark。
