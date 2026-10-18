# PackStage

`PackStage` 组合 `EvidenceScorer`、`CoverageSelector`、`DAGPackageSelector`。`redundancy_controller.py` 计算归一化词集合 Jaccard；`dependency_closure.py`、`proof_support.py`、`proof_scope.py`、`terminal_support.py` 提供依赖与来源约束。最终提案、来源保护与独立审核位于 `verification/`，不是旧路径的兼容导出。

先按 base/goal 得分、覆盖增量和冗余选择前缀，再做 Top5/Top10 有限祖先闭合提升。默认保护 Top2、再保护已选 Top5，保持文档集合和 Top200 集合。闭合针对预测计划及本地合格来源，不保证任意返回前缀都是纯闭合包。

参数从 `runtime.cfg.pack` 读取，唯一默认值位于 `configuration/defaults.json`。`wo_plan` 保留 flat greedy 并跳过 DAG selector；`wo_pack` 使用原 base/goal pointwise 分数并跳过 greedy、DAG 与 final safeguard。
