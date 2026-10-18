# PlanStage

`PlanStage` 显式组合 `DependencyPlanner`、`DependencyExecutor`、`SourceVerifier`。问题结构控制位于 `question_structure.py`，依赖规范化和验证分别位于 `plan_canonicalizer.py`、`plan_validator.py`、`validation.py`；`answer_binder.py` 只用已核验中间答案填充问题。来源实体/属性规则在 `answer_validation.py` 与 `relation_guard.py`。

Planner 读取问题与 hop/节点/深度预算，不读取候选原文。简单链与明确并行结构使用原 Full 的结构策略；只保留一个正式 planner API。执行器按拓扑层检索并绑定来源，默认单分支、核验 Top3、每篇 4000 字符；父答案缺失时不搜索未绑定后继。

参数统一从 `runtime.cfg.plan` 读取，默认数值来自 `configuration/defaults.json`。`wo_plan` 在 Pipeline 层跳过整个 Plan 并清空 DAG/绑定/proof，保留平坦子目标候选。
