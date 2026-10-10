# 最终 BindPack 证据模块

本目录只保留已评估的
`exp4_dependency_binding_plan_prune_dag_package_support_semantic_veto`
及其直接调用的工具，共 **17 个实现文件**，不含 `__init__.py`。

## 执行顺序

```text
问题结构路由 → 依赖计划与 plan_prune
    → 单分支依赖执行、逐字来源绑定、候选覆盖排序
    → 祖先闭合 DAG 证据包选择
    → 原问题交换建议 → 已有支撑保护 → 独立语义否决
```

正式阶段固定为 `4`，每次只保留一个绑定分支。四个正式开关是：

```text
planning,plan_prune,dag_package,support_semantic_veto
```

可关闭已有开关做相应对照。全部关闭仍执行基础 stage 4 的依赖检索和覆盖排序；
基础引用绑定不需要额外 `binding` 开关。

## 文件职责

| 部分 | 实现 | 作用 |
| --- | --- | --- |
| 入口 | `evidence_retrieval.py` | 单分支依赖执行、候选搜索、原始引用核验及覆盖排序 |
| 组合 | `evidence_improvements.py` | 仅组合 Planner、DAG 与基础 EvidenceRetrieval |
| 规划 | `planning/evidence_planning.py` | 按问题结构扩大并行分支容量、规范引用、剪除纯终端比较运算 |
| 路由 | `planning/question_structure.py` | 识别明确的独立问题分支，含糊结构保留原始规划输入 |
| 绑定工具 | `planning/evidence_binding.py` | 同一来源中的实体身份与个人属性检查 |
| 关系工具 | `planning/evidence_relation_guard.py` | 有限语法中的关系方向与事实主体检查 |
| 闭合包 | `selection/evidence_dag_package.py` | 枚举祖先闭合证据包，按原目标选择 Top5、Top10 |
| 图工具 | `selection/evidence_selection.py` | 计算和验证计划的传递祖先集合 |
| 引用工具 | `selection/evidence_package.py` | DAG 使用的来源身份、创办人和局部引用检查 |
| 事实工具 | `selection/evidence_terminal.py` | DAG 使用的答案类型、主体和关系检查 |
| 作用域 | `selection/evidence_scope.py` | proposal 范围、受保护证据和问题实体锚点谓词 |
| 提案 | `selection/evidence_joint_selector.py` | 原问题 Top5 缺失约束建议及逐字来源、排名不变量校验 |
| 类型限制 | `selection/evidence_typed_joint_selector.py` | 为不支持的聚合、列表、布尔等目标提前弃权 |
| 支撑保护 | `verification/evidence_support_guard.py` | 保留当前绑定来源、问题锚点及 DAG 支撑 |
| 审核输入 | `verification/evidence_review_prompt.py` | 原 v3 独立审核提示词和固定物理文档 ID；每篇可见来源为 2000 字符 |
| 语义决定 | `verification/evidence_semantic_swap_gate.py` | 检查审核 ID、完成状态、四项布尔判断与解释 |
| 最后处理 | `verification/evidence_support_semantic_veto.py` | 有界并发提案/审核、上下文预检、Top5 交换和诊断导出 |

`evidence_package.py`、`evidence_terminal.py` 和 `evidence_selection.py` 当前都是正式
DAG/veto 复用的工具，已经删除其历史独立 reranker 和 mixin。
`evidence_joint_selector.py` 是正式 veto 的提案工具，不是后来的
`dependency_joint` 评分实验。

## 已移除的历史扩展

- 桥接恢复、条件 companion、结构恢复、source witness 和失败恢复。
- 独立 `binding` 增强 verifier、selection/closure、support、terminal、package。
- adaptive 额外文档核验、多分支 beam、dependency-depth 新规划路由。
- 独立 anchor completion / bridge companion 排序。
- 来源句单元核验器及 v1/v2/v3 的历史审核 validator。
- 预算 QD/迭代、固定池等历史消融实现。

原 v3 正式审核提示词、v1 的 proposal ID 约束和 v2 的可见来源长度已合并到
`evidence_review_prompt.py`，保留实际消息字节和检查语义。模型的语义批准仍是
可出错判断，逐字引用与本地关系检查不构成一般语义蕴含保证。
