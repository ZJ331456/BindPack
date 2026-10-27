# 完整机制消融

入口：`bash scripts/run_complete_mechanism_ablations.sh`。源码、配置及数据在队列首次启动时冻结；所有新消融都重新执行 Route 和 Plan，不复用历史 LLM 响应。只复用不可变的图、向量与 OpenIE 索引。不运行最终 QA，QA 指标为 N/A。

| 版本 | Plan 的中间答案 | Plan 执行 | Pack 来源资格 | Pack 闭合 | 新联合选择 |
|---|---|---|---|---|---|
| Full | 原文来源核验 | 依赖拓扑、答案传播 | 开启 | 旧 DAG＋闭合奖励＋包动作 | 开启 |
| w/o Source-Bound Eligibility | 普通答案抽取 | 保留依赖执行 | 全部关闭 | 使用未核验的声明来源计算，明确标记非证明 | 保留 |
| w/o Dependency Closure | 原文来源核验 | 独立子问题、无答案传播 | 保留 | 全部关闭 | 保留，只用单文档动作 |
| w/o Joint Selection | 原文来源核验 | 保留依赖执行 | 保留 | 保留旧顺序 DAG 处理 | 整个 BudgetedSelector 关闭 |

来源资格关闭范围包括：Plan 原文引句与答案绑定协议；共享池/相关性中的分支资格；旧 DAG 和新联合选择中的答案、分支、引句、关系、类型、置信度等资格门槛；来源保留及可见原文校验。只保留 JSON/字段格式、有效文档 ID、原有200字符答案表示预算和现有候选预算；数字年份等标量按原接口转换为字符串。未通过来源核验的记录不标作真实证明。

依赖关闭使用独立规划器与独立执行器，不调用原依赖规划器、拓扑层执行器或答案替换。使用与Full相同的文本结构路由和native hop确定逐题节点上限，避免独立方案额外获得更多检索请求预算。旧 DAG 选择、新闭合奖励及闭合包动作全部不执行；没有用平坦节点覆盖冒充“完全关闭”。移除严格闭合通道后沿用现有“无支持通道”的软覆盖 fallback（rho=1），配置系数不调参。

联合选择关闭保留 Coverage→DAG 的顺序方案作为对照，不执行新 BudgetedSelector 的集合目标或 beam 搜索。这与整个 w/o Pack 的 pointwise 排序是不同的对照；具体保留项在表内公开。

配置通过 `BindPackConfig.from_file(..., mechanism_ablation='wo_dependency_closure')` 或 `configs/mechanism_wo_*.json` 使用。与阶段消融互斥；末端 Support / semantic veto 始终关闭；权重0.60、预算5、模型、8192上下文、2048输出上限、embedding batch4、LLM预取8及原生8题窗口不变。vLLM 不重启、不改配置。

每题 trace 保存实际开关。最终审计必须确认被删除的模块、依赖边、答案传播、闭合包或联合搜索没有残留。先运行四数据集×三消融×三题真模型预检，全部通过后才运行12组全量，各1000题。独立开发预检缓存不带入全量。失败保留断点和全部实际成本，停止队列，修复后续跑。

Full 引用最新四套已有全量。新增开关默认全部开启；另用4000条已保存的 Full 输入逐题核对当前原生 Pack 的 Top200、参考集合、候选域和目标分数。该检查不生成新的 Full 主实验成绩。新消融是独立上游运行，不能将所有差异只归因于末端排序。

结果保存在 `outputs/4_datasets_result/mechanism_complete_20261027/`：`RESULTS.md`、`summary.json`、`status.json`、`protocol.json`；各组的 report、result、逐题断点、模型请求/tokens、embedding、资源采样、审计及索引校验均保留。为避免重复存储200篇原文，只导出前10篇文本，同时保留完整排名的内容哈希、全部 gold 排名及完整原生 trace；原语料与索引保持不变。

计时区分真实上游/选包时间、准备上下文时间和未执行的 QA。预先渲染的 Reader@5 预算仅标作 prepared，不称为实际 Reader 调用。复用索引的历史建库成本不可核验时仍为 N/A。
