# 当前Full及20组消融

2026-10-29当前入口：`bash scripts/run_final_ablation_queue.sh`。

来源资格、Support / semantic veto均关闭，相关性权重0.60，原文预算5；LLM和Embedding为当前Qwen3配置。所有正式消融都重新执行原生Route/Plan，复用通过核验的公共索引，不复用历史模型响应。最终QA不执行。

| 类型 | 名称 | 删除范围 |
|---|---|---|
| 机制 | w/o Dependency | Plan换为独立子问题规划/执行；不做拓扑层次、答案替换和祖先传播；Pack关闭旧DAG、新闭合奖励及包动作，保留非依赖集合选择 |
| 机制 | w/o Joint Selection | 关闭整个新增BudgetedSelector；保留Coverage→旧DAG参考方案 |
| 阶段 | w/o Route | 关闭Route阶段，保留Plan和Pack允许的操作 |
| 阶段 | w/o Plan | 不执行整个Plan；Route的静态/路径条件查询仍按阶段协议保留 |
| 阶段 | w/o Pack | 跳过覆盖贪心、旧DAG、联合选择和末端处理，使用既有pointwise排序 |

本轮不运行w/o Soft Coverage，也不重新运行来源资格消融。当前Full本来已经关闭来源资格。

顺序固定：第一套按HotpotQA→2Wiki→MuSiQue→HoVer，每套先去依赖、后去联合（8组）；第二套仍按相同数据集顺序，每套去Route、去Plan、去Pack（12组）。每组全1000题，不以结果好坏决定是否继续。

窗口和客户端并行数仅用于吞吐校准，不更改算法参数、embedding batch4或vLLM启动参数。窗口耗时除以题数是摊销吞吐，独立单题延迟不能由此推出。校准和两题预检独立记账，不混入正式20组成本。

当前Full原始数据位于`outputs/4_datasets_result/final_full_no_source_20261028/`，公共索引位于`outputs/4_datasets_result/shared_indexes/`。旧大文件已经清理，原始Full协议中的旧路径以`relocation_20261029.json`映射。来源资格开启的历史配置只作历史参考，不是当前默认基线。

队列输出：`outputs/ablations_full_20261029/`；`RESULTS.md`和`summary.json`只列通过完成及账本校验的组。每题原始trace、完整排名哈希、gold排名、请求/tokens、embedding计时和资源采样保留。索引构建被禁止；无法核验的原始建库成本仍为N/A。

恢复执行使用同一入口，保留所有已完成题目、未写完窗口以及实际失败尝试成本。不会重跑已核验完成的组，也不会将格式错误当成成功证据；解析失败沿用当前原生弃权策略。
