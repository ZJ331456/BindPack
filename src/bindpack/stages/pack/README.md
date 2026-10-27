# PackStage

`PackStage` 组合 `EvidenceScorer`、`CoverageSelector`、`DAGPackageSelector` 与 `BudgetedSelector`。`redundancy_controller.py` 计算归一化词集合 Jaccard；`dependency_closure.py`、`proof_support.py`、`proof_scope.py`、`terminal_support.py` 提供依赖与来源约束。可选末端提案、来源保护与独立审核位于 `verification/`，当前默认关闭。

先按 base/goal 得分、覆盖增量和冗余选择前缀，再做 Top5/Top10 有限祖先闭合提升，构造参考包S0。参考构造阶段保护Top2、再保护已选Top5；最终默认由联合选择器比较候选文档集，允许替换S0任意位置，保持文档集合和Top200集合。闭合针对预测计划及本地合格来源，不保证任意返回前缀都是纯闭合包。

参数从 `runtime.cfg.pack` 读取，唯一默认值位于 `configuration/defaults.json`。`wo_plan` 保留 flat greedy 并跳过 DAG selector；`wo_pack` 使用原 base/goal pointwise 分数并跳过 greedy、DAG 与 final safeguard。

`pack.review_mode` 可选 `off`（当前默认）、`standard` 和 `suffix_stop`。`off` 关闭末端 Support / semantic veto，不调用其模型或加载其 tokenizer，保留Pack选包，并在trace明确记录关闭及零额外请求。它不等同于`wo_pack`。联合选包仍使用本地tokenizer核对Reader可见上下文，不做模型推理。

`suffix_stop` 保留相同提示、适用范围、引文/来源校验和独立审核，在关键字段生成后通过服务端 stop 停止说明后缀。原始响应、实际 usage 和 stop 序列均保存；只补 JSON 闭合符，必需的 ID、引文、事实或布尔判断缺失时仍然放弃替换，不重试模型解析。

历史核验对照可显式加载 `configs/pack_review_suffix.json`；该配置同时选用`current`参考选择器，并保持权重0.60。联合选择器要求`review_mode=off`，不能通过单独覆盖review_mode悄悄开启末端核验。历史冻结运行不受工作区默认值影响。

统一选包入口为 `budgeted_selector.py`，`selector_mode`可选
`current/closed/residual/risk/sources/combined`。当前默认是已完成四数据集全量实验的
`combined`，采用`joint_free`设置。联合模式要求`review_mode=off`，
以当前无核验结果S0及同一候选域为参考，以有界集合搜索
比较闭合支持、残余覆盖、来源替代和支持丢失风险。残余覆盖等权使用已有子目标
排名覆盖和单句内unigram/bigram与候选局部IDF词法信号，不是语义核验。
没有合格proof时将覆盖权重交给可用软通道；引用在实际Reader渲染后重新检查。
搜索同时有确定的状态上限与应急时限，属于近似求解。所有已返回合法来源仅在显式
采集或实验模式下记录，不改变Plan的单分支和默认Full的旧trace。

`configs/pack_joint_free.json` 与当前默认Pack参数一致；
`configs/pack_joint_experimental.json` 显式保留Top2与2048状态预算，用于对照。
配对100+100实验脚本为 `scripts/run_pack_joint_experiment.py`；上游采集与
Pack/Reader成本分开，任何失败和质量回退均保留，不按新结果换题。

`joint_fixed_prefix=0` 为当前默认，允许替换参考Top2；设为`2`可保留它们。
当前使用4096状态上限，为五个自由位置提供搜索空间，
因此它同时改变约束和计算预算，不能作为完全等成本的单因素对照。
论文实验若要单独归因该约束，还需匹配相同的状态预算。
`joint_search_seconds=5.0` 为应急时限而非实测耗时，
实际搜索通常先达到确定的状态上限，耗时以 report 为准。

最新研究口径只以 Recall@5 / FCR@5 决定是否保留，QA不参与门槛；
尚未生成回答的整批QA指标记 N/A。新选包的闭合是预测结构与本地引文
支持的闭合，软覆盖不是语义证明，也不保证追回原LLM核验的收益。
