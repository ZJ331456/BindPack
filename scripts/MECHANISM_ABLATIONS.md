# 当前机制消融与Full

2026-11-08入口：`bash scripts/run_redesigned_ablations.sh`。模型与数据编排使用 [redesigned_ablations_qwen3.json](../configs/redesigned_ablations_qwen3.json)，旧阶段配置 `ablations_qwen3.json` 保持原样。

正式协议以 [消融重新设计.md](../docs/消融重新设计.md) 为准，按用户本轮要求增加父答案传播组。HotpotQA、2WikiMultiHopQA、MuSiQue、HoVer各1000题，6组共24组结果；每组同时执行检索和QA。本轮不运行阶段消融，阶段定义保持原样。

| 名称 | 配置 | 唯一修改 |
|---|---|---|
| Full | `mechanism_ablation=full`、`pack_mechanism_ablation=none` | 完整方法 |
| w/o Source Binding | `mechanism_ablation=wo_source_binding` | Pack使用原答案抽取候选构成的非绑定来源集合，不访问精确答案—来源映射和来源级父上下文 |
| Reference (S₀) | `mechanism_ablation=wo_joint_selection` | 直接返回Full的参考集合，跳过有界搜索和Reader-visible rescoring |
| Flat Support | `pack_mechanism_ablation=flat_dependency_support` | 闭包奖励替换为单节点支持奖励，保留来源包动作与其他目标项 |
| Singleton Search | `pack_mechanism_ablation=singleton_search` | 只允许单篇扩展，保留完整闭包目标与同等搜索预算 |
| w/o Parent Answer Propagation | `mechanism_ablation=wo_parent_answer_propagation` | 保留Full Route与DAG，用父原问题描述展开占位符，禁用父答案替换和父答案上下文注入；重新执行Plan检索和答案抽取 |

五种机制开关互斥，不与阶段消融叠加。来源绑定不等于严格来源资格核验：当前Full的 `source_bound_eligibility=false` 保持不变，Plan仍保留模型声明的答案来源。相关性权重0.60、原文预算5、`selector_mode=combined`、`joint_fixed_prefix=0`、`joint_state_cap=4096`；Review和Support / semantic veto关闭。各变体不按预期下降幅度调参。

Full只执行一次Route和Plan并保存真实Pack输入，四个Pack消融消费该输入的独立副本；相同候选、DAG、抽取答案、物理Passage ID和参考集合可以逐题核对。Parent消融复用Full的Route和DAG，重新执行受传播影响的子问题检索与抽取，保留来源绑定、依赖闭包目标、来源包动作和联合搜索。

索引固定为 `/root/BindPack/outputs/shared_indexes/<dataset>/`。复用现有graph、passage/entity/fact向量和OpenIE；使用独立工作目录与新应用缓存，禁止索引构建。验证现有语料、模型身份、图与向量一致性，记录资产哈希，不把缺失的历史建库成本写成零成本。

Reader统一使用原生prompt和答案解析，5篇原文、8192总上下文、2048输出预算。输出Recall@5、FCR@5，MuSiQue额外输出独立gold图NodeCov@5、DCC@5、ClosureGap@5；前三个QA数据集输出EM/F1，HoVer输出标签Accuracy/Macro-F1。保留相对Full的pp变化与每题Passage ID、分数、搜索扩展数和耗时。

输出目录为 `outputs/ablations_redesigned_20261108/`。预检独立保存，通过后进入正式队列；保留逐题断点、冻结代码和配置、原始响应与运行日志。恢复使用相同入口，不重跑已核验完成的结果。耗时估算来自本机完整共享索引上的真实预检，不把窗口摊销时间当作独立单题延迟。

执行方式为整组串行：每个数据集先完成Full的1000题检索、QA及报告，再按表中顺序逐组完成五个消融；当前组验证完成后才会释放下一组。组内8/16/32客户端并发使用同一32题校准，最终选值记录在冻结的 `concurrency_profile.json`；客户端并发与vLLM物理序列上限分别报告。

```bash
tail -f /root/BindPack/outputs/ablations_redesigned_20261108/run.log
```
