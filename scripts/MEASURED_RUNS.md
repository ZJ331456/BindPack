# BindPack Full 的计量运行

`run_bindpack_measured.sh` 运行单组；`run_bindpack_measured_cases.sh` 将已有Full批次扩展为四组，严格按Full → w/o Route → w/o Plan → w/o Pack串行执行，不启动外部baseline。它们调用原生 BindPack 的建库、检索、规划、Pack、QA prompt 与答案解析；模型和方法参数读取 `configs/ablations_qwen3.json` 与其引用的方法文件。查询顺序执行，单题内原生 LLM 并发上限8，embedding batch=4，与 baseline 的计量调度对齐。

```bash
# 小样本：3题、32篇开发夹具，仅验通，分数不用于论文
bash scripts/run_bindpack_measured.sh --dataset hover --smoke \
  --output outputs/hover_full_measured_smoke_20261024

# 完整的本地1000题与完整corpus；使用新目录
bash scripts/run_bindpack_measured.sh --dataset hover \
  --output outputs/hover_full_measured_20261024

# 只在该批任务停止后恢复：保留完成题、原索引、缓存和实际请求账本
bash scripts/run_bindpack_measured.sh --dataset hover --resume \
  --output outputs/hover_full_measured_20261024

# 已有Full任务须先正常停止，再扩展或恢复整个串行队列
# Full接最后完成题；三个消融分别从第1题独立评测
bash scripts/run_bindpack_measured_cases.sh --dataset hover \
  --output outputs/hover_full_measured_20261024
```

串行入口保持原Full的`source_snapshot`、协议和已完成题不变。新的`ablation_source_snapshot`复制Full的原生算法，只增加`--case`及索引隔离等执行控制，差异记录在`serial_extension/ablation_overlay.json`。各组结果统一位于`cases/<dataset>/<full|wo_route|wo_plan|wo_pack>/`，每组都有独立报告、调用账本与断点。复用图/向量/OpenIE的私有副本，不使用可写硬链接，不继承Full或其他消融的LLM缓存。共享原始建库成本只计一次，各消融另记本组复制/验证/加载准备成本；`shared_original_index`仅为来源说明，不重复加进本组物理请求总量。

`queue_status.json`显示四组总进度，`status.json`显示当前组。根目录`completed.ok`仅在四组全部完成后生成，原Full单组根标记归档到`serial_extension/single_case_markers/`；各组自己的`completed.ok`保持独立。`comparison.json`和`comparison.md`按已完成组更新。再次调用串行入口时跳过校验通过的完成组，并恢复未完成组的断点。已完成Full仍用其原始冻结worker；消融用独立冻结控制代码，后续工作区修改不影响该队列。

首次启动复制源码、配置和数据；子进程从 `source_snapshot` 执行。恢复使用同一快照，拒绝变化的配置、样本身份和检查点。当前源码后续调整不会改变正在执行的快照。

每题先调用原生检索，再调用原生 reader；不把 gold 文档或答案传给检索。Hop 继续使用仓库现行的参考标注规则，记录 `oracle_hops=true`，这一条件须随效果表披露。HoVer 按 `hover_id`、`claim`、原生二分类标签处理；没有可信 gold DAG，DCC/NodeCov/ClosureGap 输出 N/A。默认补齐 corpus 缺失的精确 gold 文本，记录补入数量和冻结摘要；HoVer 与现有 baseline 的完整语料均为9441篇。

主要输出在 `cases/<dataset>/full/`：

- `result.json`：排名、答案、原生 trace、逐题计时与 Reader 预算、完整统计。
- `report.json`：Recall/FCR@3/5/8/10/20/200、EM/F1、HoVer Accuracy/Macro-F1、适用的结构指标，以及 `efficiency/cost/evidence_budget/resources`。
- `metrics_details.json`：逐题指标、标签与依赖诊断。
- `progress.jsonl`：每题完成立即落盘的断点，保存完整排名与行摘要。
- `llm_calls.jsonl`：每次真实 SDK 请求，含重试、错误、usage、调用ID、阶段、查询ID；缓存命中不生成物理请求记录。
- `llm_logical_calls.jsonl`：逻辑调用、缓存命中与耗时，与物理请求按调用ID关联。
- `embedding_calls.jsonl`：实际编码文本数、有效tokens、截断数量与耗时。
- `stage_timings.jsonl`：OpenIE、passage/entity/fact编码及写库、事实/文档边、相似边、构图写盘与Route辅助索引耗时。命名区间可能包含内部操作，不能和内部embedding耗时重复相加。
- `resource_samples.jsonl`：每2秒采样CPU、RAM、GPU利用率/显存/功率；`audit.json`与`completed.ok`只在全部题目完成且账本/排名/身份一致后生成。

逐题端到端区间包括查询编码、原生检索/规划/Pack、提示准备、Reader和重试；排除模型加载、离线建库和断点写盘。模型加载和建库分开计量。Reader使用与baseline相同的8192总上下文、2048预留输出和前缀截断；只影响送给Reader的文本副本，不改变检索排名，部分/丢弃passage显式计数。`evidence_tokens`采用baseline的完整证据块（包括`Wikipedia Title:`），另保存不带标记的`source_text_tokens`。

CPU资源针对BindPack进程及子进程；CPU百分比100%代表一个逻辑核。GPU采样针对整张共享GPU，包含独立vLLM与embedding，不能当成该方法独占GPU时间。另保存BindPack进程的PyTorch显存分配峰值，不包含独立vLLM进程。采样峰值是观测最大值，不保证捕获瞬时峰值。货币费用未设置单价、独占GPU秒无法归属时继续为N/A。

LLM失败请求未返回usage时，保留未知标记，报告`token_usage_complete=false`，不把未知token当作真实零成本。实际物理调用、逻辑调用和缓存命中分别统计。单纯采集计量不重置共享vLLM缓存；该条件明确记录在报告。

`llm_retries`专指一次原生infer内的HTTP重试；原生OpenIE的结构恢复请求属于额外逻辑调用，全部计入请求和tokens，其恢复历史保存在原生索引的OpenIE记录中。计量入口不改变已有的格式恢复预算或失败排除策略。

证据预算曲线仍需显式运行对应Reader预算，单次Reader@5不等于已完成@3/8/10的QA实验。本入口当前固定Reader@5，不会自动扩大本次HoVer任务。
