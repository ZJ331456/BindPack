# 四组实验：Qwen3-8B + Qwen3-Embedding-8B

主入口是 `scripts/run_bindpack_ablations.py`，Shell 入口是
`scripts/run_bindpack_ablations_qwen3.sh`，默认新建共享索引。
`scripts/run_bindpack_four_cases_reuse.sh` 复用已经核验的三个 Qwen 索引。
BindPack 的阶段模式在两个入口中都是相同四组。默认数据集为
HotpotQA、2WikiMultihopQA、MuSiQue：新建入口默认输出 12 个 BindPack 结果目录；
复用 Shell 入口默认额外执行外部 HippoRAG 2 复现，共 15 个结果目录，并开启 reader QA。

| 名称 | `stage_ablation` | 实际对照 |
| --- | --- | --- |
| Full BindPack | `full` | 原完整 Route、Plan、coverage/redundancy greedy、DAG package 和最终 safeguard |
| w/o Route | `wo_route` | 改用 fact-frequency seeds、DPR restart 和固定 damping 的基础图检索；关闭增强 seed/IDF、bridge、iterative、QD/PCQD 和 path-set，保留 Plan/Pack |
| w/o Plan | `wo_plan` | 不生成/执行依赖计划，不绑定中间答案，不产生 Plan proof；Route 和 flat coverage/redundancy Pack 保留，DAG package 因没有计划跳过 |
| w/o Pack | `wo_pack` | 保留 Route/Plan 候选和状态，以原有 base/goal pointwise 融合分数排序；关闭 coverage/redundancy greedy、DAG package 和最终 safeguard |

`wo_route` 是本项目内的基础候选发现适配器，不是外部 HippoRAG2 的完整独立复现。
`wo_plan` 保留最终 safeguard，但缺少 Plan proof 保护；`wo_pack` 关闭 safeguard，
避免最终重排补回被消融的 Pack。四组分别表示上述完整系统控制；`stage_ablation` 只接受这四个值。
独立 HippoRAG 2 复现是额外方法基线，不是第五个 BindPack 阶段模式。

## 配置放在哪里

- `src/bindpack/configuration/defaults.json`：算法默认参数唯一来源，按
  `runtime/candidates/route/plan/pack` 分组。
- `src/bindpack/configuration/*.py`：各组的类型与范围校验；`config.py` 组合
  `BindPackConfig` 并读取方法参数，没有重复的算法默认数值或历史 flags。
- `configs/ablations_qwen3.json`：实验模型资源、服务、数据、Python、批大小、输出、
  reader 与四组编排；`method_config` 引用方法参数文件，`--method-config` 可覆盖。
- 输出路径由 `--out-root` 指定；默认正式结果为
  `outputs/stage_ablations_qwen3`，smoke 为 `outputs/stage_ablations_qwen3_smoke`。

主脚本把有效方法参数复制为输出根目录的 `method_config.json`，记录源文件哈希与内容，
并传给四组评测；每个结果也导出相同参数和哈希。修改权重或预算时复制一份方法 JSON，
在新的输出目录运行，避免覆盖已经开始的协议。

配置优先级：命令行资源参数 > 实验 JSON 的 runtime/evaluation > 所引用方法文件 > 包内默认值。`route/plan/pack/candidates` 的算法数值只从方法 JSON 读取；runtime 资源允许在实验配置中明确覆盖。

当前机器默认使用：

```text
LLM 模型文件        /root/models/Qwen3-8B
LLM API model ID    qwen3-8b
LLM API 地址        http://127.0.0.1:8035/v1
Embedding 模型      /root/models/Qwen3-Embedding-8B
Python              /root/anaconda3/envs/rag/bin/python
Qwen3 依赖覆盖目录  /root/.cache/pathcondrag/runtime_deps_qwen3_tf4513
数据集根目录        /root/datasets
```

LLM 使用 OpenAI-compatible 服务。当前机器的 vLLM 环境为 `/root/anaconda3/envs/vllm`。
可在单独终端启动与默认配置地址对应的服务：

```bash
PYTHONPATH=/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513 \
/root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8035 \
  --model /root/models/Qwen3-8B --served-model-name qwen3-8b \
  --trust-remote-code --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.45 --enforce-eager \
  --max-num-batched-tokens 2048 --max-num-seqs 4 --generation-config vllm
```

本机 LLM 与 embedding 共用 GPU，实测使用上述 `0.45` 显存比例和 8192 context；先等 LLM 服务完成初始化，再运行会加载 embedding 的实验入口，避免 embedding 抢占服务启动时的 KV cache 内存测量。

确保该服务 `/v1/models` 暴露 `qwen3-8b`；
模型 ID 与本地模型路径是两件事。Embedding 通过项目的 SentenceTransformer 后端本地加载。
Qwen3 需要兼容的 transformers；当前 `rag` 环境本体的旧版本由上述依赖目录覆盖，
若更换为原生支持 Qwen3 的环境，可在自己的 JSON 中将 `runtime_deps` 设为空字符串。

运行预算沿用原工作流：生成上限 2048、embedding batch=4、OpenIE/预取并发=8、
返回候选 200、展示 10、reader Top5/一次回答。JSON 默认保留原正式脚本的
`openie_strict=false`、`optimized` prompt 和 `structural` 校验；抽取最终失败的关系
会显式排除。若实验要求所有抽取成功，可在独立配置中改为 `openie_strict=true`，
并使用新的输出目录。`structural` 不代表逐条关系已通过语义核验。

## 运行

在 `/root/BindPack` 下执行。先查看配置和命令；这一步不写输出、不加载模型、不请求 LLM：

```bash
bash scripts/run_bindpack_ablations_qwen3.sh --dry-run
```

快速验证一个数据集的四组：

```bash
bash scripts/run_bindpack_ablations_qwen3.sh \
  --smoke --datasets hotpotqa --sample-size 1
```

默认 smoke 从选中问题的完整 context 构造小语料，正式模式使用完整 corpus。
smoke 的问题数默认 2，种子为 42；输出名必须含 `smoke`。smoke 结果只验证流程，
不能作为正式 benchmark 结果。也可以对自己的 tiny fixture 做完全相同的四组验证：

```bash
bash scripts/run_bindpack_ablations_qwen3.sh \
  --smoke --datasets hotpotqa --sample-size 1 \
  --data-path /absolute/path/tiny_hotpotqa.json \
  --corpus-path /absolute/path/tiny_corpus.json \
  --llm-base-url http://127.0.0.1:8047/v1 \
  --out-root outputs/my_tiny_qwen3_smoke
```

显式给出 `--corpus-path` 时，smoke 也使用该语料。单独覆盖数据/语料文件时只能选一个数据集。
原始问题、答案与支持标注保留；同一组 sample indices/IDs 在四组间共享。

正式运行三个完整数据集：

```bash
bash scripts/run_bindpack_ablations_qwen3.sh
```

默认 `eval_mode=retrieve`，产生检索 Recall 等指标。若需要 reader 的 QA EM/F1：

```bash
bash scripts/run_bindpack_ablations_qwen3.sh \
  --eval-mode rag_qa --out-root outputs/stage_ablations_qwen3_qa
```

所有数据集/组使用相同的 reader 设置。正式抽样可用 `--sample-size N`；默认 0 表示全部。
MuSiQue 的 hop 来自问题 decomposition 的 oracle 标签，HotpotQA/2Wiki 使用数据集级 2-hop prior，
会写入样本 manifest，不能把它表述为模型自动预测 hop。

中断后的继续执行，重复同一条命令并增加 `--resume`：

```bash
bash scripts/run_bindpack_ablations_qwen3.sh --resume
```

脚本核对数据/语料、模型配置、源代码、样本顺序、协议、共享/私有索引和已有结果的哈希。
源代码覆盖 `src/` 的 Python 文件与 `scripts/` 的 Python、Shell 文件，包含语料/评测辅助函数和启动脚本；
方法参数文件的原始内容、哈希与输出快照也纳入协议。索引 synonymy 预算属于方法配置，
不再依靠历史环境变量隐式控制。
完成的组直接验证并跳过；未完成组保留该组自己的 LLM cache 继续。修改模型、数据、算法
或实验参数后应使用新输出目录。脚本拒绝覆盖已有的不匹配结果，也不自动删除历史实验。
多个正式实验不能同时占用共享实验锁。

旧 NV2 Shell 文件名与历史细粒度实验入口已删除。新建、复用两个 Shell 使用同一个 runner；BindPack 的阶段模式仍是上述四项，`--include-hipporag` 则另行调用外部方法。

## 共享索引与结果

**Qwen3-Embedding-8B 必须使用与其向量空间匹配的索引，不能复用 NV-Embed-v2 的历史索引。**
即使两个模型向量维度相同，向量空间也不同。新主入口只接受 Qwen3-Embedding-8B，
不读取旧 `3multi_hop_datasets_results_with_nv2_10_8` 实验目录。
`3multi_hop_datasets_results_10_5/shared_indexes` 已有 Qwen3-Embedding-8B 索引，可在核对模型、语料与索引身份后复用。

每个数据集的索引只构建一次，然后给四组复制相同的图、向量、OpenIE 与所需元数据。
采用独立文件副本，不使用共享可写硬链接。每组从空的独立 LLM cache 开始，
不会读到另一组先运行生成的回答。运行前后检查 graph/vector/OpenIE 哈希；
衍生 lookup cache 可以重建，但不允许改动冻结的索引输入。

```text
outputs/stage_ablations_qwen3/
├── protocol.json
├── method_config.json         本次实验冻结的方法参数
├── metadata/<dataset>/
│   ├── corpus.json
│   ├── selected_indices.json
│   ├── sample_manifest.json
│   ├── frozen_index.json
│   └── index_build.log
├── shared_indexes/<dataset>/
├── cases/<dataset>/<full|wo_route|wo_plan|wo_pack>/
│   ├── index/                  独立索引副本与该组 LLM cache
│   ├── protocol.json
│   ├── result.json             排名、候选、逐题诊断、有效配置和请求统计
│   ├── run.log
│   └── validated.ok
├── comparison.json
├── comparison.md
└── completed.ok                所有数据集的四组都通过校验后才生成
```

## 复用 10_5 的三个现有索引

在 `/root/BindPack` 下运行：

```bash
# 只读核对执行计划，不创建输出、不加载模型、不请求服务
bash scripts/run_bindpack_four_cases_reuse.sh --dry-run

# 使用完整旧语料/索引，先评测每个数据集的少量问题
bash scripts/run_bindpack_four_cases_reuse.sh --smoke

# 三个数据集：四组 BindPack + HippoRAG 2，默认检索与 QA
bash scripts/run_bindpack_four_cases_reuse.sh

# 中断后使用相同参数继续
bash scripts/run_bindpack_four_cases_reuse.sh --resume
```

默认复用根目录为 `/root/BindPack/outputs/3multi_hop_datasets_results_10_5`。
三个源索引位于它的 `shared_indexes/{hotpotqa,2wikimultihopqa,musique}`，
每个索引内的模型目录为 `qwen3-8b__root_models_Qwen3-Embedding-8B`。
旧 metadata 记载的数据规模如下；启动时仍会核对实际输入，不能仅依赖文件夹名认定兼容。

| 数据集 | 当前问题文件中的题数 | 完整 corpus 篇数 |
| --- | ---: | ---: |
| HotpotQA | 1000 | 9811 |
| 2WikiMultihopQA | 1000 | 6119 |
| MuSiQue | 1000 | 11656 |

这里的全量指上述本地 1000 题文件的全部问题，不表示上游数据集全部官方 split。
复用入口使用 `/root/datasets/<dataset>.json` 与 `/root/datasets/<dataset>_corpus.json`，
必须和原索引的 SHA 匹配，不能替换为 `*_v2.json` 等不同输入。
四组 BindPack 共享选中的问题顺序、完整 corpus、Qwen embedding、方法参数和评测设置。
复用 Shell 默认设置 `--include-hipporag --eval-mode rag_qa`，所以同时生成检索与 QA EM/F1。
外部 HippoRAG 2 使用同一份冻结索引、问题顺序及模型资源，通过
`/root/baseline/HippoRAG/main.py` 调用其自身检索与 reader 实现；该依赖位于仓库外，
推送本仓库不会携带它。缺少该入口时会在启动前报错。
当前 HippoRAG 调用从 `datasets_dir` 读取标准命名的数据文件，不支持把 BindPack 的
`--data-path` / `--corpus-path` 覆盖传给外部程序；包含该基线的批次应使用上述标准数据文件。
BindPack 协议冻结本仓库的 Python/Shell 文件，尚未包含仓库外 HippoRAG 源码的版本哈希；
复现该基线还需要单独保留外部代码版本。
仅更改评测模式可以追加 `--eval-mode retrieve`，已有批次参数不得原地更改。

如需只执行四组 BindPack，使用底层入口且不加 `--include-hipporag`，并选择新的输出目录：

```bash
/root/anaconda3/envs/rag/bin/python scripts/run_bindpack_ablations.py \
  --reuse-shared-index-root /root/BindPack/outputs/3multi_hop_datasets_results_10_5/shared_indexes \
  --out-root outputs/four_bindpack_only --eval-mode rag_qa
```

复用流程不调用建库入口，不在原索引中迁移、改名或补写身份。
通过核验后，为每组创建独立可写副本；文件系统支持时使用写时复制 reflink，
否则复制独立文件，不使用共享可写硬链接。源索引的 LLM cache 和可重建 Route cache 不继承到各组。
每组成功生成并校验结果后，释放该组临时复制的图、向量与 OpenIE 输入，保留结果、协议、日志及本组请求/派生缓存，
避免四组长期保留多份完整索引。失败组的输入副本保留供排查或继续执行；已完成组 resume 时核对原索引和完成回执后直接跳过，无需再次复制。
历史索引属于原建库协议，复用记录保存它的来源与文件摘要；新评测代码不会把它冒充成当前代码重新构建的索引。

默认正式输出如下，四组 BindPack、独立 HippoRAG 2 与原有历史实验目录并存：

```text
outputs/3multi_hop_datasets_results_10_5/
├── shared_indexes/<dataset>/             已有索引，复用时只读
├── cases/<dataset>/
│   ├── <历史实验>/                        保留
│   ├── hippporag2_reproduce/              外部 HippoRAG 2 的独立结果与缓存
│   └── <full|wo_route|wo_plan|wo_pack>/
│       ├── index/                        运行时独立输入；成功后保留本组缓存
│       ├── result.json
│       ├── run.log
│       ├── protocol.json
│       └── validated.ok
├── metadata/stage_ablations_qwen3_reuse/
│   ├── protocol.json
│   ├── method_config.json
│   ├── comparison.json
│   ├── comparison.md
│   └── completed.ok
└── logs/run_bindpack_four_cases_reuse.log
```

本批次不改写根目录的历史 `protocol.json`、`comparison.json/.md`、`completed.ok`，
不删除已有实验。遇到无法确认属于本次协议的同名 case 时拒绝覆盖。
`--resume` 核对代码、方法参数、输入、样本顺序、冻结索引和结果摘要；完成的组验证后跳过，
不重新请求 LLM。参数或代码发生变化应选择新的输出目录。

`--smoke` 默认输出到 `outputs/stage_ablations_qwen3_reuse_smoke`，
每个数据集默认评测 2 题。它仍使用完整旧 corpus 与旧索引，和新建入口默认的 tiny-corpus smoke 有区别。

查看总日志：

```bash
tail -f /root/BindPack/outputs/3multi_hop_datasets_results_10_5/logs/run_bindpack_four_cases_reuse.log
```

查看当前某一组，例如 HotpotQA Full：

```bash
tail -f /root/BindPack/outputs/3multi_hop_datasets_results_10_5/cases/hotpotqa/full/run.log
```

原评测入口 `scripts/eval_dataset.py` 也支持
`--stage_ablation full|wo_route|wo_plan|wo_pack` 和 `--index_only`。
算法参数通过 `--method_config` 读取统一方法文件；不再暴露旧组件开关。

## 验证范围

`tests/test_stage_ablation_runner.py` 用替身 subprocess 跑过主入口、一次索引构建、四组、
汇总与断点继续，检查缓存隔离、样本一致、只读 dry-run，以及配置、结果、共享/私有图漂移拒绝。
它验证编排契约，不提供模型效果结论。真实 GPU tiny 验证与旧/新 Full 对比应以单独保存的
验证报告为准；完整批次结束后读取 `metadata/stage_ablations_qwen3_reuse/comparison.md`；
批次未结束时，已完成组的 `result.json` 与 `validated.ok` 可用于逐组检查。

更换 embedding 本身会改变检索结果。重构应在相同模型、相同索引、相同样本与相同 LLM
返回下比较旧/新 Full，确认候选顺序和分数保持；不能用旧 NV2 分数直接断言新 Qwen embedding
一定升降，也不能预设三组消融必然都比 Full 差。

第二轮源码快照的 CPU、冻结 Full、真实模型四组与 resume 结果见 [2026-10-18 验证报告](三阶段重构验证_2026-10-18.md)。
当前复用入口与额外方法的验证范围见 [最终检查说明](项目最终检查_2026-10-18.md)。

## FCR、DCC 与独立指标报告

每个评测结果目录另存 `report.json`（指标汇总、分母、来源）和 `metrics_details.json`（逐题诊断）。默认 K=5/10/20/200，不改变检索与 reader。MuSiQue 的 DCC 使用原生 gold decomposition；HotpotQA/2Wiki 缺少可信依赖图时为 `null`，不能用预测 plan 代替。

已有结果无需重跑，可执行 `python scripts/report_metrics.py --cases-root outputs/3multi_hop_datasets_results_10_5/cases`。当前已经运行的旧父进程由独立 CPU watcher 补报告；后续入口在保存和校验后自动报告。报告异常不丢弃检索结果，可单独重试。完整定义、命令及不改写原协议的报告更新校验见 [新指标说明](新指标说明.md)。
