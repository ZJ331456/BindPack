# BindPack

BindPack: Source-Bound and Dependency-Closed Evidence Selection for Multi-Hop RAG

源码按 Route → Plan/Bind → Pack 组织。默认模型为 Qwen3-8B 与 Qwen3-Embedding-8B，BindPack 的阶段对照只有 `full`、`wo_route`、`wo_plan`、`wo_pack` 四组。复用脚本默认另外运行外部 HippoRAG 2 复现，并为五组输出检索与 QA 指标。

```bash
# 查看执行计划，不加载模型
bash scripts/run_bindpack_ablations_qwen3.sh --dry-run

# 配置的 Qwen3-8B 服务启动后验证四组
bash scripts/run_bindpack_ablations_qwen3.sh --smoke

# 正式实验
bash scripts/run_bindpack_ablations_qwen3.sh

# HoVer：验证 claim，输出 SUPPORTED / NOT_SUPPORTED 分类指标
bash scripts/run_bindpack_ablations_qwen3.sh --datasets hover --eval-mode rag_qa \
  --out-root /root/BindPack/outputs/hover_annotation_hops --dry-run
```

算法默认参数集中在 [configuration/defaults.json](src/bindpack/configuration/defaults.json)，[config.py](src/bindpack/config.py) 组合按阶段分组的配置并校验；模型资源、数据与输出编排在 [ablations_qwen3.json](configs/ablations_qwen3.json)。索引必须与 Qwen embedding、语料和建库协议匹配；新建入口自行建库，复用入口只读取通过核验的现有索引。

当前默认采用已完成四数据集 Full 的 `joint_free` 联合 Pack：`selector_mode=combined`、`joint_fixed_prefix=0`、`joint_state_cap=4096`、`joint_search_seconds=5.0`，相关性权重为0.60。通过来源绑定的闭合支持、残余软覆盖、来源替代及删除风险进行预算内选包，最终Reader仍读取5篇原文。`pack.review_mode=off`，Support / semantic veto保持关闭。

新方案Full和w/o Pack的冻结串行队列位于 `outputs/4_datasets_result/pack_nine_cases_20261027/`，复用 `outputs/4_datasets_result/shared_indexes/`。默认值调整只影响后续工作区入口，已经冻结的队列与历史结果不变；旧无核验Pack结果保留在 `full_no_review_20261026/` 作为对照。

数据默认读取本仓库的 `/root/BindPack/datasets`，支持 HotpotQA、2WikiMultiHopQA、MuSiQue 和 HoVer；默认批次仍为前三个，HoVer 用 `--datasets hover` 选择。Hop 读取标注的规则和分布见 [datasets/README.md](datasets/README.md)。2Wiki 使用支持事实数量的 2/4 桶，不代表串行依赖深度；四个多跳数据集均在协议中明确记录参考标注的使用。

公开 API：

```python
from bindpack import BindPack, BindPackConfig

cfg = BindPackConfig.from_file(stage_ablation="full", save_dir="outputs/my_run/index")
rag = BindPack(config=cfg)
```

配置通过 `cfg.candidates`、`cfg.route`、`cfg.plan`、`cfg.pack` 访问，运行对象使用 `rag.config`。旧导入包装和历史组件 flags 已删除。

新建索引入口的 `--smoke` 默认以选中问题的 context 构建小语料，只验证链路；正式实验使用相同完整 corpus。该入口默认 `retrieve` 输出检索指标，`--eval-mode rag_qa` 执行 reader 并输出 QA 指标。复用 Shell 入口默认已设置 `rag_qa` 和 `--include-hipporag`；外部复现依赖 `/root/baseline/HippoRAG/main.py`，该文件不随本仓库交付。

复用入口使用 `/root/BindPack/outputs/3multi_hop_datasets_results_10_5/shared_indexes/`，BindPack 新结果为同一根目录的 `cases/<dataset>/<full|wo_route|wo_plan|wo_pack>/`，HippoRAG 2 为 `cases/<dataset>/hippporag2_reproduce/`。旧实验目录与根目录的历史汇总保留；本批次的协议和汇总单独保存在 `metadata/stage_ablations_qwen3_reuse/`。复用入口的 `--smoke` 使用完整旧索引评测少量问题，默认输出到独立 smoke 目录。

当前在跑的历史批次已固定到 `outputs/runtime_snapshots/` 下的原代码和原输入。新 hop 规则需要新的输出目录；不能用新配置覆盖或续跑旧批次。MuSiQue 新旧文件的字节摘要不同，但完整 JSON 内容、顺序和 passage 文本一致；核对原文件摘要、完整 JSON 等价性及索引资产后，可在新批次中复用三个旧索引，协议会保存等价性回执。实际内容变化仍拒绝复用。HoVer 没有旧索引，需要单独建库。例如：

```bash
bash scripts/run_bindpack_four_cases_reuse.sh --datasets hotpotqa 2wikimultihopqa musique \
  --out-root /root/BindPack/outputs/annotation_hops_reuse --dry-run
```

查看复用实验总日志：

```bash
tail -f /root/BindPack/outputs/3multi_hop_datasets_results_10_5/logs/run_bindpack_four_cases_reuse.log
```

每组另存 `report.json` 汇总指标和 `metrics_details.json` 逐题诊断，包含 FCR 与基于独立 gold DAG 的 DCC。已有最终排名可离线补算，无需重跑：

```bash
python scripts/report_metrics.py --cases-root outputs/3multi_hop_datasets_results_10_5/cases
```

MuSiQue 可计算 DCC；HotpotQA/2Wiki/HoVer 缺少可信 gold DAG 时为 `null` 并注明原因。HoVer 额外报告分类 Accuracy、Macro-F1 及逐标签统计，其 EM/F1 使用整标签匹配。定义、来源与自动补算见 [指标说明](docs/指标说明.md)。

- [实验图表、消融设计与运行协议](docs/实验图表.md)
- [指标定义与离线报告](docs/指标说明.md)
- [Baseline 分类、论文出处与公平比较](docs/Baseline说明.md)
- [源码结构与配置](src/bindpack/README.md)
