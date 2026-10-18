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

# 复用 10_5 的三个 Qwen 索引，执行四组 BindPack + HippoRAG 2（含 QA）
bash scripts/run_bindpack_four_cases_reuse.sh --dry-run
bash scripts/run_bindpack_four_cases_reuse.sh
```

算法默认参数集中在 [configuration/defaults.json](src/bindpack/configuration/defaults.json)，[config.py](src/bindpack/config.py) 组合按阶段分组的配置并校验；模型资源、数据与输出编排在 [ablations_qwen3.json](configs/ablations_qwen3.json)。索引必须与 Qwen embedding、语料和建库协议匹配；新建入口自行建库，复用入口只读取通过核验的现有索引。

公开 API：

```python
from bindpack import BindPack, BindPackConfig

cfg = BindPackConfig.from_file(stage_ablation="full", save_dir="outputs/my_run/index")
rag = BindPack(config=cfg)
```

配置通过 `cfg.candidates`、`cfg.route`、`cfg.plan`、`cfg.pack` 访问，运行对象使用 `rag.config`。旧导入包装和历史组件 flags 已删除。

新建索引入口的 `--smoke` 默认以选中问题的 context 构建小语料，只验证链路；正式实验使用相同完整 corpus。该入口默认 `retrieve` 输出检索指标，`--eval-mode rag_qa` 执行 reader 并输出 QA 指标。复用 Shell 入口默认已设置 `rag_qa` 和 `--include-hipporag`；外部复现依赖 `/root/baseline/HippoRAG/main.py`，该文件不随本仓库交付。

复用入口使用 `/root/BindPack/outputs/3multi_hop_datasets_results_10_5/shared_indexes/`，BindPack 新结果为同一根目录的 `cases/<dataset>/<full|wo_route|wo_plan|wo_pack>/`，HippoRAG 2 为 `cases/<dataset>/hippporag2_reproduce/`。旧实验目录与根目录的历史汇总保留；本批次的协议和汇总单独保存在 `metadata/stage_ablations_qwen3_reuse/`。复用入口的 `--smoke` 使用完整旧索引评测少量问题，默认输出到独立 smoke 目录。

查看复用实验总日志：

```bash
tail -f /root/BindPack/outputs/3multi_hop_datasets_results_10_5/logs/run_bindpack_four_cases_reuse.log
```

每组另存 `report.json` 汇总指标和 `metrics_details.json` 逐题诊断，包含 FCR 与基于独立 gold DAG 的 DCC。已有最终排名可离线补算，无需重跑：

```bash
python scripts/report_metrics.py --cases-root outputs/3multi_hop_datasets_results_10_5/cases
```

MuSiQue 可计算 DCC；HotpotQA/2Wiki 缺少可信 gold DAG 时为 `null` 并注明原因。定义、来源与自动补算见 [指标说明](docs/新指标说明.md)。

- [四组实验运行与服务准备](docs/四组实验-Qwen3.md)
- [源码结构与配置](src/bindpack/README.md)
- [模块机制、消融定义与边界](docs/模块组合-消融说明.md)

本轮验证见 [第二轮重构验证（2026-10-18）](docs/三阶段重构验证_2026-10-18.md)。
最后检查与旧 Qwen 索引复用范围见 [最终检查说明](docs/项目最终检查_2026-10-18.md)。
