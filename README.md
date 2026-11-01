# BindPack

源码按 Route → Plan/Bind → Pack 组织，支持 HotpotQA、2WikiMultiHopQA、MuSiQue 和 HoVer。

## 当前实验入口

```bash
bash scripts/run_final_ablation_queue.sh
```

队列按以下顺序串行执行20组，每组1000题：

1. HotpotQA 的 w/o Dependency、w/o Joint Selection；然后依次2Wiki、MuSiQue、HoVer，共8组。
2. HotpotQA 的 w/o Route、w/o Plan、w/o Pack；然后按相同数据集顺序，共12组。

本轮明确不运行 w/o Soft Coverage，不运行最终QA。所有组重新执行原生上游，只复用公共图、向量和OpenIE索引；各组应用缓存独立。指标、请求/tokens、资源、证据预算和可续跑断点全部保存。

脚本先在相同小样本上校准8/16/32题窗口和客户端LLM并行数，选实测吞吐最快者，再做五种消融的真实模型预检，通过后进入20组正式队列。组间串行，组内并行；当前vLLM服务不重启。窗口摊销时间是吞吐统计，不能当作单题独立延迟。

```bash
tail -f /root/BindPack/outputs/ablations_full_20261029/run.log
```

## 当前Full和保留资产

- 最新四套Full：`outputs/4_datasets_result/final_full_no_source_20261028/cases/<dataset>/full/`。每套1000题，原报告、逐题结果、成本及资源记录保留且未改写；QA尚未运行。
- 公共索引：`outputs/4_datasets_result/shared_indexes/<dataset>/`，使用`manifest.json`核验完整语料和模型身份。
- Full原输入和源码：`final_full_no_source_20261028/input_provenance/`、`source_snapshot/`及`provenance/`。
- 旧路径迁移：`final_full_no_source_20261028/relocation_20261029.json`；历史协议的绝对路径仍是历史来源，迁移表给出保留位置。
- 先前的紧凑机制分析：`outputs/4_datasets_result/ablation_results_20261028/`，这是指向Full分析目录的兼容链接；旧大文件已按用户要求清理。
- 新20组结果：`outputs/ablations_full_20261029/RESULTS.md`、`summary.json`、`status.json`，各组位于`mechanisms/`和`stages/`。

## 默认方法与模型

默认Full为`source_bound_eligibility=false`、`mechanism_ablation=full`、`pack_mechanism_ablation=none`。Plan使用普通中间答案抽取，保留依赖执行和答案传递；Pack采用`selector_mode=combined`、`joint_fixed_prefix=0`、`joint_state_cap=4096`，相关性权重0.60，预算5篇原文。模型声明来源不等于严格核验的证明。`pack.review_mode=off`，Support / semantic veto关闭。

LLM固定`/root/models/Qwen3-8B`，Embedding固定`/root/models/Qwen3-Embedding-8B`、batch4。默认客户端预取仍为8，新队列可显式校准到16或32；这不是改变vLLM物理序列上限。vLLM当前服务的显存比例为0.55。

参数集中在[defaults.json](src/bindpack/configuration/defaults.json)，配置入口为[config.py](src/bindpack/config.py)。模型和数据编排见[ablations_qwen3.json](configs/ablations_qwen3.json)。

```python
from bindpack import BindPack, BindPackConfig
cfg = BindPackConfig.from_file(stage_ablation="full", save_dir="outputs/my_run/index")
rag = BindPack(config=cfg)
```

配置通过`cfg.candidates`、`cfg.route`、`cfg.plan`、`cfg.pack`访问。阶段和完整机制消融分开配置；完整去依赖仍保留独立子问题规划，w/o Plan则跳过整个Plan阶段，两者不是同一组。

## 数据与指标

数据位于`/root/BindPack/datasets`。Hop策略及分布见[datasets/README.md](datasets/README.md)。现有对照沿用benchmark native hop并明确记录oracle提示；不是自主推断hop。

检索报告含Recall、FCR及可计算的NodeCov、DCC、ClosureGap，无法计算者为N/A。MuSiQue有独立gold依赖标注，其他数据集不以预测DAG冒充gold。Reader@5预算可以提前渲染，但未生成答案时QA指标必须保持N/A。HoVer若执行QA，应报告标签Accuracy和Macro-F1。

- [实验图表与运行协议](docs/实验图表.md)
- [指标定义](docs/指标说明.md)
- [Baseline说明](docs/Baseline说明.md)
- [机制消融定义](scripts/MECHANISM_ABLATIONS.md)
- [源码结构](src/bindpack/README.md)
