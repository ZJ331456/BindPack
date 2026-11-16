# BindPack

源码按 Route → Plan/Bind → Pack 组织，支持 HotpotQA、2WikiMultiHopQA、MuSiQue 和 HoVer。

## 本机 CUDA 12.4 环境

使用 conda 环境 `bindpack`，本地模型位于 `/root/models`。CUDA 12.4 依赖固定在
`requirements-cu124.txt`（PyTorch 2.6.0、vLLM 0.8.5.post1、Transformers 4.51.3）。

```bash
conda activate bindpack
cd /root/BindPack
python -m pip install -r requirements-cu124.txt
bash scripts/start_llm_qwen3_nohup.sh
# 等 http://127.0.0.1:8035/health 就绪后，在空输出目录运行真实模型的小样本检查：
bash scripts/run_bindpack_measured.sh --dataset hotpotqa --smoke \
  --eval-mode rag_qa --query-window 1 --output outputs/gpu_smoke
```

服务日志位于 `outputs/services/qwen3_reader_055.log`。小样本检查使用缩小的语料，
用于验证索引、检索和问答链路，不代表完整数据集的评测结果。

## 当前实验入口

```bash
bash scripts/run_redesigned_ablations.sh
```

本轮按 HotpotQA → 2Wiki → MuSiQue → HoVer 执行；每个数据集1000题，包含以下6组，共24组检索与QA结果：

1. Full
2. w/o Source Binding
3. Reference (S₀)
4. Flat Support
5. Singleton Search
6. w/o Parent Answer Propagation

组间严格串行：当前组的1000题检索、QA和报告全部完成并通过校验后，才开始下一组；一个数据集六组完成后再开始下一个数据集。组内的客户端检索窗口和Reader并发在8、16、32档实测选择，最终配置冻结在输出目录的 `concurrency_profile.json`。

四个Pack消融与Full复用同一次Route和Plan执行记录，固定候选、DAG、答案抽取和参考集合。父答案传播消融复用Full的Route与DAG，将父答案替换为父问题描述，并重新执行子问题检索与答案抽取；保留来源绑定与完整Pack。本轮不运行阶段消融，原阶段消融定义保留。

全部使用 `outputs/shared_indexes/<dataset>/` 的现有图、向量和OpenIE索引，禁止重构索引。最终Reader统一使用5篇原文、8192上下文和2048输出预算；报告检索、QA、MuSiQue独立gold图指标及逐题trace。预检与正式结果分别保存，真实请求和可续跑断点保留。窗口摊销时间是吞吐统计，不能当作单题独立延迟。

```bash
tail -f /root/BindPack/outputs/ablations_redesigned_20261108/run.log
```

## 当前结果和共享索引

- 六组机制消融已完成：24组、每组1000题。[检索与QA结果汇总](docs/消融实验结果_20261108.md)；三线表的PNG/PDF/SVG、LaTeX与CSV位于 `outputs/ablations_redesigned_20261108/publication_tables/`。

- 本轮输出：`outputs/ablations_redesigned_20261108/`；包含汇总结果、运行状态、逐题结果、冻结上游和日志。
- 公共索引：`outputs/shared_indexes/<dataset>/`，四个数据集分别复用对应的完整语料索引。
- 索引原文件保留，运行使用独立工作目录；不继承共享索引中的历史应用响应缓存。
- 并发校准汇总、代码与配置、索引校验回执保存在本轮正式目录；临时校准在 `/tmp` 执行，不混入正式结果。

## 默认方法与模型

默认Full为`source_bound_eligibility=false`、`mechanism_ablation=full`、`pack_mechanism_ablation=none`。Plan使用普通中间答案抽取，保留依赖执行和答案传递；Pack采用`selector_mode=combined`、`joint_fixed_prefix=0`、`joint_state_cap=4096`，相关性权重0.60，预算5篇原文。模型声明来源不等于严格核验的证明。`pack.review_mode=off`，Support / semantic veto关闭。

LLM固定`/root/models/Qwen3-8B`，Embedding固定`/root/models/Qwen3-Embedding-8B`、batch4。客户端并行只改变吞吐，不更改算法。客户端请求数和服务端实际同时处理的序列数分别记录；vLLM当前服务的显存比例为0.55。

参数集中在[defaults.json](src/bindpack/configuration/defaults.json)，配置入口为[config.py](src/bindpack/config.py)。本轮六组模型和数据编排见[redesigned_ablations_qwen3.json](configs/redesigned_ablations_qwen3.json)。旧阶段实验配置 `configs/ablations_qwen3.json` 保留原四阶段与检索模式。

```python
from bindpack import BindPack, BindPackConfig
cfg = BindPackConfig.from_file(stage_ablation="full", save_dir="outputs/my_run/index")
rag = BindPack(config=cfg)
```

配置通过`cfg.candidates`、`cfg.route`、`cfg.plan`、`cfg.pack`访问。每次只启用一个机制消融；阶段消融与机制消融分开配置。旧 `w/o Dependency` 不再纳入本轮实验。

## 数据与指标

数据位于`/root/BindPack/datasets`。Hop策略及分布见[datasets/README.md](datasets/README.md)。现有对照沿用benchmark native hop并明确记录oracle提示；不是自主推断hop。

检索报告含Recall、FCR及可计算的NodeCov、DCC、ClosureGap，无法计算者为N/A。MuSiQue有独立gold依赖标注，其他数据集不以预测DAG冒充gold。Reader@5预算可以提前渲染，但未生成答案时QA指标必须保持N/A。HoVer若执行QA，应报告标签Accuracy和Macro-F1。

- [实验图表与运行协议](docs/实验图表.md)
- [指标定义](docs/指标说明.md)
- [Baseline说明](docs/Baseline说明.md)
- [机制消融定义](scripts/MECHANISM_ABLATIONS.md)
- [源码结构](src/bindpack/README.md)
