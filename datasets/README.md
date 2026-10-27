# BindPack 数据集、读取与 Hop 规则

新实验默认从 `/root/BindPack/datasets` 读取标准命名的 query 与 corpus 文件；可用 `--datasets-dir` 覆盖。目录包含三个多跳问答数据集与 HoVer 事实核验数据集，各有1000条样本。主实验默认清单仍为 HotpotQA、2WikiMultiHopQA、MuSiQue，HoVer 通过 `--datasets hover` 显式选择。

Hop 规则按本目录的数据约定分别定义，**不能将所有数据集的句子数、三元组数、文档数视为同一概念**。这些字段来自测试参考标注，输入检索预算时四个多跳数据集均记为 oracle 条件；它们不保证串行推理深度，也不是模型自行预测的跳数。

## 1. 本地文件与统计

| 数据集标识 | Query 文件 | Corpus 文件 | Queries | Passages | Hop 分布 |
| --- | --- | --- | ---: | ---: | --- |
| `musique` | `musique.json` | `musique_corpus.json` | 1000 | 11656 | 2:518 / 3:316 / 4:166 |
| `2wikimultihopqa` | `2wikimultihopqa.json` | `2wikimultihopqa_corpus.json` | 1000 | 6119 | 2:765 / 4:235 |
| `hotpotqa` | `hotpotqa.json` | `hotpotqa_corpus.json` | 1000 | 9811 | 2:1000 |
| `hover` | `hover.json` | `hover_corpus.json` | 1000 | 9440 | 3:559 / 4:441 |

数量为对应 JSON 顶层 list 长度，“全量”仅表示本地1000题全部运行，并非官方完整 split。Corpus 每项保存 `title` 与 `text`；模型输入的规范文档为 `title + "\n" + text`。

## 2. 各数据集的 Hop 读取

实现位于 `src/bindpack/evaluation/dataset_adapter.py`，通过轻量入口在加载模型前校验全部样本。缺失、非法或矛盾标注会报错，不回退固定2。`hop_provenance` 保存 `policy_version`、标注字段、`oracle=true`，统计中保留原始计数及分桶情况；JSON分布键统一为字符串，如 `{"2":765,"4":235}`，避免恢复时类型变化。

### MuSiQue：有效 `question_decomposition` 的长度

- 必须有2、3或4步，每步 `question` 为非空文本。
- 本地分布为 `{2:518, 3:316, 4:166}`；与 `id` 的 hop 前缀及 `is_supporting=true` 的段落数逐条一致。
- decomposition 节点数与串行 DAG 深度不同。离线 DCC 使用独立原生依赖图，不能拿该长度直接替代图。

### 2WikiMultiHopQA：`supporting_facts` 条数的2/4分桶

- 原始条数恰为2，记为2；条数≥4，记为4。
- 原始分布为 `{2:765, 4:234, 5:1}`，唯一5条的样本归入4，得到 `{2:765, 4:235}`。
- 条数为0、1、3或格式非法时拒绝，不使用“不是2就记4”的宽松回退。
- 这是**本项目的证据条数分桶约定**。`supporting_facts` 是句子级支持标注，条数可能含同文档多句或并行分支；4桶不能解释为已验证的四步串行推理。
- 不用 `evidences`/`evidences_id` 的三元组条数或排列推断 hop/DAG。旧代码统一固定2的行为已经修复，235题会按4桶进入新预算。

### HotpotQA：`supporting_facts` 的去重文档标题数

- 本地1000题都为两个不同支持标题。入口实际读取并验证标注，不再简单返回固定2。
- 句子条数分布为 `{2:650, 3:253, 4:81, 5:13, 6:1, 7:2}`；同文档多句不能额外算 hop。
- 若不满足两个不同支持标题，按异常数据报错。当前输出值仍全是2，但来源与缺标注校验已修复。

### HoVer：原生 `num_hops`，交叉校验支持文档数

- 必须是整数2、3或4，与 `supporting_facts` 去重标题数逐条一致；不能把 bool、字符串或缺失值当跳数。
- 本地分布为 `{3:559, 4:441}`，1000题交叉校验全部通过；读取器也支持其他合法HoVer文件中的2-hop样本。
- 句子条数分布为 `{3:391, 4:432, 5:139, 6:32, 7:3, 8:1, 10:2}`，不能替代 `num_hops`。

## 3. HoVer 查询、答案与文档映射

HoVer 是事实核验任务。使用 `hover_id` 对齐，检索文本取 `claim`，目标标签取原生 `label`：`SUPPORTED` 与 `NOT_SUPPORTED` 各500题。来源字段 `hotpotqa_question`、`hotpotqa_answer` 不作为当前查询或目标；`NOT_SUPPORTED` 表示未获支持，不能一概称为已被证伪。

Reader 输出完整分类标签；报告使用 `ClassificationAccuracy`、`ClassificationMacroF1`，同时保存非法输出数与按类别的混淆统计。普通问答的 token F1 不能作为该分类任务的主要指标。HoVer 可报告文档 Recall/FCR，但没有可信 gold DAG 时 DCC 为 `null`，不能仅据 `num_hops` 编造依赖图。

HotpotQA、HoVer 从 `context` 渲染文档时使用 `"".join(sentences)`，保留原句已有空白；2Wiki 使用 `" ".join(sentences)`。不能仅按标题模糊替代正文，否则与冻结 corpus 的 passage 身份不一致。

当前HoVer有一处 gold 正文与corpus不精确匹配：0-based样本446的 `Yevgeny Kafelnikov`。该差异必须由输入检查记录；是否补入由显式 corpus 完整性策略决定。若补入，须记录数量/最终语料摘要，并让各方法使用相同语料；不能静默忽略或自动按标题算命中。

## 4. CPU 校验与独立新批次

在 `/root/BindPack` 执行，校验不加载模型：

```bash
PYTHONPATH=scripts python3 - <<'CHECK'
import json
from pathlib import Path
from eval_utils import get_benchmark_hops
from collections import Counter

base = Path("datasets")
for dataset in ("musique", "2wikimultihopqa", "hotpotqa", "hover"):
    samples = json.loads((base / f"{dataset}.json").read_text())
    corpus = json.loads((base / f"{dataset}_corpus.json").read_text())
    hops = get_benchmark_hops(samples, dataset)
    print(dataset, len(samples), len(corpus), dict(sorted(Counter(hops).items())))
CHECK

# 仅预览 HoVer 四组 BindPack 的新建索引/评测命令，不启用外部HippoRAG2
bash scripts/run_bindpack_ablations_qwen3.sh \
  --datasets hover --eval-mode rag_qa \
  --out-root outputs/hover_qwen3_annotation_hops --dry-run
```

确认后去掉 `--dry-run` 才会运行；当前后台任务保持原样，不需要为新路径/Hop 接入而中断。正式运行应先处理并披露上述HoVer corpus差异。外部HippoRAG2入口是否支持HoVer分类需独立适配，不能因为多了数据集枚举就默认宣称完成复现。

## 5. 与历史批次分开

历史 `outputs/3multi_hop_datasets_results_10_5` 冻结的数据来自 `/root/datasets`，其HotpotQA/2Wiki采用固定2，MuSiQue读取旧文件decomposition。新规则不会改写历史协议、结果或正在运行的进程。

当前新旧HotpotQA、2Wiki query文件SHA一致；MuSiQue query/corpus字节SHA不同，但完整JSON内容、顺序、问题分解及11656篇有序文档相同，差异仅为序列化格式。新批次可通过严格等价校验复用三个现有索引：旧文件仍须匹配旧manifest字节SHA，新文件匹配新批次字节SHA，完整JSON的规范摘要一致，同时保留原有模型、索引资产、有序文档与样本ID检查，并保存等价校验回执。不改写旧manifest/协议；不必仅因格式变化重建MuSiQue索引。实际内容或顺序变化仍拒绝，不能只凭样本数或hop分布相同放行。旧共享目录没有HoVer索引，HoVer仍需新建。

新Hop规则下的2Wiki实验、新文件的MuSiQue实验均需另开输出目录，不与历史组拼成同一消融表。新数据默认值不替代离线报告的冻结来源；给旧结果补指标时仍读取旧manifest指定文件。指标与实验设计详见 [指标说明](../docs/指标说明.md)、[实验图表](../docs/实验图表.md)。
