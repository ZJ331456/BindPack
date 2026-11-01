# 四数据集全量评测输入

默认使用本目录四套数据，每套全部 1,000 题。每套包含 `<dataset>.json` 和 `<dataset>_corpus.json`，全量表示随仓库提供的本地评测集。

| Dataset | Questions | Corpus passages（补齐前） |
|---|---:|---:|
| hotpotqa | 1000 | 9811 |
| 2wikimultihopqa | 1000 | 6119 |
| musique | 1000 | 11656 |
| hover | 1000 | 9440 |

入口会按原论文协议补齐缺少的 gold passage，并记录完整语料摘要；HoVer 的实际索引语料为 9441 篇。问题和语料内容保留原样。检索只接收问题和 benchmark hop 预算，答案与 gold 来源用于离线评测。

Hop 读取逻辑见 `src/bindpack/evaluation/dataset_adapter.py`：HotpotQA 使用支持标题数；2Wiki 将两条 supporting facts 记为 2，四条及以上记为 4；MuSiQue 使用原生 question_decomposition 长度；HoVer 使用 num_hops 并校验支持标题数。这是原论文的 oracle hop 条件，不等于模型自行预测的依赖深度。

MuSiQue 使用原生依赖标注计算 NodeCov、DCC 和 ClosureGap。其他数据集不以预测 DAG 替代缺少的 gold 依赖指标。HoVer 的回答指标为 Accuracy 与 Macro-F1。
