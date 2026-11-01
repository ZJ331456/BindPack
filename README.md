# BindPack

`bindpack-experiment` 是原始论文 Full 结果的复现分支。

使用已配置的 `rag`、`vllm` 环境，以及 `/root/models/Qwen3-8B` 和 `/root/models/Qwen3-Embedding-8B`。四个数据集及对应语料放在 `datasets/`。

```bash
cd /root/BindPack
bash scripts/run_bindpack.sh
```

脚本自动启动或复用本地 Qwen3-8B 服务，依次全量评测 HotpotQA、2WikiMultiHopQA、MuSiQue、HoVer，每套 1,000 题，执行检索和 Reader@5。默认复用 `outputs/4_datasets_result/shared_indexes/` 中已验证的公共索引；该目录不存在时重新建库。用 `--rebuild-index` 可强制新建索引。

```bash
# 查看进度
tail -f outputs/bindpack_full/run.log

# 中断后续跑，保留已完成题目、索引和费用记录
bash scripts/run_bindpack.sh --resume
```

结果见 `outputs/bindpack_full/RESULTS.md` 和 `cases/<dataset>/full/report.json`，包含 Recall、FCR、可计算的结构指标、QA EM/F1、HoVer Accuracy/Macro-F1，以及时间、调用量、tokens 和资源统计。完整参数：`bash scripts/run_bindpack.sh --help`。
