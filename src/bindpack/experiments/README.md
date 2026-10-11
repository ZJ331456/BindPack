# Mechanism ablations

This package contains isolated experimental controls. Production BindPack does
not import these adapters or change its default method.

Run the three datasets against their frozen NV2 indexes:

```bash
bash scripts/run_bindpack_ablations_nv2.sh --smoke --full-after-smoke
```

The two-query checks run every condition, retain a compact validation receipt,
remove their temporary outputs, then admit the full retrieval evaluation.
Each dataset has 21 default conditions:

- Full-method capture, no plan pruning, string admission and relation admission.
- Ordinary QD and iterative retrieval at 50% and 100% token envelopes.
- Six frozen source-qualification × coverage-reward conditions.
- Relevance, lexical MMR and ordinary greedy node-coverage references.
- Four frozen support-guard × semantic-veto conditions.

All runs use NV2 batch4, LLM concurrency8, benchmark hops and unchanged output
allowances. Snapshots contain actual pre-package scores, source texts, plans,
bindings and proofs. No gold passage enters a selector.

Important scopes:

- Budget controls match resource **caps**, not forced actual expenditure. The
  common initial retrieval is replayed; incremental timing is labelled as such.
- Frozen flat/closed controls share the existing closed package proposal
  family. Only coverage rewards change; this does not remove every ancestor
  feasibility constraint.
- Frozen proof qualification filters already available proofs. Separate online
  admission controls test changes in binding and later retrieval trajectories.
- Review controls share existing proposals and reviewer responses from `ours`.
- Local grammar acceptance is not universal entailment. Human annotation
  exports retain blank labels; local queries remain development queries.

Results: `outputs/3multi_hop_datasets_results_with_nv2_10_8/cases/*/abla_*`.
Summary, query-paired bootstrap and factorial interactions:
`metadata/bindpack_ablations/`. No new index construction or QA is included.
