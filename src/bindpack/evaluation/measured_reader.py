"""Reader@5 context accounting, using the native prompt and answer parser."""
from copy import copy

from .measurement import digest


def render(rag, question, docs, dataset):
    prompt = "".join(f"Wikipedia Title: {doc}\n\n" for doc in docs)
    prompt += ("Claim: " + question + "\n" if dataset == "hover"
               else "Question: " + question + "\nThought: ")
    template = "rag_qa_" + dataset
    if not rag.prompt_template_manager.is_template_name_valid(name=template):
        if dataset == "hover":
            raise ValueError("Missing native HoVer reader prompt")
        template = "rag_qa_musique"
    return rag.prompt_template_manager.render(name=template, prompt_user=prompt)


def prepare(rag, solution, tokenizer, dataset, k=5, context_window=8192, output_tokens=2048):
    """Match baseline prefix budgeting; preserve ranking and native reader behavior."""
    def count(docs):
        return len(tokenizer.apply_chat_template(render(rag, solution.question, docs, dataset),
            tokenize=True, add_generation_prompt=True, enable_thinking=False))
    limit = context_window - output_tokens
    if count([]) > limit:
        raise ValueError("Question and fixed reader prompt exceed the input budget")
    selected, partial, dropped = [], [], []
    for rank, doc in enumerate(solution.docs[:k], 1):
        if count(selected + [doc]) <= limit:
            selected.append(doc)
            continue
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        low, high = 0, len(tokens)
        while low < high:
            middle = (low + high + 1) // 2
            value = tokenizer.decode(tokens[:middle], skip_special_tokens=False)
            if count(selected + [value]) <= limit:
                low = middle
            else:
                high = middle - 1
        value = tokenizer.decode(tokens[:low], skip_special_tokens=False)
        if value.startswith(doc.partition("\n")[0] + "\n") and value.partition("\n")[2].strip():
            selected.append(value)
            partial.append(rank)
        else:
            dropped.append(rank)
        dropped.extend(range(rank + 1, min(k, len(solution.docs)) + 1))
        break
    reader_solution = copy(solution)
    reader_solution.docs = selected
    messages = render(rag, solution.question, selected, dataset)
    return reader_solution, dict(requested_passages=k, available_passages=min(k, len(solution.docs)),
        used_passages=len(selected), partial_passage_ranks=partial, dropped_passage_ranks=dropped,
        evidence_tokens=len(tokenizer.encode("".join(f"Wikipedia Title: {d}\n\n" for d in selected), add_special_tokens=False)),
        source_text_tokens=len(tokenizer.encode("\n\n".join(selected), add_special_tokens=False)),
        prompt_tokens_local=count(selected), context_window=context_window,
        reserved_output_tokens=output_tokens, prompt_sha256=digest(messages),
        truncation_policy="preserve prefix; truncate only final passage suffix; preserve complete title",
        evidence_token_scope="same baseline rendered passage blocks, including Wikipedia Title markers; source_text_tokens excludes markers")
