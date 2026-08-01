"""Grammar-constrained JSON decoding for the AutoQA server.

Gives the same guarantee as OpenAI structured outputs: the sampler is masked at
every step so only tokens that can still lead to a schema-valid document are
allowed. Invalid JSON becomes impossible rather than unlikely.

lm-format-enforcer's own transformers integration is unusable here -- it does
`from transformers.tokenization_utils import PreTrainedTokenizerBase`, a path
that moved in transformers 5.x, and its `except ImportError` reports the
misleading "transformers is not installed". So this module talks to the core
API directly and builds the tokenizer data itself.
"""
from __future__ import annotations

import functools
from typing import Any, List, Tuple

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData

_TOKENIZER_DATA = None


def _build_regular_tokens_list(tokenizer, vocab_size: int) -> List[Tuple[int, str, bool]]:
    token_0 = tokenizer.encode("0")[-1]
    out = []
    specials = set(tokenizer.all_special_ids or [])
    for idx in range(vocab_size):
        if idx in specials:
            continue
        # Prepend a known token and drop its first char, so a leading space is
        # preserved -- that is how word-start tokens are detected.
        after_0 = tokenizer.decode([token_0, idx])[1:]
        plain = tokenizer.decode([idx])
        out.append((idx, after_0, len(after_0) > len(plain)))
    return out


def _decode(tokenizer, tokens: List[int]) -> str:
    return tokenizer.decode(tokens).rstrip("�")


def _inner(tok):
    """Gemma 4 loads as a Gemma4Processor (multimodal wrapper) with no __len__
    and no encode/decode of its own. The real tokenizer hangs off it."""
    return getattr(tok, "tokenizer", tok)


def tokenizer_data(tokenizer) -> TokenEnforcerTokenizerData:
    """Built once (~15s for a 262k vocab) and reused for every request."""
    global _TOKENIZER_DATA
    if _TOKENIZER_DATA is None:
        tokenizer = _inner(tokenizer)
        vocab_size = len(tokenizer)
        _TOKENIZER_DATA = TokenEnforcerTokenizerData(
            _build_regular_tokens_list(tokenizer, vocab_size),
            functools.partial(_decode, tokenizer),
            tokenizer.eos_token_id,
            False,
            vocab_size,
        )
    return _TOKENIZER_DATA


def prefix_allowed_tokens_fn(tokenizer, schema: dict[str, Any]):
    """Returns a callable for HF `generate(prefix_allowed_tokens_fn=...)`.

    At each step it returns only the token ids that keep the output on a path to
    a schema-valid JSON document.
    """
    enforcer = TokenEnforcer(tokenizer_data(_inner(tokenizer)), JsonSchemaParser(schema))

    def fn(batch_id: int, sent) -> List[int]:
        # get_allowed_tokens returns lm-format-enforcer's TokenList, which is
        # neither len()-able nor iterable. With use_bitmask=False the plain list
        # of ids lives on .allowed_tokens, which is what transformers'
        # PrefixConstrainedLogitsProcessor expects.
        res = enforcer.get_allowed_tokens(sent.tolist())
        return getattr(res, "allowed_tokens", res)

    return fn


_FULL_VOCAB: List[int] | None = None


def batched_prefix_allowed_tokens_fn(tokenizer, schemas: list[dict[str, Any] | None]):
    """Per-batch-index grammar dispatch, for dynamic-batched generation.

    `schemas[i]` is the JSON schema for batch row i, or None for an
    unconstrained row. Each distinct schema gets its own TokenEnforcer;
    correctness doesn't depend on rows sharing a schema being deduplicated --
    the enforcer keys its internal state purely on that row's own token
    history (see get_allowed_tokens), so one enforcer per row is always
    correct, just slightly more memory than necessary when rows share a
    schema. Rows are typically <=8 (MAX_BATCH_SIZE), so this doesn't matter.

    Left-padding (added by the batch collator so rows align) is transparent
    to the enforcer: it only requires each call's token sequence to be the
    previous call's plus exactly one new token, regardless of what the
    prefix contains.
    """
    global _FULL_VOCAB
    inner = _inner(tokenizer)
    if _FULL_VOCAB is None:
        _FULL_VOCAB = list(range(len(inner)))

    enforcers: dict[int, TokenEnforcer] = {}
    for i, schema in enumerate(schemas):
        if schema is not None:
            enforcers[i] = TokenEnforcer(tokenizer_data(inner), JsonSchemaParser(schema))

    def fn(batch_id: int, sent) -> List[int]:
        enforcer = enforcers.get(batch_id)
        if enforcer is None:
            return _FULL_VOCAB
        res = enforcer.get_allowed_tokens(sent.tolist())
        return getattr(res, "allowed_tokens", res)

    return fn


def schema_from_request(response_format: dict | None, tools: list | None,
                        tool_choice: Any = None) -> dict | None:
    """Extract the JSON schema a request is asking to be constrained to.

    Supports both shapes the OpenAI API uses:
      * response_format={"type":"json_schema","json_schema":{"schema":{...}}}
      * tools=[{"type":"function","function":{"parameters":{...}}}] with
        tool_choice naming one of them (or a single tool, which we treat as
        forced -- that is what the QA call sites mean).
    """
    if response_format:
        if response_format.get("type") == "json_schema":
            js = response_format.get("json_schema") or {}
            schema = js.get("schema")
            if schema:
                return schema
        # {"type":"json_object"} -> any object; a permissive schema still
        # guarantees well-formed JSON.
        if response_format.get("type") == "json_object":
            return {"type": "object"}

    if tools:
        wanted = None
        if isinstance(tool_choice, dict):
            wanted = (tool_choice.get("function") or {}).get("name")
        for t in tools:
            fn = t.get("function", t)
            if wanted is None or fn.get("name") == wanted:
                params = fn.get("parameters")
                if params:
                    return params
    return None
	
	
