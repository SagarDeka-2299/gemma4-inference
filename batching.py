"""Lightweight dynamic batching for the AutoQA server.

Not continuous/paged-attention batching (that's what vLLM/Triton give you) --
just request coalescing: requests arriving within a short window are padded
together and run through one generate() call instead of N separate ones.

Two things this has to get right that a naive queue wouldn't:

1. `plain-gemma` and `autoqa-gemma` can never share a physical batch -- the
   LoRA adapter is a whole-model toggle (`model.disable_adapter()`), not a
   per-sequence switch. Requests are bucketed by (model, temperature) so
   sampling parameters are never silently merged either; one shared lock
   still serializes actual GPU execution across every bucket, since it is
   the same model object underneath regardless of route.

2. Different requests in the same batch can want different JSON schemas.
   lm-format-enforcer's masking function already takes a batch_id, so this
   builds one enforcer per item and dispatches on it (see structured.py).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import structured

BATCH_WINDOW_SECONDS = 0.5
MAX_BATCH_SIZE = 8

_generation_lock = asyncio.Lock()
_batchers: dict[tuple[str, float], "RouteBatcher"] = {}


@dataclass
class BatchItem:
    input_ids: list[int]
    max_new_tokens: int
    schema: Optional[dict[str, Any]]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class RouteBatcher:
    """One queue per (model, temperature) bucket."""

    def __init__(self, model_name: str, temperature: float, run_batch):
        self.model_name = model_name
        self.temperature = temperature
        self._run_batch = run_batch
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._task = asyncio.create_task(self._worker())

    async def submit(self, item: BatchItem) -> str:
        await self._queue.put(item)
        return await item.future

    async def _worker(self):
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = time.monotonic() + BATCH_WINDOW_SECONDS
            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            try:
                print(f"[batching] {self.model_name}@{self.temperature}: "
                      f"batch of {len(batch)}", flush=True)
                async with _generation_lock:
                    await self._run_batch(self.model_name, self.temperature, batch)
            except Exception as exc:  # noqa: BLE001 -- surface to every caller in the batch
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)


def get_batcher(model_name: str, temperature: float, run_batch) -> RouteBatcher:
    key = (model_name, round(temperature, 2))
    b = _batchers.get(key)
    if b is None:
        b = RouteBatcher(model_name, temperature, run_batch)
        _batchers[key] = b
    return b


async def run_batch(model, tokenizer, plain_model_name: str, lora_model_name: str,
                    model_name: str, temperature: float, batch: list[BatchItem]):
    """Pad the collected items together, run one generate() call, resolve futures.

    Runs under the caller's generation lock -- exactly one of these executes
    on the GPU at a time, across every route and bucket.
    """
    from contextlib import nullcontext

    tok = structured._inner(tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    old_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        max_len = max(len(b.input_ids) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item.input_ids)
            input_ids[i, max_len - n:] = torch.tensor(item.input_ids, dtype=torch.long)
            attention_mask[i, max_len - n:] = 1
    finally:
        tok.padding_side = old_padding_side

    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    kwargs: dict[str, Any] = {
        "max_new_tokens": max(b.max_new_tokens for b in batch),
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs["do_sample"] = False

    schemas = [b.schema for b in batch]
    if any(s is not None for s in schemas):
        kwargs["prefix_allowed_tokens_fn"] = structured.batched_prefix_allowed_tokens_fn(
            tokenizer, schemas)

    adapter_mode = nullcontext() if model_name == lora_model_name else model.disable_adapter()
    with adapter_mode, torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    padded_prompt_len = max_len
    for i, item in enumerate(batch):
        completion_ids = output[i, padded_prompt_len:]
        text = tok.decode(completion_ids, skip_special_tokens=True)
        if not item.future.done():
            item.future.set_result({
                "text": text,
                # Real (unpadded) prompt length -- left-padding added fake
                # tokens that aren't part of this item's actual prompt.
                "prompt_tokens": len(item.input_ids),
                "completion_tokens": int(completion_ids.shape[0]),
            })
	
