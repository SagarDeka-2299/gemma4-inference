#!/usr/bin/env python
"""OpenAI-compatible, single-GPU serving fallback for the AutoQA LoRA model.

This uses the same Unsloth/Transformers stack that trained the adapter. It is
intentionally serialized: the g6.xlarge has one L4 and this keeps latency and
VRAM use predictable when vLLM cannot load this particular 4-bit checkpoint.
"""
from __future__ import annotations

import gc
import hmac
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch

import batching
import structured
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

BASE_MODEL = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
ADAPTER_DIR = Path("/opt/ml/adapters/standard")
KEYFILE = Path("/opt/ml/serve/api_key")
PLAIN_MODEL_NAME = "plain-gemma"
LORA_MODEL_NAME = "autoqa-gemma"

model: Any = None
tokenizer: Any = None


def remove_unused_towers(base: Any) -> None:
    """The adapter is text-only; freeing these towers saves nearly 1 GB of VRAM."""
    current = base
    seen: set[int] = set()
    for _ in range(3):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        for attr in (
            "vision_tower", "audio_tower", "embed_vision", "embed_audio",
            "multi_modal_projector",
        ):
            if getattr(current, attr, None) is not None:
                setattr(current, attr, None)
        current = getattr(current, "model", None)
    gc.collect()
    torch.cuda.empty_cache()


def expected_api_key() -> str:
    try:
        return KEYFILE.read_text().strip()
    except OSError as exc:
        raise RuntimeError(f"API key file unavailable: {KEYFILE}") from exc


def require_key(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not hmac.compare_digest(authorization.removeprefix("Bearer "), expected_api_key()):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, tokenizer
    from unsloth import FastModel
    from peft import PeftModel

    base, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=None,
        full_finetuning=False,
    )
    remove_unused_towers(base)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    FastModel.for_inference(model)
    yield


app = FastAPI(title="AutoQA LoRA API", lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str = LORA_MODEL_NAME
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=4096, ge=1, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # OpenAI structured-output surface. tools must reach the chat template --
    # training rendered the schema into the system turn, so omitting it puts the
    # model off-distribution and it invents key names.
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "models": [PLAIN_MODEL_NAME, LORA_MODEL_NAME]}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_key(authorization)
    return {
        "object": "list",
        "data": [
            {"id": PLAIN_MODEL_NAME, "object": "model"},
            {"id": LORA_MODEL_NAME, "object": "model"},
        ],
    }


async def _run_batch(model_name: str, temperature: float, batch: list[batching.BatchItem]):
    await batching.run_batch(model, tokenizer, PLAIN_MODEL_NAME, LORA_MODEL_NAME,
                             model_name, temperature, batch)


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_key(authorization)
    if request.model not in {PLAIN_MODEL_NAME, LORA_MODEL_NAME}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt = tokenizer.apply_chat_template(
        request.messages, tools=request.tools,
        tokenize=False, add_generation_prompt=True,
    )
    # `tokenizer` is a Gemma4Processor (multimodal wrapper) -- calling it
    # directly returns a batch-wrapped nested list ([[...]]) even for one
    # string, unlike a plain tokenizer. Use the real tokenizer underneath
    # (same _inner() unwrap structured.py already needs) to get a flat list.
    input_ids = structured._inner(tokenizer)(prompt, add_special_tokens=False)["input_ids"]

    schema = structured.schema_from_request(
        request.response_format, request.tools, request.tool_choice)

    # Requests are coalesced into a batch by (model, temperature) -- see
    # batching.py. plain-gemma and autoqa-gemma never share a physical batch
    # (the LoRA toggle is whole-model, not per-sequence); a single lock still
    # serializes actual GPU execution across every bucket, since it is the
    # same model object underneath regardless of route.
    batcher = batching.get_batcher(request.model, request.temperature, _run_batch)
    item = batching.BatchItem(
        input_ids=input_ids, max_new_tokens=request.max_tokens, schema=schema)
    result = await batcher.submit(item)

    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }
	
