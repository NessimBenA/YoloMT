"""Text models behind vLLM, answering `yolomt.facts.Ask`s greedily: Qwen3.5-9B (`QWEN`) derives the labels and scores
the RL reward; phi-4 (`HELD_OUT`) is the held-out judge that selects the RL checkpoint.

`offline` runs the model in this process (`scripts/derive.py`). `taking_turns` serves both judges from `vllm serve`
processes on one GPU, one at a time, while the student trains on another GPU (`scripts/train_grpo.py`).
"""

from __future__ import annotations

import atexit
import json
import math
import os
import signal
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from yolomt.facts import Ask, Reply, Run

QWEN = "Qwen/Qwen3.5-9B"
HELD_OUT = "microsoft/phi-4"
REVISIONS = {QWEN: "c202236235762e1c871ad0ccb60c8ee5ba337b9a", HELD_OUT: "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"}
NO_THINKING: dict[str, Any] = {"enable_thinking": False}


@dataclass(frozen=True)
class Plan:
    """How to serve and ask one judge. template: its chat-template arguments; threads: requests in flight."""

    model: str
    port: int
    max_model_len: int
    memory: float
    max_num_seqs: int
    max_num_batched_tokens: int | None
    template: dict[str, Any]
    threads: int


def offline(model: str, *, max_model_len: int, memory: float, max_num_seqs: int) -> Run:
    """A `Run` over an in-process vLLM engine, the thinking mode off."""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=model, revision=REVISIONS[model], tokenizer_revision=REVISIONS[model], max_model_len=max_model_len,
              gpu_memory_utilization=memory, max_num_seqs=max_num_seqs, language_model_only=True, seed=0)

    def run(asks: list[Ask]) -> list[Reply]:
        if not asks:
            return []
        params = [SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                                 logprobs=len(a.choice) + 5 if a.choice else None,
                                 structured_outputs=StructuredOutputsParams(choice=list(a.choice)) if a.choice
                                 else StructuredOutputsParams(json=a.schema, disable_any_whitespace=True))
                  for a in asks]
        outputs = llm.chat([[{"role": "user", "content": a.prompt}] for a in asks], params, use_tqdm=False,
                           chat_template_kwargs=NO_THINKING)
        replies = []
        for a, o in zip(asks, outputs):
            first = o.outputs[0].logprobs[0] if a.choice and o.outputs[0].logprobs else {}
            probabilities = {p.decoded_token: math.exp(p.logprob) for p in first.values()
                             if p.decoded_token in a.choice}
            replies.append(Reply(o.outputs[0].text, o.outputs[0].finish_reason == "stop", probabilities))
        return replies

    return run


@dataclass(frozen=True)
class Server:
    plan: Plan
    process: subprocess.Popen[bytes]


def listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def reap(group: int) -> None:
    if subprocess.run(["pgrep", "-g", str(group)], capture_output=True).returncode == 0:
        os.killpg(group, signal.SIGTERM)


def launch(plan: Plan, *, device: str, log: str) -> Server:
    """`vllm serve` on GPU `device`; it opens its port once the model is loaded."""
    command = ["vllm", "serve", plan.model, "--revision", REVISIONS[plan.model], "--port", str(plan.port),
               "--max-model-len", str(plan.max_model_len), "--gpu-memory-utilization", str(plan.memory),
               "--max-num-seqs", str(plan.max_num_seqs), "--language-model-only", "--seed", "0"]
    if plan.max_num_batched_tokens is not None:
        command += ["--max-num-batched-tokens", str(plan.max_num_batched_tokens)]
    process = subprocess.Popen(command, stdout=open(log, "ab"), stderr=subprocess.STDOUT, start_new_session=True,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": device})
    # the server runs in its own process group, which outlives this process unless stopped
    atexit.register(reap, process.pid)
    while not listening(plan.port):
        assert process.poll() is None, f"vllm serve {plan.model} exited; see {log}"
        time.sleep(10)
    return Server(plan, process)


def stop(server: Server) -> None:
    """Stop the server's whole process group: vLLM frees the GPU only once every process of it has exited."""
    os.killpg(server.process.pid, signal.SIGTERM)
    server.process.wait()
    while subprocess.run(["pgrep", "-g", str(server.process.pid)], capture_output=True).returncode == 0:
        time.sleep(1)


def ask(server: Server, asks: list[Ask]) -> list[Reply]:
    """One chat request per ask; a choice ask also returns the first token's probability of each choice."""
    url = f"http://127.0.0.1:{server.plan.port}/v1/chat/completions"

    def one(a: Ask) -> Reply:
        body: dict[str, Any] = {
            "model": server.plan.model, "messages": [{"role": "user", "content": a.prompt}], "temperature": 0.0,
            "max_tokens": a.max_tokens, "chat_template_kwargs": server.plan.template,
            "structured_outputs": {"choice": list(a.choice)} if a.choice else {"json": a.schema,
                                                                               "disable_any_whitespace": True},
        }
        if a.choice:
            body |= {"logprobs": True, "top_logprobs": len(a.choice) + 5}
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.loads(response.read())["choices"][0]
        probabilities = {}
        if a.choice:
            first = choice["logprobs"]["content"][0]["top_logprobs"]
            probabilities = {t["token"]: math.exp(t["logprob"]) for t in first if t["token"] in a.choice}
        return Reply(choice["message"]["content"], choice["finish_reason"] == "stop", probabilities)

    with ThreadPoolExecutor(max_workers=server.plan.threads) as pool:
        return list(pool.map(one, asks))


def taking_turns(plans: dict[str, Plan], *, device: str, logs: str) -> tuple[dict[str, Run], Callable[[], None]]:
    """A `Run` per judge on one GPU too small for both: asking a judge whose server is down stops the other's and
    starts its own. The second value stops whichever server is up."""
    up: dict[str, Server] = {}

    def stop_up() -> None:
        for server in up.values():
            stop(server)
        up.clear()

    def run_of(name: str) -> Run:
        def run(asks: list[Ask]) -> list[Reply]:
            if not asks:
                return []
            if name not in up:
                stop_up()
                up[name] = launch(plans[name], device=device, log=f"{logs}/{name}.log")
            return ask(up[name], asks)

        return run

    return {name: run_of(name) for name in plans}, stop_up
