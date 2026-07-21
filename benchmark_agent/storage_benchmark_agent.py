"""
Storage Benchmark AI Agent — fio vs vdbench
============================================
An AI agent that takes a natural language prompt describing a storage
workload and autonomously:
  1. Translates it into fio and vdbench configs
  2. Runs both tools against your target
  3. Parses the output for IOPS and throughput
  4. Compares results side by side

Supports: local Linux, AWS EC2/EKS, on-prem NFS/SAN

Set your Gemini key:
    export GOOGLE_API_KEY="AIza..."

Dry-run mode (no fio/vdbench/Gemini needed):
    export DRY_RUN=true
    python storage_benchmark_agent.py

Author : vytasta — https://vytasta.github.io/uberzunn/benchmark_agent
License: MIT
"""

import os
import json
import time
import random
import subprocess
import re
import functools
from typing import TypedDict
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────
# DRY-RUN FLAG
# Set DRY_RUN=true to simulate the full agent without running
# fio/vdbench or consuming Gemini tokens. Great for CI and demos.
# ─────────────────────────────────────────────────────────────

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

if DRY_RUN:
    print("=" * 65)
    print("DRY-RUN MODE — no real tools or LLM calls will be made")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# RATE LIMIT PROTECTION — throttle only
# A fixed delay between every LLM call keeps us safely under
# Gemini free tier (15 RPM). No retry loops needed when you
# stay under the limit in the first place.
# ─────────────────────────────────────────────────────────────

THROTTLE_SECONDS = 5        # 5s gap = max 12 RPM — safely under 15 RPM limit
_last_call_time  = 0.0


def throttle():
    """Enforce minimum gap between LLM calls."""
    global _last_call_time
    if DRY_RUN:
        return                  # no throttle needed in dry-run
    elapsed = time.time() - _last_call_time
    if elapsed < THROTTLE_SECONDS:
        wait = THROTTLE_SECONDS - elapsed
        print(f"  [throttle] waiting {wait:.1f}s to stay under rate limit...")
        time.sleep(wait)
    _last_call_time = time.time()


def rate_limited(fn):
    """Decorator: throttle before every LLM call."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        throttle()
        return fn(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────

class BenchmarkState(TypedDict):
    messages:            list   # conversation history
    workload_params:     dict   # parsed workload parameters
    fio_config:          str    # generated fio job file content
    vdbench_config:      str    # generated vdbench config content
    fio_raw_output:      str    # raw stdout from fio
    vdbench_raw_output:  str    # raw stdout from vdbench
    fio_results:         dict   # parsed fio IOPS + throughput
    vdbench_results:     dict   # parsed vdbench IOPS + throughput
    comparison:          str    # side-by-side comparison summary
    error:               str    # last error if any
    final_answer:        str    # clean output for the user


# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────

@tool
def generate_fio_config(
    target: str,
    read_pct: int = 100,
    write_pct: int = 0,
    block_size: str = "64k",
    iodepth: int = 32,
    numjobs: int = 4,
    runtime_seconds: int = 60,
    io_engine: str = "libaio",
    direct: int = 1,
    target_iops: int = 0,
) -> str:
    """
    Generate a fio job file for a storage benchmark workload.

    Args:
        target:           path to test file or block device e.g. /mnt/nvme or /dev/sdb
        read_pct:         percentage of reads (0-100)
        write_pct:        percentage of writes (0-100)
        block_size:       I/O block size e.g. 4k, 64k, 128k, 1m
        iodepth:          queue depth / concurrency per job
        numjobs:          number of parallel threads/jobs
        runtime_seconds:  how long to run in seconds
        io_engine:        libaio (Linux), windowsaio, posixaio
        direct:           1 = bypass page cache (recommended)
        target_iops:      optional IOPS rate limit (0 = unlimited)

    Returns:
        fio job file content as a string.
    """
    if DRY_RUN:
        return f"[DRY-RUN] fio config for {target} | {read_pct}/{write_pct} R/W | bs={block_size} | iodepth={iodepth} | jobs={numjobs} | {runtime_seconds}s"

    if read_pct == 100:
        rw_mode = "read"
    elif write_pct == 100:
        rw_mode = "write"
    else:
        rw_mode = "randrw"

    rwmixread_line = f"rwmixread={read_pct}\n" if rw_mode == "randrw" else ""
    rate_iops_line = f"rate_iops={target_iops}\n" if target_iops > 0 else ""

    return f"""[global]
ioengine={io_engine}
direct={direct}
runtime={runtime_seconds}
time_based=1
group_reporting=1
output-format=json

[benchmark_job]
filename={target}
rw={rw_mode}
{rwmixread_line}bs={block_size}
iodepth={iodepth}
numjobs={numjobs}
{rate_iops_line}size=10g
""".strip()


@tool
def generate_vdbench_config(
    target: str,
    read_pct: int = 100,
    block_size: str = "64k",
    threads: int = 8,
    runtime_seconds: int = 60,
    target_iops: int = 0,
) -> str:
    """
    Generate a vdbench parameter file for a storage benchmark workload.

    Args:
        target:           path to test file or block device
        read_pct:         percentage of reads (0-100)
        block_size:       I/O block size e.g. 4k, 64k, 128k, 1m
        threads:          number of concurrent I/O threads
        runtime_seconds:  how long to run in seconds
        target_iops:      optional IOPS target (0 = max throughput)

    Returns:
        vdbench parameter file content as a string.
    """
    if DRY_RUN:
        return f"[DRY-RUN] vdbench config for {target} | rdpct={read_pct} | xfersize={block_size} | threads={threads} | {runtime_seconds}s"

    iorate = str(target_iops) if target_iops > 0 else "max"

    return f"""# vdbench parameter file — generated by Storage Benchmark AI Agent
sd=sd1,lun={target},size=10g,openflags=o_direct
wd=wd1,sd=sd1,rdpct={read_pct},seekpct=100,xfersize={block_size},threads={threads}
rd=run1,wd=wd1,iorate={iorate},elapsed={runtime_seconds},interval=1,warmup=10
""".strip()


@tool
def run_fio(config_content: str, config_path: str = "/tmp/fio_benchmark.fio") -> str:
    """
    Write fio config to disk and execute it. Returns JSON output.

    Args:
        config_content: fio job file content from generate_fio_config
        config_path:    temp path to write the config

    Returns:
        Raw fio JSON output string.
    """
    if DRY_RUN or config_content.startswith("[DRY-RUN]"):
        return json.dumps({
            "jobs": [{
                "read":  {"iops": 48230, "bw": 3120000, "lat_ns": {"mean": 650000}},
                "write": {"iops": 20670, "bw": 1340000, "lat_ns": {"mean": 1520000}},
            }]
        })

    try:
        with open(config_path, "w") as f:
            f.write(config_content)
        print("  [fio] running benchmark...")
        result = subprocess.run(
            ["fio", config_path, "--output-format=json"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return json.dumps({"error": result.stderr})
        return result.stdout
    except FileNotFoundError:
        return json.dumps({"error": "fio not found. Install: sudo apt install fio"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "fio timed out after 5 minutes"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_vdbench(config_content: str, config_path: str = "/tmp/vdbench_benchmark.txt") -> str:
    """
    Write vdbench config to disk and execute it. Returns output.

    Args:
        config_content: vdbench parameter file from generate_vdbench_config
        config_path:    temp path to write the config

    Returns:
        Raw vdbench output string.
    """
    if DRY_RUN or config_content.startswith("[DRY-RUN]"):
        return (
            "vdbench output simulation\n"
            "interval   i/o    MB/sec  bytes   read    resp    read    write\n"
            "avg        67540  4221.3  65536   70.0%   0.943   0.612   1.820\n"
        )

    try:
        with open(config_path, "w") as f:
            f.write(config_content)
        print("  [vdbench] running benchmark...")
        result = subprocess.run(
            ["vdbench", "-f", config_path, "-o", "/tmp/vdbench_output"],
            capture_output=True, text=True, timeout=300,
        )
        return result.stdout + result.stderr
    except FileNotFoundError:
        return json.dumps({"error": "vdbench not found. Download from oracle.com/downloads/server-storage/vdbench-downloads.html"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "vdbench timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def parse_fio_output(raw_output: str) -> str:
    """
    Parse fio JSON output and extract IOPS and throughput.

    Args:
        raw_output: raw stdout from fio

    Returns:
        JSON string with read_iops, write_iops, read_bw_mbps, write_bw_mbps, latency.
    """
    try:
        data = json.loads(raw_output)
        job  = data.get("jobs", [{}])[0]
        read  = job.get("read",  {})
        write = job.get("write", {})
        return json.dumps({
            "tool":          "fio",
            "read_iops":     round(read.get("iops", 0)),
            "write_iops":    round(write.get("iops", 0)),
            "total_iops":    round(read.get("iops", 0) + write.get("iops", 0)),
            "read_bw_mbps":  round(read.get("bw", 0) / 1024, 2),
            "write_bw_mbps": round(write.get("bw", 0) / 1024, 2),
            "total_bw_mbps": round((read.get("bw", 0) + write.get("bw", 0)) / 1024, 2),
            "read_lat_us":   round(read.get("lat_ns", {}).get("mean", 0) / 1000, 2),
            "write_lat_us":  round(write.get("lat_ns", {}).get("mean", 0) / 1000, 2),
        })
    except Exception as e:
        return json.dumps({"tool": "fio", "error": str(e), "raw_snippet": raw_output[:300]})


@tool
def parse_vdbench_output(raw_output: str) -> str:
    """
    Parse vdbench output and extract IOPS and throughput.

    Args:
        raw_output: raw stdout from vdbench

    Returns:
        JSON string with read_iops, write_iops, total_bw_mbps, avg_resp_ms.
    """
    try:
        for line in raw_output.splitlines():
            if line.strip().startswith("avg"):
                parts        = line.split()
                total_iops   = float(parts[1].replace(",", ""))
                total_bw     = float(parts[2].replace(",", ""))
                read_pct_val = float(parts[4].replace("%", "")) / 100
                return json.dumps({
                    "tool":          "vdbench",
                    "total_iops":    round(total_iops),
                    "read_iops":     round(total_iops * read_pct_val),
                    "write_iops":    round(total_iops * (1 - read_pct_val)),
                    "total_bw_mbps": round(total_bw, 2),
                    "read_bw_mbps":  round(total_bw * read_pct_val, 2),
                    "write_bw_mbps": round(total_bw * (1 - read_pct_val), 2),
                    "avg_resp_ms":   float(parts[5]) if len(parts) > 5 else None,
                })
        return json.dumps({"tool": "vdbench", "error": "no avg line found", "raw_snippet": raw_output[:300]})
    except Exception as e:
        return json.dumps({"tool": "vdbench", "error": str(e)})


# ─────────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    generate_fio_config,
    generate_vdbench_config,
    run_fio,
    run_vdbench,
    parse_fio_output,
    parse_vdbench_output,
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

SYSTEM_PROMPT = """You are an expert storage performance engineer AI agent.

Your job is to:
1. Parse the user's workload prompt for these parameters:
   - target path or device (e.g. /mnt/nvme, /dev/sdb, /mnt/nfs)
   - read/write ratio (e.g. 70/30, 100% read, sequential write)
   - block size (e.g. 4k, 64k, 128k, 1m)
   - IOPS target (optional, 0 = max)
   - concurrency / iodepth / queue depth
   - number of threads / jobs
   - duration in seconds

2. Call generate_fio_config with the parsed parameters.
3. Call generate_vdbench_config with the same parameters.
4. Call run_fio with the fio config content.
5. Call run_vdbench with the vdbench config content.
6. Call parse_fio_output with the fio raw output.
7. Call parse_vdbench_output with the vdbench raw output.

Call ONE tool at a time. Always do fio steps first, then vdbench.

Default values if not specified:
  block_size: 64k | iodepth: 32 | numjobs/threads: 8
  runtime: 60s | read_pct: 100 | target_iops: 0 (unlimited)
"""


def _make_llm():
    """Create LLM — returns a stub in dry-run mode."""
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    ).bind_tools(ALL_TOOLS)


def _make_llm_plain():
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


LLM       = _make_llm()
LLM_PLAIN = _make_llm_plain()


# ─────────────────────────────────────────────────────────────
# DRY-RUN SIMULATION — scripted tool sequence, no LLM needed
# ─────────────────────────────────────────────────────────────

DRY_RUN_SEQUENCE = [
    ("generate_fio_config",    {"target": "/mnt/test", "read_pct": 70, "write_pct": 30, "block_size": "128k", "iodepth": 64, "numjobs": 32, "runtime_seconds": 60}),
    ("generate_vdbench_config",{"target": "/mnt/test", "read_pct": 70, "block_size": "128k", "threads": 32, "runtime_seconds": 60}),
    ("run_fio",                {"config_content": "[DRY-RUN] fio config"}),
    ("run_vdbench",            {"config_content": "[DRY-RUN] vdbench config"}),
    ("parse_fio_output",       {"raw_output": json.dumps({"jobs": [{"read": {"iops": 48230, "bw": 3120000, "lat_ns": {"mean": 650000}}, "write": {"iops": 20670, "bw": 1340000, "lat_ns": {"mean": 1520000}}}]})}),
    ("parse_vdbench_output",   {"raw_output": "interval   i/o    MB/sec  bytes   read    resp\navg        67540  4221.3  65536   70.0%   0.943\n"}),
]

_dry_run_step = 0


class _DryRunMessage:
    """Mimics an AIMessage with tool_calls for dry-run mode."""
    def __init__(self, tool_name, tool_args, call_id):
        self.tool_calls = [{"name": tool_name, "args": tool_args, "id": call_id}]
        self.content    = f"[DRY-RUN] calling {tool_name}"


# ─────────────────────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────────────────────

@rate_limited
def _llm_invoke(messages):
    return LLM.invoke(messages)


def planner_node(state: BenchmarkState) -> dict:
    """LLM (or dry-run stub) decides the next tool to call."""
    global _dry_run_step

    if DRY_RUN:
        if _dry_run_step >= len(DRY_RUN_SEQUENCE):
            # All tools done — return plain message to trigger compare
            class _Done:
                tool_calls = []
                content    = "[DRY-RUN] all tools complete"
            return {"messages": state["messages"] + [_Done()]}

        tool_name, tool_args = DRY_RUN_SEQUENCE[_dry_run_step]
        _dry_run_step += 1
        print(f"  [dry-run] step {_dry_run_step}/{len(DRY_RUN_SEQUENCE)}: {tool_name}")
        time.sleep(0.3)   # small visual delay so output is readable
        msg = _DryRunMessage(tool_name, tool_args, f"dry-call-{_dry_run_step}")
        return {"messages": state["messages"] + [msg]}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
    response = _llm_invoke(messages)
    return {"messages": state["messages"] + [response]}


def tool_executor_node(state: BenchmarkState) -> dict:
    """Execute whatever tool the planner requested."""
    last    = state["messages"][-1]
    updates = {}
    new_messages = list(state["messages"])

    for tc in last.tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        result  = tool_fn.invoke(tc["args"]) if tool_fn else json.dumps({"error": f"Unknown tool: {tc['name']}"})
        new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        if tc["name"] == "generate_fio_config":
            updates["fio_config"]        = result
        elif tc["name"] == "generate_vdbench_config":
            updates["vdbench_config"]    = result
        elif tc["name"] == "run_fio":
            updates["fio_raw_output"]    = result
        elif tc["name"] == "run_vdbench":
            updates["vdbench_raw_output"] = result
        elif tc["name"] == "parse_fio_output":
            try:    updates["fio_results"]     = json.loads(result)
            except: updates["fio_results"]     = {"raw": result}
        elif tc["name"] == "parse_vdbench_output":
            try:    updates["vdbench_results"] = json.loads(result)
            except: updates["vdbench_results"] = {"raw": result}

    return {"messages": new_messages, **updates}


@rate_limited
def _compare_invoke(messages):
    return LLM_PLAIN.invoke(messages)


def comparison_node(state: BenchmarkState) -> dict:
    """Generate side-by-side comparison of fio vs vdbench results."""
    fio     = state.get("fio_results",     {})
    vdbench = state.get("vdbench_results", {})

    if DRY_RUN:
        answer = f"""
BENCHMARK COMPARISON SUMMARY  [DRY-RUN SIMULATION]
====================================================
Metric             | fio            | vdbench
-------------------+----------------+----------------
Total IOPS         | {fio.get('total_iops', 'N/A'):<14} | {vdbench.get('total_iops', 'N/A')}
Read IOPS          | {fio.get('read_iops',  'N/A'):<14} | {vdbench.get('read_iops',  'N/A')}
Write IOPS         | {fio.get('write_iops', 'N/A'):<14} | {vdbench.get('write_iops', 'N/A')}
Total BW (MB/s)    | {fio.get('total_bw_mbps','N/A'):<14} | {vdbench.get('total_bw_mbps','N/A')}
Read BW (MB/s)     | {fio.get('read_bw_mbps','N/A'):<14} | {vdbench.get('read_bw_mbps','N/A')}
Write BW (MB/s)    | {fio.get('write_bw_mbps','N/A'):<14} | {vdbench.get('write_bw_mbps','N/A')}
Avg Read Lat (us)  | {fio.get('read_lat_us','N/A'):<14} | {vdbench.get('avg_resp_ms','N/A')} ms

ANALYSIS
--------
[DRY-RUN] vdbench reports ~{round((vdbench.get('total_iops',0) - fio.get('total_iops',0)) / max(fio.get('total_iops',1),1) * 100, 1)}% more total IOPS than fio for this workload.
This is typical — vdbench uses a tighter I/O loop and reports aggregate differently.

RECOMMENDATION
--------------
Use fio results as the primary baseline (JSON output is more precise).
Use vdbench to validate and cross-check, especially for latency profiling.
""".strip()
        return {"comparison": answer, "final_answer": answer}

    prompt = f"""
You are a storage performance expert. Compare these benchmark results and
produce a clean side-by-side table for a storage engineer.

fio results:
{json.dumps(fio, indent=2)}

vdbench results:
{json.dumps(vdbench, indent=2)}

Format:
BENCHMARK COMPARISON SUMMARY
==============================
Metric            | fio     | vdbench
------------------+---------+---------
Total IOPS        | ...     | ...
Read IOPS         | ...     | ...
Write IOPS        | ...     | ...
Total BW (MB/s)   | ...     | ...
Read BW (MB/s)    | ...     | ...
Write BW (MB/s)   | ...     | ...
Avg Latency       | ...     | ...

ANALYSIS
--------
2-3 sentences comparing results and noting differences.

RECOMMENDATION
--------------
Which result to trust more and why.
"""
    response = _compare_invoke([{"role": "user", "content": prompt}])
    return {"comparison": response.content, "final_answer": response.content}


# ─────────────────────────────────────────────────────────────
# EDGE ROUTING
# ─────────────────────────────────────────────────────────────

def route_after_planner(state: BenchmarkState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tool"
    return "compare"


def route_after_tool(state: BenchmarkState) -> str:
    fio_done     = bool(state.get("fio_results"))
    vdbench_done = bool(state.get("vdbench_results"))
    if fio_done and vdbench_done:
        return "compare"
    return "planner"


# ─────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────

def build_benchmark_agent():
    graph = StateGraph(BenchmarkState)
    graph.add_node("planner",      planner_node)
    graph.add_node("execute_tool", tool_executor_node)
    graph.add_node("compare",      comparison_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner, {
        "execute_tool": "execute_tool",
        "compare":      "compare",
    })
    graph.add_conditional_edges("execute_tool", route_after_tool, {
        "planner": "planner",
        "compare": "compare",
    })
    graph.add_edge("compare", END)
    return graph.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

def run_benchmark_agent(user_prompt: str) -> dict:
    """Run the storage benchmark agent with a natural language prompt."""
    global _dry_run_step
    _dry_run_step = 0           # reset for each run

    agent  = build_benchmark_agent()
    config = {"configurable": {"thread_id": f"bench-{int(time.time())}"}}

    initial_state: BenchmarkState = {
        "messages":           [HumanMessage(content=user_prompt)],
        "workload_params":    {},
        "fio_config":         "",
        "vdbench_config":     "",
        "fio_raw_output":     "",
        "vdbench_raw_output": "",
        "fio_results":        {},
        "vdbench_results":    {},
        "comparison":         "",
        "error":              "",
        "final_answer":       "",
    }

    print("\n" + "=" * 65)
    print("STORAGE BENCHMARK AI AGENT")
    print("=" * 65)
    print(f"Prompt : {user_prompt}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode   : {'DRY-RUN (simulated)' if DRY_RUN else 'LIVE'}")
    print("=" * 65 + "\n")

    result = agent.invoke(initial_state, config=config)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(result.get("final_answer", "(no comparison produced)"))
    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DRY_RUN and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Get a free key at https://aistudio.google.com")
        print("Or run in dry-run mode: export DRY_RUN=true")
        exit(1)

    # ── Pick a prompt ──────────────────────────────────────
    prompt = (
        "Run a 70/30 read/write mixed workload on /mnt/test "
        "with 128K block size, 32 threads, queue depth 64, "
        "target 50000 IOPS for 60 seconds."
    )
    # prompt = "Sequential read on /dev/sdb with 1MB blocks, 8 threads, 120 seconds, max throughput."
    # prompt = "Random write stress on /mnt/nfs with 4K blocks, 64 threads, iodepth 128, 90 seconds."
    # prompt = "Mixed 50/50 read write on /mnt/efs, 64K blocks, 16 threads, target 20000 IOPS, 60 seconds."

    result = run_benchmark_agent(prompt)

    # Save full results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "prompt":          prompt,
        "timestamp":       timestamp,
        "dry_run":         DRY_RUN,
        "fio_config":      result.get("fio_config"),
        "vdbench_config":  result.get("vdbench_config"),
        "fio_results":     result.get("fio_results"),
        "vdbench_results": result.get("vdbench_results"),
        "comparison":      result.get("comparison"),
    }
    out_path = f"/tmp/benchmark_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {out_path}")
