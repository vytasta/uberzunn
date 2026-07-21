# Storage Benchmark AI Agent — fio vs vdbench

> An AI agent that translates natural language workload descriptions into fio and vdbench benchmark runs, then parses and compares IOPS, throughput, and latency side by side.

**Author:** vytasta — [vytasta.github.io/uberzunn/benchmark_agent](https://vytasta.github.io/uberzunn/benchmark_agent)

![CI](https://github.com/uberzunn/benchmark_agent/actions/workflows/ci.yml/badge.svg)

---

## What it does

You describe a storage workload in plain English. The agent does the rest.

```
"Run a 70/30 read/write mixed workload on /mnt/nvme
 with 128K block size, 32 threads, queue depth 64,
 target 50000 IOPS for 60 seconds."
```

The agent will:

1. Parse your prompt for target, block size, read/write ratio, threads, iodepth, IOPS target, and runtime
2. Generate a `fio` job file and a `vdbench` parameter file
3. Execute both benchmarks against your target
4. Parse IOPS, throughput (MB/s), and latency from both outputs
5. Produce a side-by-side comparison table with analysis and recommendation

---

## Architecture

```
User prompt
    │
    ▼
┌─────────┐     tool call      ┌──────────────┐
│ Planner │ ─────────────────► │ Tool Executor│
│  (LLM)  │ ◄─────────────────  (fio/vdbench) │
└─────────┘     loop until      └──────────────┘
    │           both done
    ▼
┌──────────┐
│ Compare  │  ──► Side-by-side table + analysis
└──────────┘
```

Built with **LangGraph** (stateful graph execution) and **Google Gemini 1.5 Flash** (free tier).

Rate limiting is handled with a simple `THROTTLE_SECONDS=5` delay between LLM calls — stays comfortably under Gemini's 15 RPM free tier limit without any retry complexity.

---

## Supported targets

| Environment       | Example target            |
|-------------------|---------------------------|
| Local NVMe/SSD    | `/mnt/nvme` or `/dev/sdb` |
| NFS mount         | `/mnt/nfs`                |
| AWS EFS           | `/mnt/efs`                |
| AWS EBS (EC2)     | `/dev/nvme1n1`            |
| EKS PVC           | `/mnt/pvc`                |
| On-prem SAN       | `/dev/mapper/san_lun0`    |

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/uberzunn/benchmark_agent.git
cd benchmark_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install fio and vdbench

```bash
# fio (Linux)
sudo apt install fio          # Debian/Ubuntu
sudo yum install fio          # RHEL/CentOS

# vdbench — download from Oracle (free, requires Oracle account)
# https://www.oracle.com/downloads/server-storage/vdbench-downloads.html
# Extract and add to PATH:  export PATH=$PATH:/opt/vdbench
```

### 4. Set your Gemini API key

```bash
# Free key — no credit card needed
# Get one at: https://aistudio.google.com → "Get API key"
export GOOGLE_API_KEY="AIza..."
```

### 5. Run

```bash
python storage_benchmark_agent.py
```

---

## Dry-run mode

Test the full agent flow without fio, vdbench, or a Gemini key:

```bash
export DRY_RUN=true
python storage_benchmark_agent.py
```

Dry-run simulates all tool calls with realistic data and prints the full comparison table. Useful for CI, demos, and understanding the agent flow before running real benchmarks.

See [`sample_outputs/dry_run_sample.json`](sample_outputs/dry_run_sample.json) for an example output.

---

## Example prompts

Edit the `prompt` variable at the bottom of `storage_benchmark_agent.py`:

```python
# 70/30 mixed workload — your primary test
prompt = (
    "Run a 70/30 read/write mixed workload on /mnt/test "
    "with 128K block size, 32 threads, queue depth 64, "
    "target 50000 IOPS for 60 seconds."
)

# Sequential throughput test
prompt = "Sequential read on /dev/sdb with 1MB blocks, 8 threads, 120 seconds, max throughput."

# Random write stress
prompt = "Random write stress on /mnt/nfs with 4K blocks, 64 threads, iodepth 128, 90 seconds."

# Mixed 50/50 with IOPS cap
prompt = "Mixed 50/50 read write on /mnt/efs, 64K blocks, 16 threads, target 20000 IOPS, 60 seconds."
```

---

## Output

Results are printed to the terminal and saved to `/tmp/benchmark_results_<timestamp>.json`:

```
BENCHMARK COMPARISON SUMMARY
==============================
Metric             | fio            | vdbench
-------------------+----------------+----------------
Total IOPS         | 68900          | 67540
Read IOPS          | 48230          | 47278
Write IOPS         | 20670          | 20262
Total BW (MB/s)    | 4355.47        | 4221.3
Read BW (MB/s)     | 3046.88        | 2954.91
Write BW (MB/s)    | 1308.59        | 1266.39
Avg Read Lat (us)  | 650.0          | 0.943 ms

ANALYSIS
--------
fio and vdbench are within 2% of each other — excellent agreement.
vdbench latency reflects its warmup period smoothing.

RECOMMENDATION
--------------
Use fio as primary baseline (JSON output is more precise).
Use vdbench to cross-check, especially for latency profiling.
```

---

## Rate limiting

The agent uses a simple fixed throttle (`THROTTLE_SECONDS = 5`) between every Gemini call. This keeps API usage comfortably under the free tier's 15 RPM limit without any complex retry logic. Tune the value in `storage_benchmark_agent.py` if you upgrade to a paid tier.

---

## Project structure

```
benchmark_agent/
├── storage_benchmark_agent.py   # main agent
├── requirements.txt
├── .env.example                 # copy to .env and add your key
├── .gitignore
├── sample_outputs/
│   └── dry_run_sample.json      # example output
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions dry-run CI
```

---

## Roadmap

- [ ] Multi-target comparison (NVMe vs NFS vs EBS in one run)
- [ ] Add iometer and iozone support
- [ ] HTML/CSV report export
- [ ] Kubernetes Job runner for EKS workloads
- [ ] Slack/email result notifications

---

## License

MIT — free to use, modify, and share.
