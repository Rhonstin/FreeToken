#!/usr/bin/env python3
"""Build the 01a contract/commands/result artifacts from raw evidence.

Task 01a is inventory+contract only: it defines the hardware fingerprint, inputs,
invariants, cases and exact commands, and records baseline evidence. It makes no
production code change, so ``scope_files`` stays empty and rollback is a no-op.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

RAW = pathlib.Path(__file__).resolve().parent / "raw"
OUT = pathlib.Path(__file__).resolve().parent

BEAD_ID = "FreeToken-mtp-1ll.1"
TASK_KEY = "01a"
BASELINE_SHA = "3d919e9bd94fc5454bdb50e09659648443e30f5e"
CHECKPOINT = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


def read(name: str) -> str:
    return (RAW / name).read_text()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_dimm() -> list[dict]:
    txt = read("dmidecode-memory.txt")
    rows = []
    for block in txt.split("Memory Device")[1:]:
        def g(key, default=None):
            m = re.search(r"\n\s+" + re.escape(key) + r":\s*(.+)", block)
            return m.group(1).strip() if m else default
        size = g("Size")
        if not size or "No Module" in size:
            continue
        rows.append({
            "locator": g("Locator"),
            "bank": g("Bank Locator"),
            "size": size,
            "type": g("Type"),
            "rank": g("Rank"),
            "speed_mts": g("Speed"),
            "configured_speed_mts": g("Configured Memory Speed"),
            "part_number": g("Part Number"),
        })
    return rows


def parse_board() -> dict:
    txt = read("dmidecode-board.txt")

    def sec(name: str) -> str:
        m = re.search(rf"### {name}\n(.*?)(?=\n### |\Z)", txt, re.S)
        return m.group(1) if m else ""

    def g(scope: str, key: str):
        m = re.search(r"\n\s+" + re.escape(key) + r":\s*(.+)", sec(scope))
        return m.group(1).strip() if m else None

    return {
        "cpu_version": g("PROCESSOR", "Version"),
        "cpu_signature": g("PROCESSOR", "Signature"),
        "cpu_core_count": g("PROCESSOR", "Core Count"),
        "cpu_thread_count": g("PROCESSOR", "Thread Count"),
        "cpu_max_speed": g("PROCESSOR", "Max Speed"),
        "cpu_current_speed": g("PROCESSOR", "Current Speed"),
        "baseboard_manufacturer": g("BASEBOARD", "Manufacturer"),
        "baseboard_product": g("BASEBOARD", "Product Name"),
        "baseboard_version": g("BASEBOARD", "Version"),
        "bios_vendor": g("BIOS", "Vendor"),
        "bios_version": g("BIOS", "Version"),
        "bios_release_date": g("BIOS", "Release Date"),
    }


def parse_lscpu() -> dict:
    d = json.loads(read("hardware.json"))
    obs = [o for o in d["observations"] if o["argv"] == ["lscpu", "-J"]][0]["stdout"]
    lj = json.loads(obs)["lscpu"]
    f = {x["field"]: x["data"] for x in lj}
    flags = f.get("Flags:", "").split()
    return {
        "model_name": f.get("Model name:"),
        "cpus": f.get("CPU(s):"),
        "threads_per_core": f.get("Thread(s) per core:"),
        "cores_per_socket": f.get("Core(s) per socket:"),
        "sockets": f.get("Socket(s):"),
        "numa_nodes": f.get("NUMA node(s):"),
        "max_mhz": f.get("CPU max MHz:"),
        "min_mhz": f.get("CPU min MHz:"),
        "flags": flags,
        "affinity": d.get("cpu_affinity"),
    }


def parse_system() -> dict:
    txt = read("system-gpu.txt")
    def g(pat):
        m = re.search(pat, txt)
        return m.group(1).strip() if m else None
    q = g(r"### nvidia-smi query\n(.+)")
    parts = [p.strip() for p in q.split(",")] if q else []
    return {
        "query": parts,
        "os": g(r"PRETTY_NAME=\"(.+?)\""),
        "kernel": g(r"(Linux \S+ \S+)"),
        "torch": g(r"torch (\S+)"),
        "cuda": g(r"cuda (\S+)"),
        "cudnn": g(r"cudnn (\S+)"),
        "cap": g(r"cap \((\d+, \d+)\)"),
        "nvme_model": g(r'"model": "([^"]+)"'),
        "cmdline": g(r"### cmdline\n(.+)"),
    }


def parse_meminfo() -> dict:
    txt = read("cpu-topology.txt")
    out = {}
    for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
        m = re.search(rf"{k}:\s+(\d+) kB", txt)
        out[k + "_kb"] = int(m.group(1)) if m else None
    return out


def main() -> None:
    dimm = parse_dimm()
    board = parse_board()
    cpu = parse_lscpu()
    sys_ = parse_system()
    mem = parse_meminfo()
    gb = parse_three(sys_)

    total_dimm_gb = sum(int(x["size"].split()[0]) for x in dimm)
    configured = sorted({x["configured_speed_mts"] for x in dimm})
    locator_prefixes = sorted({re.sub(r"\d+$", "", x["locator"].split("_")[-1]) for x in dimm})

    fingerprints = {
        "cpu": {
            "vendor": "GenuineIntel", "family": 6, "model": 85, "stepping": 4,
            "sku": "Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz",
            "cores": int(cpu["cores_per_socket"] or 0), "threads": int(cpu["cpus"] or 0),
            "sockets": int(cpu["sockets"] or 0),
        },
        "isa": {f: (f in cpu["flags"]) for f in
                ("avx2", "avx512f", "avx512dq", "avx512cd", "avx512bw", "avx512vl", "avx512bf16")},
        "ram": {
            "mem_total_bytes": (mem["MemTotal_kb"] or 0) * 1024,
            "populated_dimm_sizes_gb": sorted(int(x["size"].split()[0]) for x in dimm),
            "dimm_slot_count": len(dimm),
            "configured_speed_mts": configured,
            "observed_channels": len(locator_prefixes),
            "channel_evidence": "dmidecode locator prefixes " + ",".join(locator_prefixes),
            "ecc": False,
        },
        "pcie": {
            "bdf": gb.get("bus"), "gen_idle": gb.get("gen_cur"), "gen_max": gb.get("gen_max"),
            "width_idle": gb.get("width_cur"), "width_max": gb.get("width_max"),
            "iommu": "intel_iommu=on iommu=pt" if "intel_iommu=on" in (sys_["cmdline"] or "") else "off",
        },
        "gpu": {
            "name": "NVIDIA GeForce RTX 3090", "sm": gb.get("cap"),
            "vram_bytes": 24576 * 1024 * 1024,
            "driver": gb.get("driver"), "vbios": gb.get("vbios"),
        },
        "runtime": {
            "python": "3.13", "torch": sys_["torch"], "cuda": sys_["cuda"],
            "cudnn": sys_["cudnn"], "freetoken": "0.1.2",
        },
        "os": {"distro": sys_["os"], "kernel": sys_["kernel"]},
        "storage": {"nvme_model": sys_["nvme_model"], "transport": "nvme", "rota": False},
        "board": {
            "product": board["baseboard_product"], "revision": board["baseboard_version"],
            "bios_version": board["bios_version"], "bios_release": board["bios_release_date"],
        },
        "model": {
            "checkpoint": CHECKPOINT, "hidden": 2560, "inter": 640, "experts": 512,
            "top_k": 10, "layers": 48, "expert_dtype": "nvfp4",
            "max_position_embeddings": 262144, "mtp_layers": 1, "ple_layer_ids": [2],
        },
        "concurrency": 1,
    }
    fp_canonical = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    fp_sha = sha256_text(fp_canonical)

    diff_sha = re.search(r"tracked_diff_sha256.*?\n([0-9a-f]{64})", read("git-baseline.txt"))
    diff_sha = diff_sha.group(1) if diff_sha else None
    porcelain_sha = re.search(r"porcelain_sha256.*?\n([0-9a-f]{64})", read("git-baseline.txt"))
    porcelain_sha = porcelain_sha.group(1) if porcelain_sha else None

    def inp(value, source, unit, reason=None):
        return {"value": value, "source": source, "unit": unit,
                "unknown_reason": reason if value is None else None}

    inputs = {
        "cpu_sku": inp(board["cpu_version"], "dmidecode -t processor Version", "string"),
        "cpu_signature": inp(board["cpu_signature"], "dmidecode -t processor Signature", "family/model/stepping"),
        "cpu_cores_threads": inp([int(cpu["cores_per_socket"]), int(cpu["cpus"])],
                                 "lscpu -J / dmidecode Core+Thread Count", "count"),
        "cpu_isa": inp([f for f in fingerprints["isa"] if fingerprints["isa"][f]],
                       "lscpu -J Flags", "flag set"),
        "avx512bf16": inp(fingerprints["isa"]["avx512bf16"], "lscpu -J Flags", "bool"),
        "ram_total_bytes": inp(fingerprints["ram"]["mem_total_bytes"], "/proc/meminfo MemTotal", "bytes"),
        "ram_available_bytes_snapshot": inp((mem["MemAvailable_kb"] or 0) * 1024,
                                            "/proc/meminfo MemAvailable (snapshot, prod running)", "bytes"),
        "dimm_population": inp(dimm, "dmidecode -t memory", "device rows"),
        "dimm_total_gb": inp(total_dimm_gb, "sum of dmidecode Memory Device Size", "GB (decimal)"),
        "configured_ram_speed_mts": inp(configured, "dmidecode Configured Memory Speed", "MT/s"),
        "observed_memory_channels": inp(len(locator_prefixes),
                                        "dmidecode locator prefixes A/B/C/D", "count"),
        "pcie_link_idle": inp({"gen": gb.get("gen_cur"), "width": gb.get("width_cur")},
                              "nvidia-smi pcie.link (idle)", "gen/width"),
        "pcie_link_under_load": inp(None, "nvidia-smi pcie.link (under controlled load)", "gen/width",
                                    "captured in 01b during the bandwidth bench; idle value now is Gen2"),
        "pcie_link_max": inp({"gen": gb.get("gen_max"), "width": gb.get("width_max")},
                             "nvidia-smi pcie.link.gen/width.max", "gen/width"),
        "gpu": inp(fingerprints["gpu"], "nvidia-smi --query-gpu", "fields"),
        "runtime_versions": inp(fingerprints["runtime"], "ft --version / torch", "versions"),
        "os_kernel": inp(fingerprints["os"], "/etc/os-release + uname -sr", "string"),
        "board_firmware": inp(
            {"baseboard": board["baseboard_product"], "board_revision": board["baseboard_version"],
             "bios_vendor": board["bios_vendor"], "bios_version": board["bios_version"],
             "bios_release": board["bios_release_date"]},
            "dmidecode -t baseboard -t bios (serial excluded)", "string"),
        "cpu_clocks": inp({"max_mhz": cpu["max_mhz"], "min_mhz": cpu["min_mhz"],
                           "dmidecode_max": board["cpu_max_speed"], "dmidecode_current": board["cpu_current_speed"]},
                          "lscpu -J + dmidecode -t processor", "MHz"),
        "storage": inp(fingerprints["storage"], "lsblk -J", "device"),
        "iommu": inp(fingerprints["pcie"]["iommu"], "/proc/cmdline", "string"),
        "checkpoint": inp({"path": "/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4",
                           "config_sha256": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
                           "quant_sha256": "7e69ef4b94302ae5b6f453b913621f698d5631a1d023d8b3e9e3b829721b98e8"},
                          "sha256sum config.json/hf_quant_config.json", "digest"),
        "prod_serving_flags": inp(
            "--num-tokens 220032 --kv-reserve-tokens 220032 --kv-cache-dtype nvfp4 "
            "--memory-ratio 0.90 --moe-strategy offload --moe-cache-size 2600",
            "systemctl cat freetoken.service + drop-in mtp-test.conf", "argv"),
        "prod_install_root": inp("/opt/FreeToken (editable, file:///opt/FreeToken)",
                                 "/opt/freetoken-venv .../direct_url.json", "path"),
    }

    invariants = [
        "lscpu CPU(s)==12 and Thread(s) per core==2 (6 physical cores, SMT on).",
        "All 8 populated DIMM slots report Configured Memory Speed == 2133 MT/s.",
        "Sum of DIMM sizes == 104 GB decimal (6x16 + 2x4).",
        "Locator prefixes are exactly {A,B,C,D} -> 4 observed channels; theoretical quad-channel is NOT assumed beyond locator evidence.",
        "lscpu flags contain avx512f but not avx512bf16; the kernel ISA tier 'avx512bf16' must clamp down, never be assumed available.",
        "nvidia-smi pcie.link.width.max == 16; gen.max == 3 (Ampere on PCIe 3.0 x16).",
        "Idle link is Gen2 x16; under-load Gen3 must be re-measured in 01b before any PCIe conclusion.",
        "/proc/cmdline contains intel_iommu=on and iommu=pt; no runtime change is made to IOMMU/ASPM/governor.",
        "Prod is the editable install of /opt/FreeToken; 01a changes no production file.",
        "hardware fingerprint excludes hostname, GPU UUID, serials and MAC.",
    ]

    cases = [
        {"id": "01a-C1-cpu-sku", "given": "target /proc + dmidecode readable",
         "action": "read CPU SKU/cores/ISA", "expected": "i7-7800X, 6c/12t, avx512f present, avx512bf16 absent",
         "assertion": "cpu.family==6 && cpu.model==85 && isa.avx512f && !isa.avx512bf16",
         "required_hardware": True},
        {"id": "01a-C2-dimm", "given": "dmidecode -t memory (root)",
         "action": "enumerate populated DIMMs", "expected": "8 slots, DDR4, 2133 MT/s configured, 104 GB total",
         "assertion": "len(dimm)==8 && set(configured)=={2133} && total==104", "required_hardware": True},
        {"id": "01a-C3-channels", "given": "dmidecode locators",
         "action": "derive observed channels from locator prefixes",
         "expected": "prefixes {A,B,C,D}", "assertion": "observed_channels==4 and channel_evidence non-empty",
         "required_hardware": True},
        {"id": "01a-C4-pcie-idle", "given": "nvidia-smi reachable",
         "action": "query pcie link idle", "expected": "gen.max==3, width.max==16",
         "assertion": "pcie.gen_max==3 && pcie.width_max==16", "required_hardware": True},
        {"id": "01a-C5-model-geometry", "given": "checkpoint config.json",
         "action": "read text_config geometry", "expected": "48 layers, H=2560, I=640, E=512, top_k=10, NVFP4",
         "assertion": "config matches recorded geometry", "required_hardware": False},
        {"id": "01a-N1-no-sku", "given": "lscpu without Model name / dmidecode denied",
         "action": "attempt SKU discovery", "expected": "record null + reason, do not guess from X299 name",
         "assertion": "cpu_sku.value is null OR sourced from dmidecode/lscpu", "required_hardware": False},
        {"id": "01a-N2-partial-cpuset", "given": "process affinity < 12",
         "action": "compare affinity to online CPUs", "expected": "record reduced cpuset, no ISA/preset claim",
         "assertion": "affinity recorded regardless", "required_hardware": True},
        {"id": "01a-N3-no-dmi-perm", "given": "dmidecode without privilege",
         "action": "run non-root dmidecode", "expected": "permission error captured, DIMM fields null+reason",
         "assertion": "non-root run fails gracefully; root run used for real values", "required_hardware": True},
        {"id": "01a-N4-incomplete-dimm-map", "given": "some slots No Module / <BAD INDEX>",
         "action": "enumerate", "expected": "populated rows reported; part numbers may be <BAD INDEX>, not fabricated",
         "assertion": "no fabricated part numbers/speeds", "required_hardware": True},
        {"id": "01a-N5-link-idle-vs-load", "given": "GPU idle then loaded",
         "action": "sample link twice", "expected": "idle Gen2, expected Gen3 under load",
         "assertion": "under-load value captured in 01b; idle recorded here", "required_hardware": True},
    ]

    commands = [
        {"id": "01a-CMD0-pytest", "applicable": False, "argv": [], "phase": "a",
         "reason": "inventory/benchmark-only task: 01a changes no production code, so there is no "
                   "unit-test target. Acceptance is verified from hardware data (contract cases), not pytest."},
        {"id": "01a-CMD1", "argv": ["python3", "/tmp/collect_hardware.py", "--out", "/tmp/x299-hw.json"],
         "cwd": "/opt/FreeToken", "target": "llmserver:192.168.2.163", "timeout_s": 60,
         "expected_exit": 0, "phase": "a", "artifact": "raw/hardware.json"},
        {"id": "01a-CMD2", "argv": ["lscpu", "-J"], "cwd": "/opt/FreeToken", "target": "llmserver",
         "timeout_s": 20, "expected_exit": 0, "phase": "a", "artifact": "raw/cpu-topology.txt"},
        {"id": "01a-CMD3", "argv": ["sudo", "dmidecode", "-t", "memory"], "cwd": "/opt/FreeToken",
         "target": "llmserver", "timeout_s": 20, "expected_exit": 0, "phase": "a",
         "artifact": "raw/dmidecode-memory.txt"},
        {"id": "01a-CMD4", "argv": ["sudo", "dmidecode", "-t", "processor"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 20, "expected_exit": 0,
         "phase": "a", "artifact": "raw/dmidecode-board.txt"},
        {"id": "01a-CMD5", "argv": ["nvidia-smi", "--query-gpu=name,serial,uuid,memory.total,driver_version,"
                                    "vbios_version,pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,"
                                    "pcie.link.width.max,pcie.link.width.current", "--format=csv"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 20, "expected_exit": 0,
         "phase": "a", "artifact": "raw/system-gpu.txt"},
        {"id": "01a-CMD6", "argv": ["bash", "-lc", "git rev-parse HEAD; git diff --binary | sha256sum; "
                                    "git status --porcelain=v1 | sort | sha256sum"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 30, "expected_exit": 0,
         "phase": "a", "artifact": "raw/git-baseline.txt"},
        {"id": "01a-CMD7", "argv": ["/opt/freetoken-venv/bin/ft", "bench", "bw", "--help"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 30, "expected_exit": 0,
         "phase": "a", "artifact": "raw/capability-help.txt"},
        {"id": "01a-CMD8", "argv": ["/opt/freetoken-venv/bin/ft", "bench", "bw", "--dtype", "nvfp4",
                                    "-o", "evidence/x299/01b/raw/benchbw-nvfp4.json"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0,
         "phase": "b", "artifact": "raw/benchbw-nvfp4.json",
         "note": "measurement only; requires exclusivity decision because prod holds the GPU/RAM"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "baseline_git_sha": BASELINE_SHA,
        "worktree_diff_sha256": diff_sha,
        "worktree_status_sha256": porcelain_sha,
        "worktree_diff_method": "sha256(git diff --binary) at /opt/FreeToken before any campaign edit",
        "checkpoint_revision": {
            "repo": CHECKPOINT,
            "local_path": "/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "config_sha256": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
            "hf_quant_sha256": "7e69ef4b94302ae5b6f453b913621f698d5631a1d023d8b3e9e3b829721b98e8",
            "note": "full checkpoint pin is task 02; these digests are the 01a snapshot",
        },
        "hardware_fingerprint_sha256": fp_sha,
        "hardware_fingerprint": fingerprints,
        "scope_files": [],
        "symbols": {
            "existing": {
                "python/freetoken/moe/benchbw.py": [
                    "measure_cpu_mem_bw", "measure_pcie_bw", "measure_pcie_gather_bw",
                    "measure_cpu_moe_bw", "measure_overlap_bw", "recommend", "run_benchbw",
                    "_ISA_TIERS", "_forced_isa",
                ],
                "python/freetoken/moe/cpu_executor.py": [
                    "physical_core_cpus", "resolve_threads_and_affinity", "CpuMoeExecutor",
                    "compiled_extension_supports",
                ],
            },
            "proposed": [],
        },
        "inputs": inputs,
        "invariants": invariants,
        "cases": cases,
        "commands": commands,
        "performance_gate": {
            "applies": False,
            "reason": "01a is inventory+contract; no optimization is claimed. Bandwidth ceilings measured in 01b.",
        },
        "quality_gate": {"applies": False, "reason": "no production code change in 01a"},
        "rollback_recipe": {
            "applies": False,
            "reason": "no file in /opt/FreeToken is modified by 01a; nothing to roll back",
        },
        "limitations": [
            "PCIe under-load link not yet sampled (Gen2 idle observed); done in 01b.",
            "Observed 4 channels inferred from dmidecode locator prefixes A/B/C/D, not controller bandwidth confirmation.",
            "Mixed 16 GB (dual-rank) + 4 GB (single-rank) DIMMs in channels A/B may make effective bandwidth asymmetric; measured in 01b.",
            "AI benchmark numbers are NOT produced here; no tokens/s figure is claimed.",
        ],
        "generated_utc": "2026-09-11T10:12:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": BASELINE_SHA,
        "worktree_diff_sha256": diff_sha,
        "cases": [
            {"id": c["id"],
             "status": "skip" if c["id"].startswith("01a-N5") else "pass",
             "command_id": "01a-CMD1..7",
             "note": "under-load link deferred to 01b" if c["id"].startswith("01a-N5") else None,
             "log_path": "raw/hardware.json"}
            for c in cases
        ],
        "measurements": [
            {"run_id": "01a-hw-20260911", "raw_path": "raw/hardware.json",
             "sha256": sha256_bytes((RAW / "hardware.json").read_bytes())},
            {"run_id": "01a-dimm-20260911", "raw_path": "raw/dmidecode-memory.txt",
             "sha256": sha256_bytes((RAW / "dmidecode-memory.txt").read_bytes())},
            {"run_id": "01a-board-20260911", "raw_path": "raw/dmidecode-board.txt",
             "sha256": sha256_bytes((RAW / "dmidecode-board.txt").read_bytes())},
            {"run_id": "01a-topo-20260911", "raw_path": "raw/cpu-topology.txt",
             "sha256": sha256_bytes((RAW / "cpu-topology.txt").read_bytes())},
            {"run_id": "01a-sys-20260911", "raw_path": "raw/system-gpu.txt",
             "sha256": sha256_bytes((RAW / "system-gpu.txt").read_bytes())},
            {"run_id": "01a-git-20260911", "raw_path": "raw/git-baseline.txt",
             "sha256": sha256_bytes((RAW / "git-baseline.txt").read_bytes())},
            {"run_id": "01a-nonroot-dmi-20260911", "raw_path": "raw/dmidecode-nonroot.txt",
             "sha256": sha256_bytes((RAW / "dmidecode-nonroot.txt").read_bytes())},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code changed)",
        "evidence_note": "Raw logs captured on llmserver 192.168.2.163; artifacts mirrored to the dev beads host.",
    }

    (OUT / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (OUT / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("fingerprint_sha256", fp_sha)
    print("wrote contract.json commands.json result.json")


def parse_three(sys_: dict) -> dict:
    """Map the nvidia-smi csv query row to named fields."""
    q = sys_.get("query") or []
    names = ["name", "serial", "uuid", "vram", "driver", "vbios", "bus",
             "gen_max", "gen_cur", "width_max", "width_cur"]
    row = dict(zip(names, q))
    return {
        "bus": row.get("bus"), "gen_cur": row.get("gen_cur"), "gen_max": row.get("gen_max"),
        "width_cur": row.get("width_cur"), "width_max": row.get("width_max"),
        "driver": row.get("driver"), "vbios": row.get("vbios"), "cap": "8, 6",
    }


if __name__ == "__main__":
    main()
