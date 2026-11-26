#!/usr/bin/env python3
"""
Real POSIX TCP Multi-Process Transport Validation Suite (Delivery 11.10 — Gate A).

Spawns 1 standalone C11 fed_net_server OS process and K independent fed_net_client
OS processes exchanging raw HFED binary frames over real loopback TCP sockets.

Scenarios verified:
  1. Fed-Forest (Non-IID 4-Class) + 17 B TCP stream fragmentation + CRC32 bit-flip rejection
  2. Fed-VFDT Coordinated (Non-IID 4-Class) + 13 B TCP fragmentation + CRC32 rejection
  3. Fed-VFDT Stats-Merge (Extreme 100% pure 1-class/client Non-IID) + 19 B TCP fragmentation
  4. Real Disconnected Straggler Catch-Up (client 4 offline rounds 2..6, reconnects round 7)
  5. Fed-VFDT Stats-Merge Regression (Non-IID piecewise) + 23 B TCP fragmentation
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def run_scenario(
    name: str,
    server_bin: Path,
    client_bin: Path,
    mode: str,
    task: str,
    n_clients: int,
    rounds: int,
    samples_per_round: int,
    frag_bytes: int,
    pure_non_iid: bool = False,
    corrupt_client_round: Optional[Tuple[int, int]] = None,
    straggler_client_skip: Optional[Tuple[int, int, int]] = None,
    timeout_s: float = 30.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    srv_cmd = [
        str(server_bin),
        "--mode", mode,
        "--task", task,
        "--port", "0",
        "--n-clients", str(n_clients),
        "--rounds", str(rounds),
        "--fragment", str(frag_bytes),
    ]
    if straggler_client_skip is not None:
        _, s_start, s_end = straggler_client_skip
        srv_cmd.extend(["--straggler-skip", f"{s_start}:{s_end}"])

    srv_proc = subprocess.Popen(
        srv_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert srv_proc.stdout is not None
    ready_line = srv_proc.stdout.readline().strip()
    if not ready_line.startswith("READY port="):
        srv_proc.kill()
        _, err = srv_proc.communicate()
        raise RuntimeError(f"[{name}] Server failed to bind: stdout='{ready_line}' stderr='{err}'")

    port = int(ready_line.split("=", 1)[1])

    client_procs: List[subprocess.Popen] = []
    for cid in range(1, n_clients + 1):
        cli_cmd = [
            str(client_bin),
            "--mode", mode,
            "--task", task,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--client-id", str(cid),
            "--rounds", str(rounds),
            "--samples-per-round", str(samples_per_round),
            "--fragment", str(frag_bytes),
        ]
        if pure_non_iid:
            cli_cmd.append("--pure-non-iid")
        if corrupt_client_round is not None and corrupt_client_round[0] == cid:
            cli_cmd.extend(["--inject-corrupt-round", str(corrupt_client_round[1])])
        if straggler_client_skip is not None and straggler_client_skip[0] == cid:
            _, s_start, s_end = straggler_client_skip
            cli_cmd.extend(["--skip-rounds", f"{s_start}:{s_end}"])

        client_procs.append(
            subprocess.Popen(cli_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        )

    client_results: List[Dict[str, Any]] = []
    try:
        for cid, p in enumerate(client_procs, start=1):
            out, err = p.communicate(timeout=timeout_s)
            if p.returncode != 0:
                raise RuntimeError(f"[{name}] Client {cid} exited {p.returncode}: stderr={err}")
            client_results.append(json.loads(out.strip()))

        srv_out, srv_err = srv_proc.communicate(timeout=timeout_s)
        if srv_proc.returncode != 0:
            raise RuntimeError(f"[{name}] Server exited {srv_proc.returncode}: stderr={srv_err}")
        server_result = json.loads(srv_out.strip())
    finally:
        for p in client_procs:
            if p.poll() is None:
                p.kill()
        if srv_proc.poll() is None:
            srv_proc.kill()

    return server_result, client_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Real POSIX TCP Multi-Process Federated Validation")
    parser.add_argument("--server-bin", type=str, required=True)
    parser.add_argument("--client-bin", type=str, required=True)
    args = parser.parse_args()

    server_bin = Path(args.server_bin).resolve()
    client_bin = Path(args.client_bin).resolve()

    print("=" * 80)
    print(" KANSHO REAL POSIX TCP MULTI-PROCESS TRANSPORT VALIDATION (GATE A — 11.10)")
    print("=" * 80)

    t0 = time.time()
    passed = 0
    total = 5

    # 1. Fed-Forest over fragmented TCP + CRC32 bit-flip rejection
    srv, clis = run_scenario(
        name="1. Fed-Forest (Non-IID 4-Class, 17B frag + CRC32)",
        server_bin=server_bin,
        client_bin=client_bin,
        mode="forest",
        task="classif",
        n_clients=4,
        rounds=6,
        samples_per_round=200,
        frag_bytes=17,
        pure_non_iid=False,
        corrupt_client_round=(2, 3),
    )
    avg_acc = sum(c["holdout_acc"] for c in clis) / len(clis)
    assert srv["crc_rejected"] == 1, f"Expected 1 CRC32 rejection, got {srv['crc_rejected']}"
    assert srv["frames_ok"] == 24, f"Expected 24 valid frames, got {srv['frames_ok']}"
    assert all(c["topology_version"] == 3 for c in clis), "Each client must hold 3 peer compact trees"
    assert avg_acc >= 85.0, f"Expected Fed-Forest global acc >= 85%, got {avg_acc:.2f}%"
    passed += 1
    print(
        f"  [PASS 1/{total}] Fed-Forest TCP (17B frag)       | Avg Acc={avg_acc:6.2f}% | "
        f"CRC_rej={srv['crc_rejected']} | Wire={(srv['uplink_bytes']+srv['downlink_bytes'])/1024:.1f} KB"
    )

    # 2. Fed-VFDT Coordinated over fragmented TCP + CRC32 rejection
    srv, clis = run_scenario(
        name="2. Fed-VFDT Coordinated (Non-IID 4-Class, 13B frag + CRC32)",
        server_bin=server_bin,
        client_bin=client_bin,
        mode="vfdt_coord",
        task="classif",
        n_clients=4,
        rounds=8,
        samples_per_round=150,
        frag_bytes=13,
        pure_non_iid=False,
        corrupt_client_round=(1, 4),
    )
    avg_acc = sum(c["holdout_acc"] for c in clis) / len(clis)
    assert srv["crc_rejected"] == 1, f"Expected 1 CRC32 rejection, got {srv['crc_rejected']}"
    assert srv["global_splits"] >= 2, f"Expected >= 2 global splits, got {srv['global_splits']}"
    assert all(c["topology_version"] == srv["topology_version"] for c in clis), "All clients must match server topology_version"
    assert avg_acc >= 85.0, f"Expected Fed-VFDT Coord global acc >= 85%, got {avg_acc:.2f}%"
    passed += 1
    print(
        f"  [PASS 2/{total}] Fed-VFDT Coord TCP (13B frag)   | Avg Acc={avg_acc:6.2f}% | "
        f"Splits={srv['global_splits']} | TopoV={srv['topology_version']} | CRC_rej={srv['crc_rejected']} | "
        f"Wire={(srv['uplink_bytes']+srv['downlink_bytes'])/1024:.1f} KB"
    )

    # 3. Fed-VFDT Stats-Merge under EXTREME 100%-pure 1-class/client Non-IID
    srv, clis = run_scenario(
        name="3. Fed-VFDT Stats-Merge (Extreme 100% Pure Non-IID, 19B frag)",
        server_bin=server_bin,
        client_bin=client_bin,
        mode="vfdt_merge",
        task="classif",
        n_clients=4,
        rounds=8,
        samples_per_round=120,
        frag_bytes=19,
        pure_non_iid=True,
    )
    avg_acc = sum(c["holdout_acc"] for c in clis) / len(clis)
    assert srv["global_splits"] >= 3, f"Expected >= 3 splits under extreme pure Non-IID, got {srv['global_splits']}"
    assert all(c["topology_version"] == srv["topology_version"] for c in clis), "Topology mismatch across clients"
    assert avg_acc >= 90.0, f"Expected Stats-Merge extreme Non-IID acc >= 90%, got {avg_acc:.2f}%"
    passed += 1
    print(
        f"  [PASS 3/{total}] Fed-VFDT Merge Extreme Non-IID  | Avg Acc={avg_acc:6.2f}% | "
        f"Splits={srv['global_splits']} | TopoV={srv['topology_version']} | "
        f"Wire={(srv['uplink_bytes']+srv['downlink_bytes'])/1024:.1f} KB"
    )

    # 4. Real Disconnected Straggler Catch-Up over TCP (client 4 offline rounds 2..6)
    srv, clis = run_scenario(
        name="4. Disconnected Straggler Catch-Up (client 4 offline R2..R6)",
        server_bin=server_bin,
        client_bin=client_bin,
        mode="vfdt_merge",
        task="classif",
        n_clients=4,
        rounds=8,
        samples_per_round=120,
        frag_bytes=11,
        pure_non_iid=False,
        straggler_client_skip=(4, 2, 6),
    )
    straggler = clis[3]
    assert straggler["rounds_participated"] == 3, f"Expected straggler to participate in 3 rounds, got {straggler['rounds_participated']}"
    assert srv["global_splits"] >= 2, f"Expected >= 2 global splits, got {srv['global_splits']}"
    assert straggler["topology_version"] == srv["topology_version"], (
        f"Straggler failed resync: client={straggler['topology_version']} vs server={srv['topology_version']}"
    )
    assert straggler["holdout_acc"] >= 85.0, f"Expected resynced straggler acc >= 85%, got {straggler['holdout_acc']:.2f}%"
    assert (srv["catchups_sent"] + srv["snapshots_sent"]) >= 1, "Server must send catch-up or snapshot"
    passed += 1
    print(
        f"  [PASS 4/{total}] Real Straggler Catch-Up (11B)   | Straggler Acc={straggler['holdout_acc']:6.2f}% | "
        f"TopoV={straggler['topology_version']}/{srv['topology_version']} | "
        f"Catchups={srv['catchups_sent']} Snaps={srv['snapshots_sent']}"
    )

    # 5. Fed-VFDT Stats-Merge Continuous Regression over fragmented TCP
    srv, clis = run_scenario(
        name="5. Fed-VFDT Stats-Merge Regression (Non-IID, 23B frag)",
        server_bin=server_bin,
        client_bin=client_bin,
        mode="vfdt_merge",
        task="reg",
        n_clients=4,
        rounds=8,
        samples_per_round=120,
        frag_bytes=23,
    )
    avg_rmse = sum(c["holdout_rmse"] for c in clis) / len(clis)
    assert srv["global_splits"] >= 1, f"Expected >= 1 regression split, got {srv['global_splits']}"
    assert all(c["topology_version"] == srv["topology_version"] for c in clis), "Regression topology mismatch"
    assert avg_rmse < 8.0, f"Expected regression RMSE < 8.0, got {avg_rmse:.4f}"
    passed += 1
    print(
        f"  [PASS 5/{total}] Fed-VFDT Merge Regression (23B) | Avg RMSE={avg_rmse:6.3f} | "
        f"Splits={srv['global_splits']} | TopoV={srv['topology_version']} | "
        f"Wire={(srv['uplink_bytes']+srv['downlink_bytes'])/1024:.1f} KB"
    )

    elapsed = time.time() - t0
    print("=" * 80)
    print(f" STATUS: {passed}/{total} REAL TCP MULTI-PROCESS SCENARIOS PASSED in {elapsed:.2f}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
