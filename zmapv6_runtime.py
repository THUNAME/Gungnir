"""Runtime adapter for the workspace-local ZMapv6 build.

Only the new experiment process is patched; legacy scanner files stay unchanged.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ZMAP = ROOT.parent / ".local-tools/zmapv6/build/src/zmap"
DEFAULT_LIBRARY_DIR = ROOT.parent / ".local-tools/zmapv6-deps/usr/lib/x86_64-linux-gnu"


def install_zmapv6(
    scanner_module,
    interface: str,
    zmap_path: str,
    gateway_mac: str,
    bandwidth: str = "1M",
) -> None:
    is_tcp = scanner_module.__name__.endswith("trap_tcp")

    def start_scanning(config):
        command = [
            "sudo",
            "-S",
            str(zmap_path),
            "-i",
            interface,
            f"--ipv6-source-ip={config['local_ipv6']}",
            f"--gateway-mac={gateway_mac}",
            f"--ipv6-target-file={config['generated_address_path']}",
            "-o",
            config["zmap_result_path"],
            "-B",
            bandwidth,
            "--verbosity=0",
        ]
        if is_tcp:
            command.extend(["-M", "ipv6_tcp_synscan", "-p", "443"])
        else:
            command.extend(["-M", "icmp6_echoscan"])

        environment = dict(os.environ)
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(DEFAULT_LIBRARY_DIR), current])
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        stdout, stderr = process.communicate(
            input=(config["passport"] + "\n").encode("utf-8")
        )
        if process.returncode != 0:
            details = stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ZMapv6 exited with {process.returncode}: {details}")

        text = stderr.decode("utf-8", errors="replace")
        matches = re.findall(r"\d+\.?\d*", text[-200:])
        hit_rate = matches[-1] if matches else "0"
        print(f"Hit rate: {hit_rate}%", flush=True)
        return hit_rate

    scanner_module.startScaning = start_scanning
