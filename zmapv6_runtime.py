"""Runtime helpers for independent ZMapv6 worker processes."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ZMAP = ROOT.parent / ".local-tools/zmapv6/build/src/zmap"
DEFAULT_LIBRARY_DIR = ROOT.parent / ".local-tools/zmapv6-deps/usr/lib/x86_64-linux-gnu"


def make_zmapv6_scanner(
    interface: str,
    zmap_path: str,
    gateway_mac: str,
    rate_pps: int = 5000,
    is_tcp: bool = False,
):
    """Create an isolated scanner; each worker owns its rate and no global patch is needed."""
    if rate_pps <= 0:
        raise ValueError("rate_pps must be positive")

    def start_scanning(config):
        batch_rate = int(config.get("rate_pps", rate_pps))
        if batch_rate <= 0:
            raise ValueError("batch rate_pps must be positive")
        result_path = config["zmap_result_path"]
        metadata_path = result_path + ".metadata.json"
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
            result_path,
            "--rate",
            str(batch_rate),
            "--verbosity=0",
            "--disable-syslog",
            "--output-module=csv",
            "--output-fields=saddr",
            "--output-filter=success=1 && repeat=0",
            f"--metadata-file={metadata_path}",
        ]
        command.extend(
            ["-M", "ipv6_tcp_synscan", "-p", "443"]
            if is_tcp else ["-M", "icmp6_echoscan"]
        )

        environment = dict(os.environ)
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(DEFAULT_LIBRARY_DIR), current])
        )
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            _, stderr = process.communicate(
                input=(config["passport"] + "\n").encode("utf-8")
            )
            if process.returncode != 0:
                details = stderr.decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"ZMapv6 exited with {process.returncode}: {details}"
                )
            if not Path(result_path).exists():
                raise RuntimeError(f"ZMapv6 result file not found: {result_path}")

            try:
                with open(metadata_path, encoding="utf-8") as metadata_file:
                    hit_rate = json.load(metadata_file).get("hitrate")
                if hit_rate is not None:
                    print(f"Hit rate: {hit_rate:.2f}%", flush=True)
            except FileNotFoundError:
                pass
        finally:
            try:
                os.remove(metadata_path)
            except FileNotFoundError:
                pass

    return start_scanning


def install_zmapv6(
    scanner_module,
    interface: str,
    zmap_path: str,
    gateway_mac: str,
    rate_pps: int = 5000,
) -> None:
    """Backward-compatible installer; new concurrent code uses the factory directly."""
    scanner_module.startScaning = make_zmapv6_scanner(
        interface=interface,
        zmap_path=zmap_path,
        gateway_mac=gateway_mac,
        rate_pps=rate_pps,
        is_tcp=scanner_module.__name__.endswith("trap_tcp"),
    )
