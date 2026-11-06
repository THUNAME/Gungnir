"""Runtime helpers for independent, multi-protocol ZMapv6 workers."""
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
    gateway_mac: str | None = None,
    rate_pps: int = 5000,
    is_tcp: bool = False,
    *,
    protocol: str | None = None,
    port: int = 0,
    dns_domain: str = "example.com",
    use_sudo: bool = True,
):
    """Create an isolated scanner; protocol and rate cannot leak across workers."""
    if rate_pps <= 0:
        raise ValueError("rate_pps must be positive")
    protocol = protocol or ("tcp" if is_tcp else "icmp")
    if protocol not in {"icmp", "tcp", "udp"}:
        raise ValueError(f"unsupported scan protocol: {protocol}")
    if is_tcp and protocol == "tcp" and not port:
        port = 443
    if protocol in {"tcp", "udp"} and not 1 <= int(port) <= 65535:
        raise ValueError("TCP/UDP scanning requires a port in 1..65535")
    if protocol == "udp" and int(port) == 53 and not dns_domain.strip():
        raise ValueError("UDP/53 requires a non-empty DNS domain")

    def start_scanning(config):
        batch_rate = int(config.get("rate_pps", rate_pps))
        if batch_rate <= 0:
            raise ValueError("batch rate_pps must be positive")
        result_path = config["zmap_result_path"]
        metadata_path = result_path + ".metadata.json"
        command = ["sudo", "-n"] if use_sudo else []
        command.extend([
            str(zmap_path),
            "-i", interface,
            f"--ipv6-source-ip={config['local_ipv6']}",
            f"--ipv6-target-file={config['generated_address_path']}",
            "-o", result_path,
            "--rate", str(batch_rate),
            "--verbosity=0",
            "--disable-syslog",
            "--output-module=csv",
            "--output-fields=saddr",
            "--output-filter=success=1 && repeat=0",
            f"--metadata-file={metadata_path}",
        ])
        configured_gateway = config.get("gateway_mac") or gateway_mac
        if configured_gateway:
            command.append(f"--gateway-mac={configured_gateway}")
        if protocol == "icmp":
            command.extend(["-M", "icmp6_echoscan"])
        elif protocol == "tcp":
            command.extend(["-M", "ipv6_tcp_synscan", "-p", str(port)])
        elif int(port) == 53:
            domain = str(config.get("dns_domain", dns_domain)).strip()
            if not domain:
                raise ValueError("UDP/53 requires a non-empty DNS domain")
            command.extend([
                "-M", "ipv6_dns", "-p", "53",
                "--probe-args", f"name:{domain}",
            ])
        else:
            command.extend(["-M", "ipv6_udp", "-p", str(port)])

        environment = dict(os.environ)
        current = environment.get("LD_LIBRARY_PATH", "")
        if DEFAULT_LIBRARY_DIR.exists():
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                filter(None, [str(DEFAULT_LIBRARY_DIR), current])
            )
        try:
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            if process.returncode != 0:
                details = process.stderr.decode("utf-8", errors="replace")[-2000:]
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
    gateway_mac: str | None = None,
    rate_pps: int = 5000,
) -> None:
    """Backward-compatible installer; concurrent code uses the factory directly."""
    scanner_module.startScaning = make_zmapv6_scanner(
        interface=interface,
        zmap_path=zmap_path,
        gateway_mac=gateway_mac,
        rate_pps=rate_pps,
        protocol="tcp" if scanner_module.__name__.endswith("trap_tcp") else "icmp",
        port=443,
    )
