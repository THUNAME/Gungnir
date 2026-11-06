#!/usr/bin/env python3
"""Production CLI for multi-protocol Adaptive Threshold Probing."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import tempfile
import time
from pathlib import Path

from adaptive_probe import (
    DEFAULT_AS_RATE_PPS,
    DEFAULT_BATCH_TARGETS,
    DEFAULT_BGP_RATE_PPS,
    DEFAULT_RESPONSE_THRESHOLD,
    DEFAULT_SCAN_RATE_PPS,
    DEFAULT_ZMAP_WORKERS,
    AdaptiveFrpPipeline,
)

ROOT = Path(__file__).resolve().parent
MODULES = {"icmp": "icmp6_echoscan", "tcp": "ipv6_tcp_synscan", "udp": "ipv6_dns"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One IPv6 CIDR per line")
    parser.add_argument("--output-frp", required=True, help="Accepted CIDR output")
    parser.add_argument("--out-json", help="Audit JSON; defaults to OUTPUT_FRP.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", choices=("icmp", "tcp", "udp"), required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--local-ipv6", required=True)
    parser.add_argument("--zmap-bin", default="zmap")
    parser.add_argument("--gateway-mac")
    parser.add_argument("--dns-domain", default="example.com")
    parser.add_argument("--bgp-dat", default=str(ROOT / "data" / "latestv6.dat"))
    parser.add_argument("--generated-folder", required=True)
    parser.add_argument("--zmap-folder", required=True)
    parser.add_argument("--rate", type=int, default=DEFAULT_SCAN_RATE_PPS)
    parser.add_argument("--bgp-rate", type=float, default=DEFAULT_BGP_RATE_PPS)
    parser.add_argument("--as-rate", type=float, default=DEFAULT_AS_RATE_PPS)
    parser.add_argument("--batch-targets", type=int, default=DEFAULT_BATCH_TARGETS)
    parser.add_argument("--workers", type=int, default=DEFAULT_ZMAP_WORKERS)
    parser.add_argument("--response-threshold", type=float, default=DEFAULT_RESPONSE_THRESHOLD)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()
    args.out_json = args.out_json or args.output_frp + ".json"
    if args.protocol in {"tcp", "udp"} and not 1 <= args.port <= 65535:
        parser.error("TCP/UDP scanning requires --port in 1..65535")
    if args.protocol == "icmp":
        args.port = 0
    if args.protocol == "udp" and args.port != 53:
        parser.error("the production UDP probe currently supports only UDP/53")
    if not args.dns_domain.strip():
        parser.error("--dns-domain must not be empty")
    if args.rate <= 0 or args.batch_targets <= 0 or args.workers not in {1, 2}:
        parser.error("rate/batch-targets must be positive and workers must be 1 or 2")
    if not all(math.isfinite(value) for value in (args.bgp_rate, args.as_rate)):
        parser.error("BGP/AS rates must be finite")
    if args.bgp_rate <= 0 or args.as_rate < 0:
        parser.error("BGP rate must be positive and AS rate must be non-negative")
    if not 0 < args.response_threshold <= 1:
        parser.error("--response-threshold must be in (0, 1]")
    try:
        ipaddress.IPv6Address(args.local_ipv6)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def read_prefixes(path):
    prefixes = set()
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                network = ipaddress.ip_network(value, strict=False)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid IPv6 CIDR: {value}") from exc
            if network.version != 6:
                raise ValueError(f"{path}:{line_number}: not an IPv6 CIDR: {value}")
            prefixes.add(str(network))
    if not prefixes:
        raise ValueError("input contains no IPv6 prefixes")
    return prefixes


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporary_text_path(destination):
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=directory, prefix=".active-scan-", text=True)
    return fd, path


def publish_results(output_frp, out_json, frp_set, manifest):
    """Build both files completely; the CIDR output is the final success marker."""
    frp_fd, frp_tmp = temporary_text_path(output_frp)
    json_fd, json_tmp = temporary_text_path(out_json)
    try:
        with os.fdopen(frp_fd, "w", encoding="utf-8") as output:
            for prefix in sorted(frp_set, key=lambda item: (
                int(ipaddress.ip_network(item).network_address), ipaddress.ip_network(item).prefixlen
            )):
                output.write(prefix + "\n")
            output.flush()
            os.fsync(output.fileno())
        with os.fdopen(json_fd, "w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(json_tmp, 0o644)
        os.chmod(frp_tmp, 0o644)
        os.replace(json_tmp, out_json)
        os.replace(frp_tmp, output_frp)
    except Exception:
        for path in (frp_tmp, json_tmp):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise


def main():
    args = parse_args()
    prefixes = read_prefixes(args.input)
    config = {
        "protocol": args.protocol,
        "port": args.port,
        "dns_domain": args.dns_domain,
        "interface": args.interface,
        "local_ipv6": args.local_ipv6,
        "zmap_path": args.zmap_bin,
        "gateway_mac": args.gateway_mac,
        "bgp_dat": args.bgp_dat,
        "generated_folder_path": args.generated_folder,
        "zmap_folder_path": args.zmap_folder,
    }
    print(
        f"run_id={args.run_id}; protocol={args.protocol}; port={args.port}; "
        f"unique_prefixes={len(prefixes)}; threshold={args.response_threshold}; "
        f"rate={args.rate}; bgp_rate={args.bgp_rate}; as_rate={args.as_rate}; "
        f"batch_targets={args.batch_targets}; workers={args.workers}",
        flush=True,
    )
    pipeline = AdaptiveFrpPipeline(
        config=config,
        response_threshold=args.response_threshold,
        scan_rate_bps=args.rate,
        bgp_rate_pps=args.bgp_rate,
        batch_targets=args.batch_targets,
        as_rate_pps=args.as_rate,
        zmap_workers=args.workers,
        early_stop=not args.no_early_stop,
    )
    started = time.monotonic()
    frp_set = pipeline.run(prefixes)
    manifest = {
        "run_id": args.run_id,
        "complete": True,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "module": MODULES[args.protocol],
        "protocol": args.protocol,
        "port": args.port,
        "probe_args": f"name:{args.dns_domain}" if args.protocol == "udp" else None,
        "response_threshold": args.response_threshold,
        "early_stop": not args.no_early_stop,
        "samples_per_prefix": 16,
        "input": os.path.abspath(args.input),
        "input_sha256": file_sha256(args.input),
        "output_frp": os.path.abspath(args.output_frp),
        "duration_seconds": time.monotonic() - started,
        **pipeline.summary(),
    }
    publish_results(args.output_frp, args.out_json, frp_set, manifest)
    print(
        f"Active scan complete: accepted={len(frp_set)}; "
        f"output={args.output_frp}; audit={args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
