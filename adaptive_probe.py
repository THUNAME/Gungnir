"""自适应阈值 FRP 探测。

本模块不修改 trap.py / trap_tcp.py。AddressProducer 只负责生成地址文件，
ZMapConsumer 只负责调用 ZMapv6 并读取响应，AdaptiveFrpPipeline 负责逐层调度。
"""
from __future__ import annotations

import ipaddress
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from zmapv6_runtime import DEFAULT_ZMAP, install_zmapv6
except ImportError:  # 支持作为 Gungnir 包导入
    from .zmapv6_runtime import DEFAULT_ZMAP, install_zmapv6


DEFAULT_RESPONSE_THRESHOLD = 0.75
DEFAULT_SCAN_RATE_PPS = 100
MAX_LAYER = 15


@dataclass(frozen=True)
class Branch:
    """一个待探测分支；root 用于保持输入前缀与最终 FRP 的对应关系。"""

    root: str
    prefix: str


@dataclass(frozen=True)
class ProbeUnit:
    branch: Branch
    addresses: tuple[str, ...]


@dataclass(frozen=True)
class ProbeBatch:
    """Producer 与 Consumer 之间传递的文件批次。"""

    depth: int
    units: tuple[ProbeUnit, ...]
    address_path: Path
    result_path: Path

    @property
    def address_count(self) -> int:
        return sum(len(unit.addresses) for unit in self.units)


class Consumer(Protocol):
    def scan(self, batch: ProbeBatch) -> set[str]:
        """扫描一个批次，返回规范化后的响应 IPv6 地址集合。"""


def _canonical_prefix(prefix: str) -> str:
    return str(ipaddress.IPv6Network(prefix.strip(), strict=False))


def _canonical_address(address: str) -> str:
    return str(ipaddress.IPv6Address(address.strip()))


def _generate_addresses(prefix: str) -> tuple[str, ...]:
    """生成最多 16 个地址，依次覆盖接下来的 0..15 子前缀。"""
    network = ipaddress.IPv6Network(prefix, strict=False)
    remaining = 128 - network.prefixlen
    count = min(1 << remaining, 16)

    if remaining <= 4:
        return tuple(
            str(ipaddress.IPv6Address(int(network.network_address) + index))
            for index in range(count)
        )

    host_bits = remaining - 4
    addresses = []
    for index in range(16):
        suffix = (index << host_bits) | secrets.randbits(host_bits)
        addresses.append(
            str(ipaddress.IPv6Address(int(network.network_address) | suffix))
        )
    return tuple(addresses)


def _child_prefixes(prefix: str, addresses: tuple[str, ...]) -> tuple[str, ...]:
    """将本层的 16 个地址映射为下一层的 16 个子前缀。"""
    network = ipaddress.IPv6Network(prefix, strict=False)
    child_length = network.prefixlen + 4
    if child_length > 128:
        return ()

    mask = ((1 << child_length) - 1) << (128 - child_length)
    children = {
        str(
            ipaddress.IPv6Network(
                (int(ipaddress.IPv6Address(address)) & mask, child_length)
            )
        )
        for address in addresses
    }
    return tuple(
        sorted(
            children,
            key=lambda item: int(ipaddress.IPv6Network(item).network_address),
        )
    )


class AddressProducer:
    """Producer：计算一批活跃分支的地址并写入临时文件，不调用 subprocess。"""

    def __init__(self, generated_dir: Path, result_dir: Path):
        self.generated_dir = generated_dir
        self.result_dir = result_dir
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def produce(self, branches: list[Branch], depth: int) -> ProbeBatch:
        units = tuple(
            ProbeUnit(branch=branch, addresses=_generate_addresses(branch.prefix))
            for branch in branches
        )
        stem = f"adaptive_{time.time_ns()}_{depth}"
        address_path = self.generated_dir / f"{stem}.txt"
        result_path = self.result_dir / f"{stem}.txt"

        with address_path.open("w", encoding="utf-8") as output:
            for unit in units:
                for address in unit.addresses:
                    output.write(address + "\n")

        return ProbeBatch(
            depth=depth,
            units=units,
            address_path=address_path,
            result_path=result_path,
        )


def _required_setting(config: dict[str, Any], key: str, env_name: str) -> str:
    """从 config 或环境变量读取敏感网络配置，不在源码中保存默认值。"""
    value = config.get(key) or os.environ.get(env_name)
    if not value:
        raise ValueError(f"{key} must be set in config or {env_name}")
    return str(value)


class ZMapConsumer:
    """Consumer：复用 zmapv6_runtime 的绝对路径、接口和动态库配置。"""

    def __init__(self, config: dict[str, Any], scan_rate_bps: int):
        self.config = dict(config)
        self.scan_rate_bps = scan_rate_bps
        scanner_name = str(self.config.get("scanner", ""))
        probe_module = str(self.config.get("probe_module", ""))
        is_tcp = scanner_name == "tcp443" or probe_module == "ipv6_tcp_synscan"

        if is_tcp:
            try:
                from trap_det import trap_tcp as scanner_module
            except ImportError:
                from .trap_det import trap_tcp as scanner_module
        else:
            try:
                from trap_det import trap as scanner_module
            except ImportError:
                from .trap_det import trap as scanner_module

        # 以 128 字节作为保守线速包长，并配合批次节奏限制平均 pps。
        estimated_bandwidth = scan_rate_bps * 128 * 8
        zmap_path = (
            self.config.get("zmap_path")
            or os.environ.get("GUNGNIR_ZMAP_PATH")
            or DEFAULT_ZMAP
        )
        install_zmapv6(
            scanner_module,
            interface=_required_setting(
                self.config, "interface", "GUNGNIR_INTERFACE"
            ),
            zmap_path=str(zmap_path),
            gateway_mac=_required_setting(
                self.config, "gateway_mac", "GUNGNIR_GATEWAY_MAC"
            ),
            bandwidth=str(estimated_bandwidth),
        )
        self._start_scanning = scanner_module.startScaning

    def scan(self, batch: ProbeBatch) -> set[str]:
        scan_config = dict(self.config)
        scan_config["generated_address_path"] = str(batch.address_path)
        scan_config["zmap_result_path"] = str(batch.result_path)
        scan_config["local_ipv6"] = _required_setting(
            scan_config, "local_ipv6", "GUNGNIR_LOCAL_IPV6"
        )
        scan_config["passport"] = _required_setting(
            scan_config, "passport", "GUNGNIR_SCAN_PASSWORD"
        )

        started = time.monotonic()
        self._start_scanning(scan_config)

        # 即使扫描器提前返回，批次平均速率也不超过 scan_rate_bps（pp/s）。
        minimum_duration = batch.address_count / self.scan_rate_bps
        remaining_sleep = minimum_duration - (time.monotonic() - started)
        if remaining_sleep > 0:
            time.sleep(remaining_sleep)

        if not batch.result_path.exists():
            raise RuntimeError(f"ZMapv6 result file not found: {batch.result_path}")

        with batch.result_path.open("r", encoding="utf-8") as result_file:
            return {
                _canonical_address(line)
                for line in result_file
                if line.strip()
            }


class AdaptiveFrpPipeline:
    """按层执行 Producer -> Consumer -> 阈值剪枝。"""

    def __init__(
        self,
        config: dict[str, Any],
        response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
        scan_rate_bps: int = DEFAULT_SCAN_RATE_PPS,
        consumer: Consumer | None = None,
    ):
        if not 0.0 <= response_threshold <= 1.0:
            raise ValueError("response_threshold must be between 0 and 1")
        if scan_rate_bps <= 0:
            raise ValueError("scan_rate_bps must be a positive packets/sec value")

        self.config = dict(config)
        self.response_threshold = response_threshold
        self.scan_rate_bps = scan_rate_bps
        generated_dir = Path(
            self.config.get("generated_folder_path", "scanningdata/generated_address")
        )
        result_dir = Path(
            self.config.get("zmap_folder_path", "scanningdata/zmap_result")
        )
        self.producer = AddressProducer(generated_dir, result_dir)
        self.consumer = consumer or ZMapConsumer(self.config, scan_rate_bps)

    def run(self, prefix_set: set[str] | list[str] | tuple[str, ...]) -> set[str]:
        networks = {
            ipaddress.IPv6Network(prefix.strip(), strict=False)
            for prefix in prefix_set
            if prefix and prefix.strip()
        }
        ordered = sorted(
            networks,
            key=lambda network: (network.prefixlen, int(network.network_address)),
        )
        roots = [str(network) for network in ordered]
        if not roots:
            print("to be checked FRP number: 0")
            print("found FRP number: 0")
            return set()

        active = [Branch(root=root, prefix=root) for root in roots]
        frp_set: set[str] = set()

        for depth in range(MAX_LAYER + 1):
            if not active:
                break

            batch = self.producer.produce(active, depth)
            active_addresses = self.consumer.scan(batch)
            next_active: list[Branch] = []

            for unit in batch.units:
                addresses = unit.addresses
                response_count = sum(
                    address in active_addresses for address in addresses
                )
                address_count = len(addresses)
                response_rate = response_count / address_count
                meets_threshold = response_rate >= self.response_threshold
                all_responded = response_count == address_count
                full_sixteen = address_count == 16 and response_count == 16

                # 默认返回口径保持原 checkFrp：输入前缀首层全部地址响应才是 FRP。
                if depth == 0 and all_responded:
                    frp_set.add(unit.branch.root)

                # 达到阈值但不足 16/16 时停止；只有 16/16 才向下一层扩展。
                expand = meets_threshold and full_sixteen and depth < MAX_LAYER
                pruned = not meets_threshold
                print(
                    f"layer: {depth}, subprefix: {unit.branch.prefix}, "
                    f"to be checked address number: {address_count}"
                )
                print(
                    f"layer: {depth}, subprefix: {unit.branch.prefix}, "
                    f"found address number: {response_count}, "
                    f"response rate: {response_rate:.2%}, "
                    f"continue expansion: {'yes' if expand else 'no'}, "
                    f"pruned: {'yes' if pruned else 'no'}"
                )

                if expand:
                    children = _child_prefixes(unit.branch.prefix, addresses)
                    next_active.extend(
                        Branch(root=unit.branch.root, prefix=child)
                        for child in children
                    )

            active = next_active

        print(f"to be checked FRP number: {len(roots)}")
        print(f"found FRP number: {len(frp_set)}")
        return frp_set


def checkFrp(
    config: dict[str, Any],
    prefix_set: set[str] | list[str] | tuple[str, ...],
    response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
    scan_rate_bps: int = DEFAULT_SCAN_RATE_PPS,
) -> set[str]:
    """自适应探测入口；scan_rate_bps 按需求解释为 packets/sec。

    response_threshold 控制逐层剪枝，输入前缀是否进入返回集合仍由首层
    全响应决定，因此 checkFrp(config, prefix_set) 保持原有 16/16 判定语义。
    对 /128 等不足 16 个地址的前缀，与原代码一样要求实际地址全部响应。
    """
    pipeline = AdaptiveFrpPipeline(
        config=config,
        response_threshold=response_threshold,
        scan_rate_bps=scan_rate_bps,
    )
    return pipeline.run(prefix_set)
