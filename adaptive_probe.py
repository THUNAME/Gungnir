"""Single-round adaptive-threshold FRP probing with a bounded ZMap pipeline."""
from __future__ import annotations

import ipaddress
import math
import os
import secrets
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from zmapv6_runtime import DEFAULT_ZMAP, make_zmapv6_scanner
except ImportError:
    from .zmapv6_runtime import DEFAULT_ZMAP, make_zmapv6_scanner


DEFAULT_RESPONSE_THRESHOLD = 0.75
DEFAULT_SCAN_RATE_PPS = 5000
DEFAULT_BGP_RATE_PPS = 10.0
DEFAULT_BATCH_TARGETS = 100000
DEFAULT_AS_RATE_PPS = 0.0
DEFAULT_ZMAP_WORKERS = 2
ROOT = Path(__file__).resolve().parent
_PREFIX_MASKS = tuple(
    ((1 << length) - 1) << (128 - length) for length in range(129)
)


@dataclass(frozen=True)
class Branch:
    root: str
    prefix: str


@dataclass(frozen=True)
class ProbeUnit:
    branch: Branch
    addresses: tuple[str, ...]


@dataclass(frozen=True)
class Target:
    root: str
    address: str
    bgp_key: str
    as_key: str


@dataclass
class ProbeBatch:
    depth: int
    units: tuple[ProbeUnit, ...]
    address_path: Path
    result_path: Path
    targets: tuple[Target, ...] = ()
    batch_id: int = 0
    worker_id: int = 0
    rate_pps: int = 1
    window_seconds: float = 0.0
    submitted_at: float = 0.0

    @property
    def address_count(self) -> int:
        if self.targets:
            return len(self.targets)
        return sum(len(unit.addresses) for unit in self.units)


class Consumer(Protocol):
    def scan(self, batch: ProbeBatch) -> set[str]:
        """Scan one batch and return canonical responsive addresses."""


@dataclass
class PrefixState:
    root: str
    network_int: int
    prefix_len: int
    total: int
    required: int
    sent: int = 0
    hits: int = 0
    next_address: str | None = None
    next_bgp_key: str | None = None
    next_as_key: str | None = None


def _prefix_parts(prefix: str) -> tuple[int, int]:
    address, length_text = prefix.strip().rsplit("/", 1)
    prefix_len = int(length_text)
    if not 0 <= prefix_len <= 128:
        raise ValueError(f"invalid IPv6 prefix length: {prefix!r}")
    network_int = int(ipaddress.IPv6Address(address)) & _PREFIX_MASKS[prefix_len]
    return network_int, prefix_len


def _canonical_prefix(prefix: str) -> str:
    network_int, prefix_len = _prefix_parts(prefix)
    return f"{ipaddress.IPv6Address(network_int)}/{prefix_len}"


def _canonical_address(address: str) -> str:
    return str(ipaddress.IPv6Address(address.strip()))


def _sample_count(prefix_len: int) -> int:
    return min(16, 1 << (128 - prefix_len))


def _sample_address(network_int: int, prefix_len: int, index: int) -> str:
    host_bits = 128 - prefix_len
    if not 0 <= index < _sample_count(prefix_len):
        raise IndexError("sample index exceeds the prefix address space")
    if host_bits < 4:
        value = network_int + index
    else:
        value = network_int | (index << (host_bits - 4))
        if host_bits > 4:
            value |= secrets.randbits(host_bits - 4)
    return str(ipaddress.IPv6Address(value))


def _generate_addresses(prefix: str) -> tuple[str, ...]:
    """Compatibility helper using the same on-demand integer sampler."""
    network_int, prefix_len = _prefix_parts(prefix)
    return tuple(
        _sample_address(network_int, prefix_len, index)
        for index in range(_sample_count(prefix_len))
    )


def _required_count(sample_count: int, threshold: float) -> int:
    return max(1, math.ceil(sample_count * threshold))


def _read_active_addresses(path: Path) -> set[str]:
    active = set()
    with path.open("r", encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, 1):
            value = line.strip()
            if not value or value == "saddr":
                continue
            try:
                active.add(_canonical_address(value))
            except ValueError as exc:
                raise RuntimeError(
                    f"{path}:{line_number} is not a valid ZMap saddr row: {value!r}"
                ) from exc
    return active


def _required_setting(config: dict[str, Any], key: str, env_name: str) -> str:
    value = config.get(key) or os.environ.get(env_name)
    if not value:
        raise ValueError(f"{key} must be set in config or {env_name}")
    return str(value)


class AddressProducer:
    """Write bounded batches while scans and result processing run elsewhere."""

    def __init__(self, generated_dir: Path, result_dir: Path):
        self.generated_dir = generated_dir
        self.result_dir = result_dir
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def produce(self, branches: list[Branch], depth: int) -> ProbeBatch:
        """Retain the original single-batch producer as a compatibility API."""
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
        return ProbeBatch(depth, units, address_path, result_path)

    def produce_targets(
        self,
        targets: list[Target],
        batch_id: int,
        worker_id: int,
        rate_pps: int,
        window_seconds: float,
    ) -> ProbeBatch:
        stem = f"adaptive_{time.time_ns()}_{batch_id}_{worker_id}"
        address_path = self.generated_dir / f"{stem}.txt"
        result_path = self.result_dir / f"{stem}.txt"
        with address_path.open("w", encoding="utf-8") as output:
            for target in targets:
                output.write(target.address + "\n")
        return ProbeBatch(
            depth=0,
            units=(),
            address_path=address_path,
            result_path=result_path,
            targets=tuple(targets),
            batch_id=batch_id,
            worker_id=worker_id,
            rate_pps=rate_pps,
            window_seconds=window_seconds,
        )


class ZMapConsumer:
    """A single-worker consumer with an isolated ZMap scanner closure."""

    def __init__(self, config: dict[str, Any], scan_rate_bps: int, worker_id: int = 0):
        self.config = dict(config)
        self.scan_rate_bps = scan_rate_bps
        self.worker_id = worker_id
        scanner_name = str(self.config.get("scanner", ""))
        probe_module = str(self.config.get("probe_module", ""))
        is_tcp = scanner_name == "tcp443" or probe_module == "ipv6_tcp_synscan"
        zmap_path = (
            self.config.get("zmap_path")
            or os.environ.get("GUNGNIR_ZMAP_PATH")
            or DEFAULT_ZMAP
        )
        self._start_scanning = make_zmapv6_scanner(
            interface=_required_setting(
                self.config, "interface", "GUNGNIR_INTERFACE"
            ),
            zmap_path=str(zmap_path),
            gateway_mac=_required_setting(
                self.config, "gateway_mac", "GUNGNIR_GATEWAY_MAC"
            ),
            rate_pps=scan_rate_bps,
            is_tcp=is_tcp,
        )

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
        scan_config["rate_pps"] = batch.rate_pps or self.scan_rate_bps

        started = time.monotonic()
        try:
            self._start_scanning(scan_config)
            minimum_duration = batch.window_seconds or (
                batch.address_count / scan_config["rate_pps"]
            )
            remaining_sleep = minimum_duration - (time.monotonic() - started)
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)
            if not batch.result_path.exists():
                raise RuntimeError(f"ZMapv6 result file not found: {batch.result_path}")
            return _read_active_addresses(batch.result_path)
        finally:
            for path in (batch.address_path, batch.result_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


class AdaptiveFrpPipeline:
    """Work-conserving, shared-window, single-round FRP classification."""

    def __init__(
        self,
        config: dict[str, Any],
        response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
        scan_rate_bps: int = DEFAULT_SCAN_RATE_PPS,
        consumer: Consumer | None = None,
        bgp_rate_pps: float = DEFAULT_BGP_RATE_PPS,
        batch_targets: int = DEFAULT_BATCH_TARGETS,
        as_rate_pps: float = DEFAULT_AS_RATE_PPS,
        zmap_workers: int = DEFAULT_ZMAP_WORKERS,
        early_stop: bool = True,
    ):
        if not 0.0 < response_threshold <= 1.0:
            raise ValueError("response_threshold must be in (0, 1]")
        numeric_rates = (scan_rate_bps, bgp_rate_pps, as_rate_pps)
        if not all(math.isfinite(value) for value in numeric_rates):
            raise ValueError("rate values must be finite")
        if scan_rate_bps <= 0 or bgp_rate_pps <= 0 or as_rate_pps < 0:
            raise ValueError("global/BGP rates must be positive and AS rate non-negative")
        if batch_targets <= 0:
            raise ValueError("batch_targets must be positive")
        if zmap_workers not in (1, 2):
            raise ValueError("zmap_workers must be 1 or 2")
        if consumer is not None and zmap_workers != 1:
            zmap_workers = 1

        self.config = dict(config)
        self.response_threshold = response_threshold
        self.scan_rate_bps = scan_rate_bps
        self.bgp_rate_pps = bgp_rate_pps
        self.batch_targets = batch_targets
        self.as_rate_pps = as_rate_pps
        self.early_stop = early_stop
        self.window_seconds = max(
            batch_targets / scan_rate_bps,
            1 / bgp_rate_pps,
            1 / as_rate_pps if as_rate_pps else 0.0,
        )
        generated_dir = Path(
            self.config.get("generated_folder_path", "scanningdata/generated_address")
        )
        result_dir = Path(
            self.config.get("zmap_folder_path", "scanningdata/zmap_result")
        )
        self.producer = AddressProducer(generated_dir, result_dir)
        if consumer is not None:
            self.consumers = [consumer]
        else:
            self.consumers = [
                ZMapConsumer(self.config, scan_rate_bps, worker_id=index)
                for index in range(zmap_workers)
            ]
        self.consumer = self.consumers[0]
        self.bgp_db = self.config.get("bgp_db") or self._load_bgp_db()
        self.states: dict[str, PrefixState] = {}
        self.groups: dict[str, deque[str]] = {}
        self._next_batch_id = 0
        self._total_probes = 0
        self._early_accept = 0
        self._early_reject = 0
        self._frp_set: set[str] = set()

    def _load_bgp_db(self):
        try:
            import pyasn
        except ImportError as exc:
            raise RuntimeError("per-BGP pacing requires pyasn") from exc
        database = self.config.get("bgp_dat", str(ROOT / "data" / "latestv6.dat"))
        return pyasn.pyasn(str(database))

    def _enqueue(self, root: str) -> None:
        state = self.states[root]
        if state.sent >= state.total:
            return
        address = _sample_address(state.network_int, state.prefix_len, state.sent)
        asn, bgp_prefix = self.bgp_db.lookup(address)
        state.next_address = address
        state.next_bgp_key = bgp_prefix or f"unmatched-bgp:{root}"
        state.next_as_key = (
            str(asn) if asn is not None else f"unmatched-as:{root}"
        )
        self.groups.setdefault(state.next_bgp_key, deque()).append(root)

    def _has_pending(self) -> bool:
        return any(self.groups.values())

    def _take_targets(
        self,
        capacity: int,
        bgp_quota: int,
        bgp_remaining: dict[str, int],
        as_quota: int | None,
        as_remaining: dict[str, int],
    ) -> list[Target]:
        records = []
        keys = deque(self.groups)
        while keys and len(records) < capacity:
            bgp_key = keys.popleft()
            group = self.groups.get(bgp_key)
            if not group:
                continue
            if bgp_remaining.get(bgp_key, bgp_quota) <= 0:
                continue
            state = self.states[group[0]]
            as_key = state.next_as_key
            if (
                as_quota is not None
                and as_remaining.get(as_key, as_quota) <= 0
            ):
                continue

            group.popleft()
            target = Target(
                root=state.root,
                address=state.next_address,
                bgp_key=state.next_bgp_key,
                as_key=as_key,
            )
            state.sent += 1
            state.next_address = None
            state.next_bgp_key = None
            state.next_as_key = None
            records.append(target)
            bgp_remaining[bgp_key] = bgp_remaining.get(bgp_key, bgp_quota) - 1
            if as_quota is not None:
                as_remaining[as_key] = as_remaining.get(as_key, as_quota) - 1
            if group:
                keys.append(bgp_key)
        return records

    def _produce_window(self, worker_ids: list[int]) -> list[ProbeBatch]:
        global_remaining = max(1, int(self.scan_rate_bps * self.window_seconds))
        bgp_quota = max(1, int(self.bgp_rate_pps * self.window_seconds))
        as_quota = (
            max(1, int(self.as_rate_pps * self.window_seconds))
            if self.as_rate_pps else None
        )
        bgp_remaining: dict[str, int] = {}
        as_remaining: dict[str, int] = {}
        batches = []

        for worker_id in worker_ids:
            if global_remaining <= 0 or not self._has_pending():
                break
            records = self._take_targets(
                min(self.batch_targets, global_remaining),
                bgp_quota,
                bgp_remaining,
                as_quota,
                as_remaining,
            )
            if not records:
                break
            rate_pps = max(1, int(len(records) / self.window_seconds))
            batch = self.producer.produce_targets(
                targets=records,
                batch_id=self._next_batch_id,
                worker_id=worker_id,
                rate_pps=min(self.scan_rate_bps, rate_pps),
                window_seconds=self.window_seconds,
            )
            self._next_batch_id += 1
            batches.append(batch)
            global_remaining -= len(records)
        return batches

    def _process_future(self, future: Future, futures: dict[Future, ProbeBatch]) -> None:
        batch = futures.pop(future)
        active = future.result()
        scheduled = set()
        for target in batch.targets:
            scheduled.add(target.root)
            if target.address in active:
                self.states[target.root].hits += 1

        for root in scheduled:
            state = self.states[root]
            remaining = state.total - state.sent
            complete = state.sent == state.total
            enough = state.hits >= state.required
            impossible = state.hits + remaining < state.required
            if self.early_stop and enough:
                self._frp_set.add(root)
                self._early_accept += 1
            elif self.early_stop and impossible:
                self._early_reject += 1
            elif complete:
                if enough:
                    self._frp_set.add(root)
            else:
                self._enqueue(root)

        pending_prefixes = sum(
            state.next_address is not None for state in self.states.values()
        )
        pending_samples = sum(
            state.total - state.sent
            for state in self.states.values()
            if state.next_address is not None
        )
        print(
            f"Online batch {batch.batch_id}: worker={batch.worker_id}; "
            f"targets={batch.address_count}; rate={batch.rate_pps} pps; "
            f"elapsed={time.monotonic() - batch.submitted_at:.3f}s; "
            f"pending_prefixes={pending_prefixes}; pending_samples={pending_samples}; "
            f"eligible_bgp={sum(bool(group) for group in self.groups.values())}",
            flush=True,
        )

    def run(self, prefix_set: set[str] | list[str] | tuple[str, ...]) -> set[str]:
        roots = set()
        for prefix in prefix_set:
            if prefix and prefix.strip():
                network_int, prefix_len = _prefix_parts(prefix)
                roots.add((prefix_len, network_int))
        ordered = sorted(roots)
        if not ordered:
            print("to be checked FRP number: 0")
            print("found FRP number: 0")
            return set()

        self.states = {}
        self.groups = {}
        self._frp_set = set()
        self._total_probes = 0
        self._early_accept = 0
        self._early_reject = 0
        self._next_batch_id = 0
        for prefix_len, network_int in ordered:
            root = f"{ipaddress.IPv6Address(network_int)}/{prefix_len}"
            total = _sample_count(prefix_len)
            self.states[root] = PrefixState(
                root=root,
                network_int=network_int,
                prefix_len=prefix_len,
                total=total,
                required=_required_count(total, self.response_threshold),
            )
            self._enqueue(root)

        futures: dict[Future, ProbeBatch] = {}
        next_window = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self.consumers)) as executor:
            while self._has_pending() or futures:
                done = [future for future in futures if future.done()]
                if done:
                    for future in done:
                        self._process_future(future, futures)
                    continue

                busy_workers = {batch.worker_id for batch in futures.values()}
                free_workers = [
                    index for index in range(len(self.consumers))
                    if index not in busy_workers
                ]
                if free_workers and self._has_pending():
                    now = time.monotonic()
                    if now < next_window:
                        timeout = next_window - now
                        if futures:
                            done, _ = wait(
                                futures,
                                timeout=timeout,
                                return_when=FIRST_COMPLETED,
                            )
                            if done:
                                for future in done:
                                    self._process_future(future, futures)
                                continue
                        else:
                            time.sleep(timeout)
                        if time.monotonic() < next_window:
                            continue

                    batches = self._produce_window(free_workers)
                    if not batches:
                        raise RuntimeError("scheduler has pending states but produced no batch")
                    # 文件生成不计入窗口间隔，避免下一批提前发送并叠加速率。
                    window_started = time.monotonic()
                    for batch in batches:
                        batch.submitted_at = window_started
                        future = executor.submit(
                            self.consumers[batch.worker_id].scan, batch
                        )
                        futures[future] = batch
                        self._total_probes += batch.address_count
                    next_window = window_started + self.window_seconds
                    continue

                if futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        self._process_future(future, futures)
                elif self._has_pending():
                    raise RuntimeError("no worker available for pending states")

        print(f"to be checked FRP number: {len(self.states)}")
        print(f"found FRP number: {len(self._frp_set)}")
        print(
            f"Online scan: batches={self._next_batch_id}; probes={self._total_probes}; "
            f"early_accept={self._early_accept}; early_reject={self._early_reject}; "
            f"workers={len(self.consumers)}; window={self.window_seconds:.3f}s"
        )
        return self._frp_set


def checkFrp(
    config: dict[str, Any],
    prefix_set: set[str] | list[str] | tuple[str, ...],
    response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
    scan_rate_bps: int = DEFAULT_SCAN_RATE_PPS,
    *,
    bgp_rate_pps: float = DEFAULT_BGP_RATE_PPS,
    batch_targets: int = DEFAULT_BATCH_TARGETS,
    as_rate_pps: float = DEFAULT_AS_RATE_PPS,
    zmap_workers: int = DEFAULT_ZMAP_WORKERS,
    early_stop: bool = True,
) -> set[str]:
    """Classify each input prefix once without expanding any child prefix."""
    pipeline = AdaptiveFrpPipeline(
        config=config,
        response_threshold=response_threshold,
        scan_rate_bps=scan_rate_bps,
        bgp_rate_pps=bgp_rate_pps,
        batch_targets=batch_targets,
        as_rate_pps=as_rate_pps,
        zmap_workers=zmap_workers,
        early_stop=early_stop,
    )
    return pipeline.run(prefix_set)
