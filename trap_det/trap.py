import ipaddress
import json
import math
import os
import secrets
import subprocess
import argparse
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def startScaning(config):
    protocol = config.get('protocol', 'icmp')
    port = config.get('port', 0)
    metadata_path = config['zmap_result_path'] + ".metadata.json"

    cmd = [
        "sudo", "-n", "zmap",
        "-i", config.get('interface', 'eth0'),
        f"--ipv6-source-ip={config['local_ipv6']}",
        f"--ipv6-target-file={config['generated_address_path']}",
        "-o", config['zmap_result_path'],
        "--rate", str(config.get('rate', 10000)),
        "--verbosity=0",
        "--disable-syslog",
        "--output-module=csv",
        "--output-fields=saddr",
        "--output-filter=success=1 && repeat=0",
        f"--metadata-file={metadata_path}"
    ]

    if protocol == 'icmp':
        cmd.extend(["-M", "icmp6_echoscan"])
    elif protocol == 'tcp':
        cmd.extend(["-M", "ipv6_tcp_synscan", "-p", str(port)])
    elif protocol == 'udp' and port == 53:
        cmd.extend([
            "-M", "ipv6_dns", "-p", "53",
            "--probe-args", f"name:{config['dns_domain']}"
        ])
    elif protocol == 'udp':
        cmd.extend(["-M", "ipv6_udp", "-p", str(port)])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-4000:].strip()
            raise RuntimeError(
                f"zmap 扫描失败（rc={proc.returncode}）：{stderr_tail}"
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        hit_rate = metadata.get('hitrate')
        if hit_rate is None:
            raise RuntimeError("zmap metadata 中缺少 hitrate")

        print(f"Hit rate: {hit_rate:.2f}%")
        return hit_rate
    finally:
        try:
            os.remove(metadata_path)
        except FileNotFoundError:
            pass


def add_ipv6_colon(ipv6):
    return ":".join(ipv6[i:i+4] for i in range(0, len(ipv6), 4))


def prefix_to_sixteen(prefix):
    address, length = prefix.rsplit("/", 1)
    length = int(length)
    network_int = int(ipaddress.IPv6Address(address)) & _PREFIX_MASKS[length]
    host_bits = 128 - length
    count = min(16, 1 << host_bits)

    ipv6s = []
    if host_bits >= 4:
        for i in range(16):
            value = network_int | (i << (host_bits - 4))
            if host_bits > 4:
                value |= secrets.randbits(host_bits - 4)
            ipv6s.append(ipaddress.IPv6Address(value).exploded)
    else:
        # 小地址空间直接枚举，避免重复采样及 /128 的负随机位数。
        for i in range(count):
            ipv6s.append(ipaddress.IPv6Address(network_int + i).exploded)
    return ipv6s


def required_active_count(sample_count, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("全响应阈值必须在 (0, 1] 范围内")
    return max(1, math.ceil(sample_count * threshold))


def is_full_response(ipv6s, active, threshold):
    active_count = sum(ip in active for ip in ipv6s)
    return active_count >= required_active_count(len(ipv6s), threshold)


def read_active_addresses(path):
    active = set()
    with open(path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            value = line.strip()
            if not value or value == 'saddr':
                continue
            try:
                active.add(ipaddress.IPv6Address(value).exploded)
            except ValueError as exc:
                raise RuntimeError(
                    f"{path}:{line_number} 不是合法的 ZMap saddr 输出：{value!r}"
                ) from exc
    return active


def scan_single_prefix(config, prefix):
    """每个前缀使用独立 ZMap 进程，确保其 --rate 只作用于当前前缀。"""
    ipv6s = prefix_to_sixteen(prefix)
    gen_tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=config['generated_folder_path'], suffix=".txt"
    )
    zmap_tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=config['zmap_folder_path'], suffix=".txt"
    )
    gen_tmp.close()
    zmap_tmp.close()

    local_config = config.copy()
    local_config['generated_address_path'] = gen_tmp.name
    local_config['zmap_result_path'] = zmap_tmp.name
    local_config['rate'] = config['per_prefix_rate']

    try:
        with open(gen_tmp.name, 'w') as f:
            for ip in ipv6s:
                f.write(ip + "\n")

        startScaning(local_config)
        active = read_active_addresses(zmap_tmp.name)
        return is_full_response(
            ipv6s, active, config.get('response_threshold', 1.0)
        )
    finally:
        for path in (gen_tmp.name, zmap_tmp.name):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def check_frp_per_prefix(config, prefix_set):
    """以有界并发运行逐前缀扫描，最大聚合速率为 workers × per_prefix_rate。"""
    prefixes = iter(prefix_set)
    total = len(prefix_set)
    workers = config['prefix_workers']
    FRP_set = set()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for _ in range(workers):
            try:
                prefix = next(prefixes)
            except StopIteration:
                break
            pending[executor.submit(scan_single_prefix, config, prefix)] = prefix

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                prefix = pending.pop(future)
                if future.result():
                    FRP_set.add(prefix)
                try:
                    next_prefix = next(prefixes)
                except StopIteration:
                    continue
                pending[
                    executor.submit(scan_single_prefix, config, next_prefix)
                ] = next_prefix

    print("to be checked FRP number:", total)
    print("found FRP number:", len(FRP_set))
    return FRP_set


def _route_keys(config, prefix, address):
    asn, bgp_prefix = config['bgp_db'].lookup(address)
    bgp_key = bgp_prefix or f"unmatched-bgp:{prefix}"
    as_key = asn if asn is not None else f"unmatched-as:{prefix}"
    return bgp_key, as_key


def _interleave_rate_groups(records):
    """先交错 AS 内的 BGP，再交错各 AS，减少同组目标连续突发。"""
    as_groups = {}
    for record in records:
        bgp_key, as_key = record[2], record[3]
        as_groups.setdefault(as_key, {}).setdefault(bgp_key, []).append(record)

    as_streams = []
    for bgp_groups in as_groups.values():
        streams = list(bgp_groups.values())
        stream = []
        for index in range(max(map(len, streams))):
            for targets in streams:
                if index < len(targets):
                    stream.append(targets[index])
        as_streams.append(stream)

    interleaved = []
    for index in range(max(map(len, as_streams))):
        for stream in as_streams:
            if index < len(stream):
                interleaved.append(stream[index])
    return interleaved


DEFAULT_BATCH_WINDOW = 20.0


def _batch_window(config):
    # 全局速率设为 0 时仍使用固定 20 秒配额窗口；聚合速率由各 BGP 配额自然形成。
    if config['batch_rate'] and config['batch_targets']:
        global_window = config['batch_targets'] / config['batch_rate']
    else:
        global_window = DEFAULT_BATCH_WINDOW
    window = max(global_window, 1 / config['bgp_rate'])
    if config.get('as_rate'):
        window = max(window, 1 / config['as_rate'])
    return window


def scan_target_batch(config, records):
    gen_tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=config['generated_folder_path'], suffix=".txt"
    )
    zmap_tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=config['zmap_folder_path'], suffix=".txt"
    )
    gen_tmp.close()
    zmap_tmp.close()
    local_config = config.copy()
    local_config['generated_address_path'] = gen_tmp.name
    local_config['zmap_result_path'] = zmap_tmp.name
    # 稀疏批次同步降速；未设置全局速率时，速率完全由本批目标数与时间窗决定。
    paced_rate = max(1, int(len(records) / _batch_window(config)))
    local_config['rate'] = (
        min(config['batch_rate'], paced_rate)
        if config['batch_rate'] else paced_rate
    )

    try:
        with open(gen_tmp.name, 'w') as f:
            for _, address, _, _ in records:
                f.write(address + "\n")
        startScaning(local_config)
        return read_active_addresses(zmap_tmp.name)
    finally:
        for path in (gen_tmp.name, zmap_tmp.name):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _take_rate_limited_batch(config, states, queue):
    batch_targets = config['batch_targets']
    window = _batch_window(config)
    global_quota = (
        max(1, int(config['batch_rate'] * window))
        if config['batch_rate'] else None
    )
    limits = [
        value for value in (batch_targets or None, global_quota)
        if value is not None
    ]
    batch_limit = min(limits) if limits else None
    bgp_quota = max(1, int(config['bgp_rate'] * window))
    as_quota = (
        max(1, int(config['as_rate'] * window))
        if config.get('as_rate') else None
    )
    bgp_counts = {}
    as_counts = {}
    records = []
    deferred = []

    while queue and (batch_limit is None or len(records) < batch_limit):
        prefix = queue.popleft()
        state = states[prefix]
        address = state['samples'][state['sent']]
        bgp_key, as_key = _route_keys(config, prefix, address)
        if (
            bgp_counts.get(bgp_key, 0) >= bgp_quota
            or (
                as_quota is not None
                and as_counts.get(as_key, 0) >= as_quota
            )
        ):
            deferred.append(prefix)
            continue

        bgp_counts[bgp_key] = bgp_counts.get(bgp_key, 0) + 1
        as_counts[as_key] = as_counts.get(as_key, 0) + 1
        state['sent'] += 1
        records.append((prefix, address, bgp_key, as_key))

    queue.extendleft(reversed(deferred))
    return _interleave_rate_groups(records) if records else []


def check_frp_batch(config, prefix_set):
    """分层配额批量扫描，并按 75% 等阈值的可达结论提前停止。"""
    states = {}
    threshold = config.get('response_threshold', 1.0)
    for prefix in sorted(prefix_set):
        samples = prefix_to_sixteen(prefix)
        states[prefix] = {
            'samples': samples,
            'sent': 0,
            'hits': 0,
            'required': required_active_count(len(samples), threshold),
        }

    queue = deque(states)
    frp_set = set()
    early_accept = 0
    early_reject = 0
    batches = 0
    total_probes = 0
    window = _batch_window(config)

    while queue:
        records = _take_rate_limited_batch(config, states, queue)
        if not records:
            raise RuntimeError("批量调度器未能生成目标，状态机无法继续")
        batch_started = time.monotonic()
        active = scan_target_batch(config, records)
        batches += 1
        total_probes += len(records)

        scheduled = set()
        for prefix, address, _, _ in records:
            scheduled.add(prefix)
            if address in active:
                states[prefix]['hits'] += 1

        for prefix in scheduled:
            state = states[prefix]
            total = len(state['samples'])
            remaining = total - state['sent']
            complete = state['sent'] == total
            enough = state['hits'] >= state['required']
            impossible = state['hits'] + remaining < state['required']
            if config['early_stop'] and enough:
                frp_set.add(prefix)
                early_accept += 1
            elif config['early_stop'] and impossible:
                early_reject += 1
            elif complete:
                if enough:
                    frp_set.add(prefix)
            else:
                queue.append(prefix)

        elapsed = time.monotonic() - batch_started
        if elapsed < window:
            time.sleep(window - elapsed)

    print(
        f"Batch scan: batches={batches}; probes={total_probes}; "
        f"early_accept={early_accept}; early_reject={early_reject}; "
        f"FRP={len(frp_set)}"
    )
    return frp_set


def checkFrp(config, prefix_set):
    if not prefix_set:
        raise ValueError("没有可扫描的前缀")
    if config.get('per_prefix_rate') is not None:
        return check_frp_per_prefix(config, prefix_set)
    # 默认使用批量分层调度；batch_rate=0 仅关闭全局速率上限，不再关闭批量模式。
    return check_frp_batch(config, prefix_set)

    # 每次生成两条新的临时文件路径 —— 始终覆盖旧的，不删除 config 的 key
    gen_tmp = tempfile.NamedTemporaryFile(delete=False, dir=config['generated_folder_path'], suffix=".txt")
    zmap_tmp = tempfile.NamedTemporaryFile(delete=False, dir=config['zmap_folder_path'], suffix=".txt")

    gen_tmp.close()
    zmap_tmp.close()

    # 永远写回 key，确保一定存在
    config['generated_address_path'] = gen_tmp.name
    config['zmap_result_path'] = zmap_tmp.name

    Frp_dict = {}
    for prefix in prefix_set:
        try:
            Frp_dict[prefix] = prefix_to_sixteen(prefix)
        except Exception as exc:
            raise ValueError(f"无法为前缀 {prefix} 生成扫描地址：{exc}") from exc

    # 写入目标地址文件
    with open(config['generated_address_path'], 'w') as f:
        for ipv6s in Frp_dict.values():
            for ip in ipv6s:
                f.write(ip + "\n")

    startScaning(config)

    # 读取 zmap 输出
    active = read_active_addresses(config['zmap_result_path'])

    # 计算 FRP
    threshold = config.get('response_threshold', 1.0)
    FRP_set = {
        prefix for prefix, ipv6s in Frp_dict.items()
        if is_full_response(ipv6s, active, threshold)
    }

    print("to be checked FRP number:", len(prefix_set))
    print("found FRP number:", len(FRP_set))

    # -----------------------------
    # 🔥 删除临时文件（只删除文件，不删除 config key）
    # -----------------------------
    try:
        os.remove(config['generated_address_path'])
    except FileNotFoundError:
        pass

    try:
        os.remove(config['zmap_result_path'])
    except FileNotFoundError:
        pass

    return FRP_set


def save_to_file(FRP_set, config):
    with open(config['outputfrp'], 'a') as f:
        for line in FRP_set:
            f.write(line + "\n")


_PREFIX_MASKS = tuple(
    ((1 << length) - 1) << (128 - length) for length in range(129)
)


def make_jump_prefix(prefix, trap_len):
    """用整数掩码直接生成规范父网，避免逐分支构造 IPv6Network。"""
    if not isinstance(trap_len, int):
        raise TypeError(f"前缀 {prefix} 的跳转步长不是整数：{trap_len!r}")
    if trap_len < 1:
        raise ValueError(f"前缀 {prefix} 的跳转步长必须为正数")

    addr, length = prefix.rsplit("/", 1)
    new_len = int(length) - trap_len
    if not 0 <= new_len <= 128:
        raise ValueError(f"前缀 {prefix} 跳转后长度无效：{new_len}")

    network_int = int(ipaddress.IPv6Address(addr)) & _PREFIX_MASKS[new_len]
    return f"{ipaddress.IPv6Address(network_int)}/{new_len}"


def jump_outoftrap(prefix_dict, total_budget, config):
    print("to be checked ", len(prefix_dict))

    # 同一规范父网只扫描一次，同时保留全部来源分支用于独立回退。
    jump_groups = {}
    source_count = 0
    for prefix, trap_len in prefix_dict.items():
        prefix_len = int(prefix.rsplit('/', 1)[1])
        # 成功上移后步长可能大于当前前缀长度；越过 /0 的分支正常结束。
        if trap_len < 1 or trap_len > prefix_len:
            continue
        jump = make_jump_prefix(prefix, trap_len)
        jump_groups.setdefault(jump, []).append(prefix)
        source_count += 1

    if not jump_groups:
        return total_budget, {}

    print(
        "canonical parent groups:", len(jump_groups),
        "merged branches:", source_count - len(jump_groups)
    )
    FRP_set = checkFrp(config, jump_groups.keys())
    total_budget += len(jump_groups)
    save_to_file(FRP_set, config)

    next_prefix_dict = {}

    def retain_state(prefix, trap_len):
        if trap_len < 1:
            return
        previous = next_prefix_dict.get(prefix)
        next_prefix_dict[prefix] = (
            trap_len if previous is None else min(previous, trap_len)
        )

    for jump, sources in jump_groups.items():
        if jump in FRP_set:
            # 多分支收敛时采用最保守步长，避免跳过尚未验证的中间前缀。
            retain_state(jump, min(prefix_dict[source] for source in sources))
        else:
            for source in sources:
                retain_state(source, prefix_dict[source] // 2)

    return total_budget, next_prefix_dict


def initialize_prefix_states(predict_path, bgp_db, fallback_jump=8):
    """按目标长度与最长匹配 BGP 前缀长度之差初始化跳转步长。"""
    if not isinstance(fallback_jump, int) or not 1 <= fallback_jump <= 128:
        raise ValueError("回退跳转步长必须在 1..128 范围内")

    prefix_dict = {}
    stats = {'dynamic': 0, 'unmatched': 0, 'nonpositive': 0}
    with open(predict_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                network = ipaddress.ip_network(line, strict=False)
            except ValueError as exc:
                raise ValueError(f"{predict_path}:{line_number} 不是合法前缀：{line}") from exc
            if network.version != 6:
                raise ValueError(f"仅支持 IPv6 前缀：{line}")

            _, bgp_prefix = bgp_db.lookup(str(network.network_address))
            delta = 0
            if bgp_prefix is None:
                stats['unmatched'] += 1
            else:
                try:
                    bgp_len = int(bgp_prefix.rsplit('/', 1)[1])
                except (AttributeError, IndexError, ValueError) as exc:
                    raise ValueError(
                        f"BGP 查询返回非法前缀 {bgp_prefix!r}，目标为 {network}"
                    ) from exc
                if not 0 <= bgp_len <= 128:
                    raise ValueError(f"BGP 前缀长度无效 {bgp_prefix!r}，目标为 {network}")
                delta = network.prefixlen - bgp_len
                if delta <= 0:
                    stats['nonpositive'] += 1

            if delta > 0:
                prefix_dict[str(network)] = delta
                stats['dynamic'] += 1
            else:
                prefix_dict[str(network)] = fallback_jump

    if not prefix_dict:
        raise ValueError("输入文件没有有效的 IPv6 前缀")
    return prefix_dict, stats


def main():
    parser = argparse.ArgumentParser(description="FRP IPv6 Prefix Checker")

    parser.add_argument("--predict_path", type=str, required=True)
    parser.add_argument(
        "--bgp-dat", default=os.path.join(PROJECT_ROOT, "data", "latestv6.dat"),
        help="pyasn IPv6 BGP 数据库，用于计算动态初始跳转步长"
    )
    # 兼容尚未退出的旧编排进程；参数不再读取或用于 sudo。
    parser.add_argument("--passport", dest="legacy_passport", help=argparse.SUPPRESS)
    parser.add_argument("--local_ipv6", type=str, required=True)
    parser.add_argument("--generated_folder", type=str, required=True)
    parser.add_argument("--zmap_folder", type=str, required=True)
    parser.add_argument("--outputfrp", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="icmp", choices=["icmp", "tcp", "udp"])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--rate", type=int, default=10000, help="ZMap 发送速率，单位 pps")
    parser.add_argument("--response-threshold", type=float, default=1.0, help="全响应判定阈值，默认 1.0")
    parser.add_argument("--per-prefix-rate", type=int, help="每个前缀独立的 ZMap pps")
    parser.add_argument("--prefix-workers", type=int, default=4, help="逐前缀扫描的最大并发数")
    parser.add_argument(
        "--batch-rate", type=int, default=0,
        help="批量模式的全局最高 ZMap pps；0 表示不限全局速率，仅保留 BGP/AS 配额"
    )
    parser.add_argument("--bgp-rate", type=float, default=10, help="批量模式每 BGP 时间窗平均 pps")
    parser.add_argument(
        "--as-rate", type=float, default=0,
        help="批量模式每 AS 时间窗平均 pps；设为 0 表示不限速"
    )
    parser.add_argument(
        "--batch-targets", type=int, default=0,
        help="每批最大目标地址数；0 表示不限制，仅由 BGP/AS 窗口配额决定"
    )
    parser.add_argument(
        "--no-early-stop", action="store_true",
        help="批量模式发送全部样本，不在结论已确定时提前停止"
    )
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--dns-domain", default="example.com", help="UDP/53 DNS 查询域名")

    args = parser.parse_args()
    if args.protocol in ("tcp", "udp") and not 1 <= args.port <= 65535:
        parser.error("TCP/UDP 扫描必须指定 1..65535 的目标端口")
    if args.rate <= 0:
        parser.error("--rate 必须是正整数")
    if not 0 < args.response_threshold <= 1:
        parser.error("--response-threshold 必须在 (0, 1] 范围内")
    if args.per_prefix_rate is not None and args.per_prefix_rate <= 0:
        parser.error("--per-prefix-rate 必须是正整数")
    if args.prefix_workers <= 0:
        parser.error("--prefix-workers 必须是正整数")
    if args.batch_rate is not None and args.batch_rate < 0:
        parser.error("--batch-rate 必须是非负整数")
    if args.batch_rate and args.per_prefix_rate is not None:
        parser.error("--per-prefix-rate 模式请将 --batch-rate 设为 0")
    if (
        not math.isfinite(args.bgp_rate) or not math.isfinite(args.as_rate)
        or args.bgp_rate <= 0 or args.as_rate < 0
    ):
        parser.error("--bgp-rate 必须为正数，--as-rate 必须为非负数（0 表示不限速）")
    if args.batch_targets < 0:
        parser.error("--batch-targets 必须是非负整数（0 表示不限制）")
    if not args.dns_domain:
        parser.error("--dns-domain 不能为空")
    ipaddress.IPv6Address(args.local_ipv6)

    config = dict(
        predict_path=args.predict_path,
        local_ipv6=args.local_ipv6,
        generated_folder_path=args.generated_folder if args.generated_folder.endswith("/") else args.generated_folder + "/",
        zmap_folder_path=args.zmap_folder if args.zmap_folder.endswith("/") else args.zmap_folder + "/",
        outputfrp=args.outputfrp,
        protocol=args.protocol,
        port=args.port,
        rate=args.rate,
        response_threshold=args.response_threshold,
        per_prefix_rate=args.per_prefix_rate,
        prefix_workers=args.prefix_workers,
        batch_rate=args.batch_rate,
        bgp_rate=args.bgp_rate,
        as_rate=args.as_rate,
        batch_targets=args.batch_targets,
        early_stop=not args.no_early_stop,
        interface=args.interface,
        dns_domain=args.dns_domain
    )
    os.makedirs(config['generated_folder_path'], exist_ok=True)
    os.makedirs(config['zmap_folder_path'], exist_ok=True)
    print(f"Response threshold: {args.response_threshold:.2%}")
    if args.per_prefix_rate is not None:
        print(
            f"Per-prefix rate: {args.per_prefix_rate} pps; "
            f"workers: {args.prefix_workers}; "
            f"max aggregate rate: {args.per_prefix_rate * args.prefix_workers} pps"
        )

    try:
        import pyasn
    except ImportError as exc:
        raise RuntimeError("动态初始步长需要 pyasn，请在 Gungnir 的 wct 环境中运行") from exc

    bgp_db = pyasn.pyasn(args.bgp_dat)
    config['bgp_db'] = bgp_db
    prefix_dict, jump_stats = initialize_prefix_states(config['predict_path'], bgp_db)
    print(
        f"Initial jump: dynamic={jump_stats['dynamic']}; "
        f"fallback-unmatched={jump_stats['unmatched']}; "
        f"fallback-nonpositive={jump_stats['nonpositive']}; fallback-step=8"
    )
    if args.batch_rate:
        window = _batch_window(config)
        as_rate_text = "unlimited" if args.as_rate == 0 else f"{args.as_rate} pps"
        print(
            f"Batch rate: global={args.batch_rate} pps; BGP={args.bgp_rate} pps; "
            f"AS={as_rate_text}; window={window:.3f}s; "
            f"early_stop={not args.no_early_stop}"
        )

    total_budget = 0
    while prefix_dict:
        total_budget, prefix_dict = jump_outoftrap(prefix_dict, total_budget, config)

    print("total buget:", total_budget)


if __name__ == "__main__":
    main()
