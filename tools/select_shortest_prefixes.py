#!/usr/bin/env python3
"""保留最短覆盖前缀：已有父网时丢弃其子网，不合并相邻兄弟网。"""
import argparse
import ipaddress
import os
import tempfile

_PREFIX_MASKS = tuple(
    ((1 << length) - 1) << (128 - length) for length in range(129)
)


def read_networks(path):
    networks = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                address, length = line.rsplit("/", 1)
                prefix_len = int(length)
                if not 0 <= prefix_len <= 128:
                    raise ValueError("前缀长度超出 0..128")
                network_int = (
                    int(ipaddress.IPv6Address(address)) & _PREFIX_MASKS[prefix_len]
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} 非法 IPv6 前缀：{line}") from exc
            networks.add((network_int, prefix_len))
    return networks


def shortest_covering(networks):
    """CIDR 区间只会包含或不相交；排序后只需与最后保留区间比较。"""
    selected = []
    last_end = -1
    for network_int, prefix_len in sorted(networks):
        if network_int <= last_end:
            continue
        selected.append((network_int, prefix_len))
        last_end = network_int + (1 << (128 - prefix_len)) - 1
    return selected


def write_atomic(path, networks):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for network_int, prefix_len in networks:
                f.write(f"{ipaddress.IPv6Address(network_int)}/{prefix_len}\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    networks = read_networks(args.input)
    selected = shortest_covering(networks)
    write_atomic(args.output, selected)
    print(f"唯一输入前缀：{len(networks)}")
    print(f"最短覆盖前缀：{len(selected)}")
    print(f"移除被覆盖子网：{len(networks) - len(selected)}")
    print(f"输出：{args.output}")


if __name__ == "__main__":
    main()
