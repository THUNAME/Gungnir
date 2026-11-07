# [ICNP2025] FRPv6-Gungnir 🔍
This repo contains the official implementation of  [**"Gungnir: Autoregressive Model for Unified Generation of IPv6 Fully Responsive Prefixes"**]()






## Quick Start
We propose **Gungnir**, *a multi-protocol unified FRP probing framework based on autoregressive semantic modeling*. **Gungnir** captures the intricate relationships between FRP patterns and their influencing factors through a deep semantic learning architecture. It leverages *prefix inference* and a *granularity correction mechanism* to accurately predict and validate FRPs while avoiding errors introduced by incorrect prefix length estimation.

## Requirement
Linux 5.15.0-130-generic #140~20.04.1-Ubuntu

python 3.12.8

NVIDIA GeForce RTX 4090



## File Structure
```
Gungnir/
├── data/
│   └── as_org_categeory.py         # Data Lookup Table
├── make_Population/                # Preparation Data generation module
│   └── make_Population.py          # Preparation Data generation script
│   └── Config.py 
├── Prediction/                     # Prediction output directory
│   └── Prediction.csv              # Target Routing Prefix Prediction file
│   └── PredictionFRP.txt           # Prediction output file
├── train.py                        # Model training script
└── Strategy.py                     # Strategy execution script
└── requirements.txt                # pip requirements
```


### Preparation
- Import the IP-ASN32-DAT file into the ```data/20250123.1600.dat```.(https://archive.routeviews.org/route-views6/bgpdata/)
- Import the seeded data into the ```data/FRPseed``` folder classified by active type.


### 1. Generate Population Data
Run the following command to generate Population_Gungnir.csv:
```bash
python /make_Population/make_Population.py
```


### 2. Train the Model
Run the following command to train the model:
```bash
python train.py
```



### 3. Execute Strategy
Run the following command to execute the prediction strategy (ensure Prediction/Prediction.csv exists):
```bash
python Strategy.py
```


### 4. Online Scanning (Adaptive Threshold Probing)

`adaptive_probe.py` implements single-round Online Scanning with a bounded, two-worker ZMap pipeline. A threshold of 100% (`response_threshold=1.0`) corresponds to the paper's 16/16 acceptance rule; lower thresholds are configurable alternatives. No child prefix is expanded. For subsequent BGP-guided multi-round granularity correction, use `trap_det/trap.py` as described below.

The pipeline works as follows:

- Normalize and deduplicate input prefixes, then use integer masks to generate each next sample on demand. Samples cover the `0-f` child prefixes when space permits; smaller address spaces are enumerated completely.
- A central work-conserving scheduler queues targets by BGP prefix and interleave eligible groups. An idle or exhausted BGP group does not block other groups.
- At most two ZMap workers run concurrently, allowing one worker to send while the other is in cooldown. Target generation, result parsing, and threshold updates overlap with scanning.
- All workers share one 20-second accounting window. The global packet-rate cap is disabled by default (`scan_rate_bps=0`); only the per-BGP limit of 10 packets/s is applied by default, so the aggregate rate emerges from the number of eligible BGP groups. Per-AS pacing is also disabled (`as_rate_pps=0`). A default window allows up to 200 targets per BGP prefix across all workers. `batch_targets=0` means no per-batch target cap; the two workers split each BGP's window quota so every eligible group can be scheduled concurrently without doubling its limit.
- Each prefix stores its sent and responsive counts. With early stopping enabled, `hits >= ceil(total × response_threshold)` accepts immediately, while `hits + remaining < required` rejects immediately. At 0.75 this means 12 responses or five missing responses; at 1.0 the first missing address rejects. Set `early_stop=False` to send every sample.

These are shared time-window average limits, not a strict per-packet token bucket. Each batch logs its target count, assigned rate, remaining work, eligible BGP groups, and elapsed time.

#### Runtime Configuration

Install a ZMapv6 build that provides `icmp6_echoscan`, `ipv6_tcp_synscan`, and `ipv6_dns`. Set network-specific values through the CLI, environment variables, or the `config` dictionary. The scanner uses ZMap `--rate` directly in packets/s and each worker has an independent scanner closure. Production scans use `sudo -n`; configure non-interactive privilege in advance rather than providing a password to the process. Gateway MAC is optional and is included only when explicitly configured.

```bash
export GUNGNIR_LOCAL_IPV6='<your-source-ipv6>'
export GUNGNIR_INTERFACE='<your-interface>'
export GUNGNIR_GATEWAY_MAC='<your-gateway-mac>'  # Optional
export GUNGNIR_ZMAP_PATH='/path/to/zmap'
```

A pyasn database is required for per-BGP pacing. Set `bgp_dat` in `config`, or place the database at `data/latestv6.dat`. Tests and embedded callers can inject an initialized `bgp_db` instead. The pipeline fails rather than bypassing BGP pacing when the database is unavailable.

Only probe networks that you are authorized to measure.

#### Python API

```python
from adaptive_probe import checkFrp

config = {
    "generated_folder_path": "scanningdata/generated_address/",
    "zmap_folder_path": "scanningdata/zmap_result/",
    "scanner": "tcp443",  # Omit this field for ICMPv6 probing.
    "bgp_dat": "data/latestv6.dat",
}

prefixes = {
    "2001:db8:1::/48",  # Documentation prefixes only; replace with authorized targets.
    "2001:db8:2::/60",
}

frp_set = checkFrp(
    config,
    prefixes,
    response_threshold=0.75,
    scan_rate_bps=0,     # Historical name; 0 disables the global packets/s cap.
    bgp_rate_pps=10,
    batch_targets=0,     # 0 leaves the batch size unlimited.
    as_rate_pps=0,
    zmap_workers=2,
    early_stop=True,
)
```

Calling `checkFrp(config, prefixes)` uses the defaults shown above. Set `response_threshold=1.0` for the paper's 16/16 rule and `early_stop=False` when full 16-address budget accounting is required. Temporary target and result files are written below the two configured output directories and removed after each batch. Set `protocol` to `icmp`, `tcp`, or `udp`, with `port` set to 80, 443, or 53 as appropriate. UDP/53 uses `ipv6_dns` and `dns_domain`.

#### Production CLI and four-protocol scheduling

`ipv6_bulk_zmap_scan_nibble.py` is now a production CLI around this pipeline. It atomically writes the exact threshold-accepted CIDR set plus a `run_id`-scoped audit JSON. It does not apply the legacy extractor's separate "at least 10 responses" rule, so small address spaces retain the same `ceil(samples × threshold)` semantics as the Python API.

```bash
python ipv6_bulk_zmap_scan_nibble.py \
    --input /path/to/candidates.txt --output-frp /path/to/active.txt \
    --out-json /path/to/active.json --run-id '<unique-run-id>' \
    --protocol tcp --port 80 --interface eth0 \
    --local-ipv6 '<your-source-ipv6>' --zmap-bin zmap \
    --bgp-dat data/latestv6.dat \
    --generated-folder /tmp/gungnir/generated \
    --zmap-folder /tmp/gungnir/results \
    --response-threshold 0.75 --rate 0 --bgp-rate 10 \
    --as-rate 0 --batch-targets 0 --workers 2
```

`run_week.sh` schedules ICMPv6, TCP80, TCP443, and UDP53 sequentially so their limits do not stack. Its default long-running path is `make_Prediction → Strategy → active scanning → granularity correction`. Every protocol runs in `/home/wct/Gungnir_result/<run_id>/`; only after all four succeed are the shortest covering results atomically published under `server_gungnir`. Any failed stage prevents publication of a partial four-protocol set.

To refresh from the latest existing four snapshots without rerunning the model or consuming the much larger `PredictionFRP` files:

```bash
SKIP_GENERATION=1 USE_LATEST_SNAPSHOTS=1 UPLOAD_RESULTS=1 \
RESPONSE_THRESHOLD=0.75 /home/wct/Gungnir/run_week.sh
```

This mode freezes and hashes all four input snapshot paths before scanning, applies shortest-cover filtering before active scanning and after granularity correction, and invokes `/home/wct/upload/run.sh` only after all four final files are published. Set `UPLOAD_RESULTS=0` to keep results local. The optional 75% mode used by the scheduler is not the paper's 100% baseline.


### 5. Granularity Correction

The ICNP 2025 paper describes a three-stage pipeline: semantic learning, prefix inference, and online scanning. Online Scanning is the Adaptive Threshold Probing stage described above. `trap_det/trap.py` implements the multi-round granularity correction stage and its batch scheduling.

#### Paper baseline (Section IV-D)

The paper divides online scanning into two operations:

1. **FRP confirmation.** Probe 16 randomly selected addresses, one under each of the `0-f` child prefixes when the address space permits. Confirm the candidate as an FRP only when all 16 addresses respond. The evaluation uses the same 16/16 rule, and one probing budget represents 16 addresses within one prefix.
2. **Granularity correction.** Starting from an initial correction length `lc`, progressively probe shorter prefixes and decay `lc` by half in successive rounds. The purpose is to correct prefix-length errors in existing FRP datasets rather than to infer unverified parents by aggregation alone.

The paper does not prescribe a formula for choosing the initial `lc`. Its baseline acceptance threshold is 100%; other threshold values are configurable implementation options.

#### Repository implementation and engineering extensions

The current implementation follows the paper's confirmation-and-correction objective, with the following concrete state machine and scalability extensions:

- **Initial correction length (`lc`):** the dynamic initial jump is the initial correction length `lc` used by granularity correction. Concretely, set `lc = target_prefix_length - bgp_prefix_length` when positive, based on the longest pyasn BGP match; use 8 when the difference is non-positive or no route matches. The paper leaves the initial value open, and this formula is the repository's concrete instantiation.
- **Canonical parent grouping:** compute parents with integer 128-bit masks. Branches resolving to the same canonical parent are measured once per round, while their source state is retained independently.
- **State transition:** move to a responsive parent while retaining the current correction length. After an unsuccessful probe, retain the current prefix and halve the correction length with integer division. Converging branches retain the smallest correction length.
- **Integer address generation:** generate up to 16 addresses with the same `0-f` child-prefix structure. Prefixes with fewer than 16 available addresses are enumerated completely.
- **Configurable response threshold and early stopping:** the baseline is 16/16 (`response_threshold=1.0`). The threshold can be configured when a different acceptance ratio is desired, using `ceil(sample_count * response_threshold)` responses. For example, 0.75 requires 12 responses and permits rejection after five missing responses. Early stopping avoids sending probes that can no longer change the selected threshold's decision.
- **Hierarchical rate control:** limit only the per-BGP rate by default, interleave all eligible BGP groups, and reduce the rate of sparse batches. An optional global cap, per-AS pacing, and sparse-batch pacing are operational controls rather than claims made by the paper.

To reproduce the paper's 16/16 acceptance rule and its full 16-address probing behavior, set `response_threshold=1.0` and disable early stopping:

```bash
python trap_det/trap.py \
    --predict_path /path/to/authorized-prefixes.txt \
    --bgp-dat data/latestv6.dat \
    --local_ipv6 '<your-source-ipv6>' \
    --generated_folder /tmp/gungnir/generated \
    --zmap_folder /tmp/gungnir/results \
    --outputfrp /tmp/gungnir/confirmed.txt \
    --protocol icmp \
    --response-threshold 1.0 \
    --no-early-stop \
    --batch-rate 0 \
    --bgp-rate 10 \
    --as-rate 0 \
    --batch-targets 0
```

For the optional 75% mode, use `--response-threshold 0.75` and omit `--no-early-stop`. At threshold 1.0, early stopping after the first missing address is logically equivalent to the 16/16 decision, but it changes the number of probes and should be disabled when comparing against the paper's budget accounting.

The batch scheduler defaults to no global packet-rate cap (`--batch-rate 0`), no per-batch address cap (`--batch-targets 0`), 10 packets/s per BGP prefix, and no per-AS rate limit (`--as-rate 0`). The fixed default accounting window is 20 seconds, allowing up to 200 addresses per BGP prefix per window. All eligible BGP groups are interleaved, so aggregate throughput grows with the number of eligible BGP prefixes. Set `--batch-rate` to a positive value to restore an explicit global cap, `--batch-targets` to bound each batch, and `--as-rate` to a positive value to enable per-AS pacing. Each batch invokes one ZMap process and pays its cooldown once. These are batch-window average limits, not a per-packet kernel token bucket.

`--batch-rate 0` keeps the batch scheduler enabled and only disables its global cap. Independent per-prefix mode remains available through `--per-prefix-rate`; when using that mode, also set `--batch-rate 0`.

ZMap is invoked with `sudo -n`, so non-interactive privilege must be configured in advance. UDP/53 uses the `ipv6_dns` module and `--dns-domain`. Any nonzero ZMap status or missing metadata aborts the scan rather than being treated as an empty result. Only probe networks that you are authorized to measure.

To discard prefixes already covered by a shorter input prefix without merging adjacent siblings, use:

```bash
python tools/select_shortest_prefixes.py \
    --input /path/to/prefixes.txt \
    --output /path/to/prefixes-shortest.txt
```


## Acknowledgement


## Citation

Gungnir, Luori (conference version), and its journal extension (IEEE Transactions on Networking 2026) are a series of our research work on IPv6 Fully Responsive Prefix (FRP) discovery. If these works are helpful to your research, you are welcome to cite them and follow our ongoing work in this series.

```
@inproceedings{Wei2025gungnir,
  title = {Gungnir: Autoregressive Model for Unified Generation of IPv6 Fully Responsive Prefixes},
  author = {Wei, Chentian and Liu, Ying and He, Lin and Cheng, Daguo and Zhou, Jiasheng},
  booktitle = {Proceedings of the 33rd IEEE International Conference on Network Protocols (ICNP 2025)},
  year = {2025},
  pages = {},
  doi = {},
  address = {Seoul, South Korea},
  date = {September 22-25},
}
```

```
@INPROCEEDINGS{cheng2024luori,
  author={Cheng, Daguo and He, Lin and Wei, Chentian and Yin, Qilei and Jin, Boran and Wang, Zhaoan and Pan, Xiaoteng and Zhou, Sixu and Liu, Ying and Zhang, Shenglin and Tan, Fuchao and Liu, Wenmao},
  booktitle={2024 IEEE 32nd International Conference on Network Protocols (ICNP)},
  title={Luori: Active Probing and Evaluation of Internet-Wide IPv6 Fully Responsive Prefixes},
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Protocols;Current measurement;Reinforcement learning;Transforms;Routing;Optimization;IPv6;fully responsive prefix;active probing},
  doi={10.1109/ICNP61940.2024.10858548}}
```

```
@ARTICLE{11616620,
  author={He, Lin and Wei, Chentian and Cheng, Daguo and Yin, Qilei and Jin, Boran and Wang, Zhaoan and Pan, Xiaoteng and Zhou, Sixu and Liu, Ying and Zhang, Shenglin and Tan, Fuchao and Liu, Wenmao},
  journal={IEEE Transactions on Networking},
  title={Do Not Fall Into the Trap: Efficiently Discovering IPv6 Fully Responsive Prefixes in the Wild},
  year={2026},
  volume={34},
  number={},
  pages={6364-6379},
  keywords={Probes;Internet;Seeds (agriculture);Trees (botanical);Vegetation;Measurement;Ports (computers);Pediatrics;Conferences;Routing;IPv6;fully responsive prefix;active probing},
  doi={10.1109/TON.2026.3715537}}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
