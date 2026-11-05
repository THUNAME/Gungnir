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


### 4. Adaptive Threshold Probing (Optional)

`adaptive_probe.py` provides an optional single-round `checkFrp` implementation with adaptive branch pruning and a decoupled address-generation/scanning pipeline. For BGP-guided multi-round granularity correction, use `trap_det/trap.py` as described below.

The pipeline consists of two components:

- `AddressProducer`: generates up to 16 addresses for each input prefix, one under each of the `0-f` child prefixes when space permits, and writes a temporary target file.
- `ZMapConsumer`: scans the target file through ZMapv6 and returns the responsive addresses to the threshold-based classifier.

Input prefixes are deduplicated and processed from shorter to longer prefixes. Every input prefix is probed exactly once: it is classified as FRP when its response ratio reaches `response_threshold`, and no child prefix is ever expanded. A 16/16 result has no special expansion behavior. For prefixes with fewer than 16 available addresses, the response ratio is calculated over the actual number of generated addresses. Set `response_threshold=1.0` to reproduce the original 16/16 acceptance rule.

#### Runtime Configuration

Install a ZMapv6 build that provides `icmp6_echoscan` and/or `ipv6_tcp_synscan`. Set the network-specific values through environment variables or the `config` dictionary; no local addresses, interface names, or gateway addresses are stored in the source code.

```bash
export GUNGNIR_LOCAL_IPV6='<your-source-ipv6>'
export GUNGNIR_INTERFACE='<your-interface>'
export GUNGNIR_GATEWAY_MAC='<your-gateway-mac>'
export GUNGNIR_ZMAP_PATH='/path/to/zmap'  # Optional when using the default local build
read -rsp "sudo password for ZMap: " GUNGNIR_SCAN_PASSWORD
echo
export GUNGNIR_SCAN_PASSWORD
```

Only probe networks that you are authorized to measure.

#### Python API

```python
from adaptive_probe import checkFrp

config = {
    "generated_folder_path": "scanningdata/generated_address/",
    "zmap_folder_path": "scanningdata/zmap_result/",
    "scanner": "tcp443",  # Omit this field for ICMPv6 probing.
}

prefixes = {
    "2001:db8:1::/48",  # Documentation prefixes only; replace with authorized targets.
    "2001:db8:2::/60",
}

frp_set = checkFrp(
    config,
    prefixes,
    response_threshold=0.75,
    scan_rate_bps=100,  # Packets per second, despite the historical parameter name.
)
```

Calling `checkFrp(config, prefixes)` uses `response_threshold=0.75` and `scan_rate_bps=100`. Temporary target and result files are written below the two configured output directories.


### 5. BGP-Guided Multi-Round Probing

`trap_det/trap.py` performs multi-round granularity correction with canonical parent grouping, adaptive early stopping, and hierarchical batch rate limits. It is independent of the single-round `adaptive_probe.py` implementation.

The algorithm works as follows:

1. Load the pyasn database once and find the longest matching BGP prefix for each input prefix. The initial jump is `target_prefix_length - bgp_prefix_length` when positive; otherwise, or when no route matches, it falls back to 8.
2. Generate candidate parents with integer 128-bit masks. All branches that resolve to the same canonical parent share one probe in that round.
3. Generate up to 16 addresses per parent using integer sampling. Smaller address spaces are enumerated completely.
4. Classify a parent using `ceil(sample_count * response_threshold)` unique responses. At a threshold of 0.75, 12 responses accept a 16-sample parent and five missing responses reject it early; undecided parents continue to at most 16 probes.
5. Move to a responsive parent while keeping the current jump. After an unsuccessful probe, keep the current prefix and halve the jump using integer division. Multiple converging branches retain the most conservative jump.

The batch scheduler defaults to a global maximum of 5000 packets/s, 10 packets/s per BGP prefix, 50 packets/s per AS, and 20000 addresses per batch. The default window is therefore four seconds. Each batch invokes one ZMap process and pays its cooldown once. Sparse batches are paced at a lower rate, and BGP/AS groups are interleaved in the target file. These are batch-window average limits, not a per-packet kernel token bucket.

```bash
python trap_det/trap.py \
    --predict_path /path/to/authorized-prefixes.txt \
    --bgp-dat data/latestv6.dat \
    --local_ipv6 '<your-source-ipv6>' \
    --generated_folder /tmp/gungnir/generated \
    --zmap_folder /tmp/gungnir/results \
    --outputfrp /tmp/gungnir/confirmed.txt \
    --protocol icmp \
    --response-threshold 0.75 \
    --batch-rate 5000 \
    --bgp-rate 10 \
    --as-rate 50 \
    --batch-targets 20000
```

`--no-early-stop` sends all 16 samples. Set `--batch-rate 0` to disable the new batch scheduler; the legacy single-batch path and the independent per-prefix mode remain available. When using `--per-prefix-rate`, also set `--batch-rate 0`.

ZMap is invoked with `sudo -n`, so non-interactive privilege must be configured in advance. UDP/53 uses the `ipv6_dns` module and `--dns-domain`. Any nonzero ZMap status or missing metadata aborts the scan rather than being treated as an empty result. Only probe networks that you are authorized to measure.

To discard prefixes already covered by a shorter input prefix without merging adjacent siblings, use:

```bash
python tools/select_shortest_prefixes.py \
    --input /path/to/prefixes.txt \
    --output /path/to/prefixes-shortest.txt
```


## Acknowledgement


## Citation

If you find this paper useful in your research, please cite this paper.

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

## License

This project is released under the [Apache 2.0 license](LICENSE).
