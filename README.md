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

`adaptive_probe.py` provides an optional `checkFrp` implementation with adaptive branch pruning and a decoupled address-generation/scanning pipeline. The original `trap_det/trap.py` and `trap_det/trap_tcp.py` entry points are unchanged.

The pipeline consists of two components:

- `AddressProducer`: generates up to 16 addresses for each active prefix and writes each layer to a temporary target file.
- `ZMapConsumer`: scans the target file through ZMapv6 and returns the responsive addresses to the pruning scheduler.

Input prefixes are deduplicated and processed from shorter to longer prefixes. A branch is pruned when its response ratio is below `response_threshold`; only a 16/16 responsive branch is expanded to the next layer. The returned set keeps the legacy semantics: an input prefix is an FRP only when all addresses generated in its initial layer respond. For prefixes with fewer than 16 available addresses, all generated addresses must respond.

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
