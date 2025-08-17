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
