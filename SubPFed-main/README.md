# A Personalized Federated Learning Approach with Subgraphs

## Requirement
- Python 3.9.18
- PyTorch 2.1.2
- PyTorch Geometric 2.5.0
- METIS (for data generation), https://github.com/james77777778/metis_python

## Data Generation
Following command lines automatically generate the dataset.
```sh
$ cd data/generators
$ python overlapping.py
```

## Run 
Following command lines run the experiments for our SubPFed.
```sh
$ sh ./scripts/overlapping.sh [gpus] [num_workers]
```