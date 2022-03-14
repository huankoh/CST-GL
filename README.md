# CST-GNN
This is a PyTorch implementation of the paper: [X: Multivariate Time Series Anomaly Detection](). 

## Requirements
The model is implemented using Python3.7 with dependencies specified in requirements.txt
## Data Preparation
### Water Treatment datasets

Follow the instruction in [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/) to download SWaT and WADI datasets. Following [Deng & Hooi (2021)](https://arxiv.org/abs/2106.06947), we pre-process and downsample the original dataset. See the jupyter notebook in generate_data folder.

### Server Machine datasets

Download the SMD dataset from [https://github.com/zhhlee/InterFusion](https://github.com/zhhlee/InterFusion). Pre-process and place it to the data folder. Refer to generate_data foler.

## Anomaly Detection

### Water Treatment datasets

* SWaT
```
python run.py --data data/swat --save ./model-swat.pt --delays [0,6,30,60,120,180,360] --num_nodes 51 --subgraph_size 15 --normalization_window 25000 --pca_compo 10
```

* WADI
```
python run.py --data data/wadi --save ./model-wadi.pt --delays [0,6,30,60,120,180,360] --num_nodes 127 --subgraph_size 30  --normalization_window 200 -- pca_compo 100
```

### Server Machine datasets

* SMD 

Few examples from server machine of different groups. For more information about dataset grouping please refer to [Su et al. (2019)](https://dl.acm.org/doi/10.1145/3292500.3330672).

Machine-1-1
```
python run.py --data data/machine-1-1  --save ./model-machine-1-1.pt --delays [0,1,5,10,20,30,60] --num_nodes 38 --subgraph_size 10 -- pca_compo 8
```
Machine-2-7
```
python run.py --data data/machine-2-7 --save ./model-machine-2-7.pt --delays [0,1,5,10,20,30,60] --num_nodes 38 --subgraph_size 10 -- pca_compo 8
```
Machine-3-4
```
python run.py --data data/machine-3-4 --save ./model-machine-3-4.pt --delays [0,1,5,10,20,30,60] --num_nodes 38 --subgraph_size 10 -- pca_compo 26
```



## Citation

```
citation for our work.
```
