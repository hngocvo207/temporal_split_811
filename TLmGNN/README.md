# Ethereum Fraud Detection--TLMG4Eth
This is an implementation of the paper - "Ethereum Fraud Detection via Joint Transaction Language Model and Graph Representation Learning"
## Overview
We propose a hybrid approach combining a transaction language model with graph-based methods to improve fraudulent Ethereum account detection.  We first propose a transaction language model that converts historical transaction sequences into sentences, where transaction attributes are represented as words, to learn explicit transaction semantics. We then propose a transaction attribute similarity graph to model global semantic relationships and apply graph convolution to learn attribute embeddings that encode similarity information. A deep multi-head attention network is then employed to fuse transaction semantic and similarity embeddings. Additionally, we construct an account interaction graph to model transaction behaviors among accounts. Finally, we propose a joint training method for the multi-head attention network and the account interaction graph to leverage the benefits of both.
## Model Designs
![image](https://github.com/lincozz/TLmGNN/blob/main/framework.png)


## Requirements

```
- Python (>=3.8.10)
- Pytorch (>2.3.1)
- Numpy (>=1.24.4)
- Pandas (>=1.4.4)
- Transformers (2.0.0)
- Scikit-learn (>=1.1.3)
- dgl (>=2.0.0)
- Gensim (>=4.3.2)
- Scipy (>=1.10.1)
```

## Dataset

We evaluated the performance of the model using two publicly available and newly released datasets. The composition of the dataset is as follows, you can click on the **"Source"** to download them.

| *Dataset*        | *Nodes*      | *Edges*       | *Avg Degree*   |*Phisher* | *Source*  |
| ---------------- | ------------- | -------------- | -------------- |------- |---------- |
| MulDigraph       |  2,973,489    |  13,551,303    |  4.5574        | 1,165  |  XBlock     |
| B4E              |  597,258      |  11,678,901    |  19.5542       | 3,220  |    Github,Chrome   |
| SPN  |  496,740      |  1831,082      |  1.6730        | 5,619  |    Github       |

## Getting Started 
#### Step1 Create environment and install required packages for TLMG4Eth.
#### Step2 Download the dataset.
#### Step3 Preprocess the dataset to generate transaction text records and transaction network.
```sh
cd gen_MulDi_seq
python dataset1.py
 ...
python dataset11.py

cd gen_b4e_seq
python bedataset1.py
 ...
python bedataset6.py

cd gen_spn_seq
python mydataset1.py
 ...
python mydataset5.py
```
#### Step4 Generate vocabulary list and vocabulary graph
```sh
python gen_train_MulDi.py

python gen_train_b4e.py

python train_on_spn.py
```
#### Step5 Train TLMG4Eth 
```sh
python train_on_MulDi.py

python train_on_b4e.py

python gen_train_spn.py
```

| Parameter                | Description                                                                        |
|--------------------------|------------------------------------------------------------------------------------|
| `m`                | The trade-off parameters between transaction language model and GNNs model.                                             |
| `model_name`         | Which GNNs model should be combined with the trading language model, options (`BertGCN`, `BertGAT`, `BertSAGE`).                                                  |
| `use_baseline`         | if = `True`,the TLM will not contribute to updating the model and making predictions, only GNNs.                                    |
| `epochs`                 | Number of training epochs.                                       |
| `batch_size`             | Batch size, default = `64`.                                                       |
| `vocab`          | The way of building vocab graph to enhance transaction semantics, options (`tf`, `pmi`, `all`)                    |
| `threshold`        | Threshold for establishing a connection between two nodes in a vocab graph,  default = `0.2`.                        |
| `device` | Device used for training models.        |



## Main Results

Here only the F1-Score(Percentage) results are displayed; for other metrics, please refer to the paper.

| *Model*              | *MulDiGraph* | *B4E*     | *SPN*     |
| -------------------- | ------------ | --------- | --------- |
| *Role2Vec*           | 56.08        | 66.73     | 55.12     | 
| *Trans2Vec*          | 70.29        | 38.42     | 51.34     | 
| *GCN*                | 42.47        | 63.59     | 50.09     | 
| *GAT*                | 40.14        | 60.38     | 61.30     |
| *SAGE*               | 34.30        | 51.34     | 51.10     | 
| *BERT4ETH*           | 55.57        | 67.11     | 71.14     | 
| ***Our***            | **90.41**    | **81.23** | **81.46** | 
