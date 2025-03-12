## WWW2025_MMCTR_Challenge
The WWW 2025 Multimodal CTR Prediction Challenge: https://www.codabench.org/competitions/5372/

### Quickly predict the test set:
```
sh pred.sh
```

### Quickly Start
```
sh run.sh
```

### Data Preparation
**Update item embeddings**
```
python combine_item.py
```
Use **item_emb_d128_v3** in item_emb.parquet as 128-d item embeddings,  add **likes_level** and **views_level** in item_feature.parquet as item features, saved in ./MicroLens_1M_x1/item_info_v3.parquet.
```
.data/MicroLens_1M_x1
    ./MicroLens_1M_x1
    ./MicroLens_1M_x1/train.parquet
    ./MicroLens_1M_x1/valid.parquet
    ./MicroLens_1M_x1/test.parquet
    ./MicroLens_1M_x1/item_info.parquet
    ./MicroLens_1M_x1/item_info_v3.parquet
    ./item_feature.parquet
    ./item_emb.parquet    
```

item_info_v3.parquet contains 5 columns: {item_id, item_tags, item_emb_d128, likes_level, views_level}, as shown below:
```
┌─────────┬───────────────┬─────────────────────────────────┬─────────────┬─────────────┐
│ item_id ┆ item_tags     ┆ item_emb_d128                   ┆ likes_level ┆ views_level │
│ ---     ┆ ---           ┆ ---                             ┆ ---         ┆ ---         │
│ i64     ┆ array[i64, 5] ┆ array[f32, 128]                 ┆ i64         ┆ i64         │
╞═════════╪═══════════════╪═════════════════════════════════╪═════════════╪═════════════╡
│ 0       ┆ [0, 0, … 0]   ┆ [0.0, 0.0, … 0.0]               ┆ 0           ┆ 0           │
│ 1       ┆ [0, 0, … 1]   ┆ [-0.587753, -0.38462, … -0.19…] ┆ 7           ┆ 2           │
│ 2       ┆ [0, 0, … 4]   ┆ [-2.505449, 1.56058, … 0.0067…] ┆ 5           ┆ 9           │
└─────────┴───────────────┴─────────────────────────────────┴─────────────┴─────────────┘
```

### How to Run

1. Train the model on train and validation sets:
The main model : src/DIN_att6.py

The best.yaml is the best hyperparameter configuration found by the hyperparameter tuner, only 2 epoch training we can get the best performance.
```
    embedding_regularizer: 1.0e-07  
    net_regularizer: 0.01
    net_dropout: 0.1
    learning_rate: 1.e-3
    batch_size: 4096 
    if_use_self_attn: False
    num_layers: 2
    dnn_hidden_units: [[1024, 512, 256, 64, 16]]
    seed: 21
```

```
python run_param_tuner.py --config config/best.yaml --gpu 0
```

    The best performance: [exp_id] DIN_att6_001_ff34d5a9  [val] AUC: 0.985013 - logloss: 0.115100
    Test Dataset  AUC: 0.988259  Leaderboard Rank: 1

2. Make predictions on the test set:
```
python prediction.py --config config/best --expid DIN_att6_001_ff34d5a9 --gpu 0
```
    
3. Submit the best prediction results to the competition:
 - submission/DIN_att6_001_ff34d5a9.zip

### Environment
We run the experiments on a 4090 GPU server with 24G GPU memory.
Please set up the environment as follows. 

+ torch==2.2.0+cuda12.1
+ fuxictr==2.3.7

```
conda create -n fuxictr python==3.9
pip install -r requirements.txt
source activate fuxictr
```
