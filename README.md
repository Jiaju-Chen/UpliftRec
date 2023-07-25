# UpliftRec


## Overview

This is the official code of "Causal Effect Estimation for User Interest Exploration on Recommender Systems" (AAAI'24).

## Dataset

We use three datasets: Yahoo!R3, Coat, KuaiRec. The former two datasets are accessible in our code while KuaiRec is supposed to download from https://kuairec.com/ due to its size. We process the data respectively for three backend models and UpliftRec in the "data" directory. Just run the jupyter notebook code to generate processed data.

## Backend Models

We choose MF, FM and LightGCN to be our backend models. Specifically, we choose MF for Yahoo!R3 and KuaiRec, and FM for Coat. The code of backend models is in the "code" directory. We save the embedding files of backend models. We also save the embedding file for sub-users for UpliftRec. You can use them directly to train UpliftRec. 

FYI, please use the following command to train optimal backend models:

+ To train MF for Yahoo!R3, first set dataset to "yahoo" and "main_path"  to "../../data/yahoo/yahoo-FM/" in "config.py". Then use this command for the best backend model:

  ```
  python -u main.py --lr=0.0001 --factor_num=512 --batch_size=256 --epochs=30 --gpu=0 --num_ng=24
  ```

+ To train FM for Coat, use this command for the best backend model:

  ```
  python -u main.py --dataset=coat --data_path=../../data/coat/coat-FM/ --lr=0.0001 --hidden_factor=256 --batch_size=16 --epochs=100 --dropout=[0.4,0.2] --gpu=0
  ```

To fetch embeddings of all users (both real users and sub-users), please train under the data_path "dataset-subusers-?cate?" for each dataset with different sub-user numbers and category-clustering numbers.

## Quick Examples with Optimal Parameters

Use the following command to train the model with the optimal parameters: 

+ Uplift-MTEF on Yahoo!R3:

```
python -u main.py --dataset=yahoo --similar_user_num=30 --similar_user_num_propensity=20 --treat_clip_num=8 --ADRF_null=0.01 --propensity_null=0.01 --MTEF_null=0.45 --top_k=[10] --eps=0 --delta_T=1 --use_MTEF=1 --alpha_MTEF=0.15 --check_user=14 --embed_file=embed_user_MF_yahoo_256emb_all.npy --use_onehot_embed=0 --use_topk=1000 --data_path=./data/yahoo/yahoo-FM/ --subdata_path=./data/yahoo/yahoo-subusers-cate5/  --backend_modelname=_MF_yahoo_512emb_backend.npy --backend_path=./code/MF/embeddings/ --gamma=7.5
```

+ Uplift-MTEF on Coat:

```
python -u main.py --dataset=coat --similar_user_num=80 --similar_user_num_propensity=100 --treat_clip_num=6 --ADRF_null=0.01 --propensity_null=0.01 --MTEF_null=0.05 --top_k=[10] --use_MLP=0 --eps=1 --delta_T=1 --use_MTEF=1 --alpha_MTEF=0.4 --check_user=39 --embed_file=embed_user_FM_coat_512emb_all.npy --use_onehot_embed=0 --use_topk=100 --data_path=./data/coat/coat-FM/ --subdata_path=./data/coat/coat-subusers-cate3/ --backend_path=./code/FM/embeddings/ --backend_modelname=_FM_coat_256emb_backend.npy --gamma=0.5
```

+ Uplift-MTEF on KuaiRec:

```
python -u main.py --dataset=kuai --similar_user_num=12288 --similar_user_num_propensity=1000 --treat_clip_num=5 --ADRF_null=0.01 --propensity_null=0.01 --MTEF_null=0.35 --top_k=[10] --eps=0 --delta_T=1 --use_MTEF=1 --alpha_MTEF=0.1 --check_user=0 --embed_file=embed_user_MF_kuai_256emb_all.npy --use_onehot_embed=0 --use_topk=1000 --data_path=./data/kuai/kuai-FM/ --subdata_path=./data/kuai/kuai-subusers-2cate5/ --backend_modelname=_MF_kuai_256emb_backend.npy --backend_path=./code/MF/embeddings/ --gamma=4
```

We train UpliftRec on GPU. The embedding files are also on GPU.




## Requirements

- python == 3.8.13
- pytorch == 1.7.1+cuda9.1

UpliftRec does not rely on specific python and pytorch version.
