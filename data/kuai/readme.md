"./data" saves the raw data of  KuaiRec. Since the raw data is too large to put here, please download it from elsewhere.

+ kuai-FM generates the data for MF and FM.
+ kuai-LGN generates the data for LightGCN. 
+ kuai-subusers-2cate5 generates the data for the augmented dataset. "2cate5" means we generate 2 samples for each real user and cluster categories into 5 groups. We use that to generate sample information, i.e. tuples of (D, T, Y). We also use D to train models for embeddings of both real users and generated user samples.
