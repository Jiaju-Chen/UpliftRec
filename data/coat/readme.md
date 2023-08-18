coat-origin saves the raw data of Coat.

+ coat-FM generates the data for MF and FM.
+ coat-LGN generates the data for LightGCN. Please generate coat-FM first since some data are loaded from that.
+ coat-subusers-cate3 generates the data for subusers. "cate3" means we cluster categories into 3 groups. We use that to generate sub-user (sample) information, i.e. tuples of (D, T, Y). We also use D to train models for embeddings of both real users and sub-users (generated samples).

