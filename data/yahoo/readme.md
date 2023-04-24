yahoo saves the raw data of Yahoo!R3.

+ yahoo-FM generates the data for MF and FM.
+ yahoo-LGN generates the data for LightGCN. Please generate yahoo-FM first since some data are loaded from that.
+ yahoo-subusers-cate5 generates the data for subusers. "cate5" means we cluster categories into 5 groups. We use that to generate sub-user information, i.e. tuples of (D, T, Y). We also use D to train models for embeddings of both real users and sub-users.