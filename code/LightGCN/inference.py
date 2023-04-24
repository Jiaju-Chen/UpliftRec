import world
import utils
from world import cprint
import torch
import numpy as np
#from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if 1:#world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    results, indices_top1k, scores_top1k= Procedure.Test(dataset, Recmodel, 1, w, world.config['multicore'], return_pred = True)
    #embed_user = embed_user.cpu()
    #embed_item = embed_item.cpu()
    #print(embed_user.size())
    np.save('./embeddings/indices_lightGCN_{}_layer{}_dim{}_decay1e-4_lr0.0005_dropout_0.npy'.format(
                             world.dataset,world.config['lightGCN_n_layers'], world.config['latent_dim_rec']), indices_top1k)
    np.save('./embeddings/scores_lightGCN_{}_layer{}_dim{}_decay1e-4_lr0.0005_dropout_0.npy'.format(
                             world.dataset,world.config['lightGCN_n_layers'], world.config['latent_dim_rec']), scores_top1k)
    
finally:
    if world.tensorboard:
        w.close()