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
if world.LOAD:
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
    best_recall = -1
    best_epoch = -1
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        if 1:#epoch %10 == 9:
            
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            if results[0][0] > best_recall:
                best_epoch = epoch
                best_recall = results[0][0]
                torch.save(Recmodel.state_dict(), weight_file)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            
finally:
    if world.tensorboard:
        w.close()
    print("End. Best epoch {:03d}".format(best_epoch+1))