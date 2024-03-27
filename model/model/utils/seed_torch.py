import torch
import os
import numpy as np

def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # set cpu random seed
    torch.cuda.manual_seed(seed)# set gpu random seed
    torch.cuda.manual_seed_all(seed)# set all gpus random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False