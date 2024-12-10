import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import argparse

from checkerboard import *
from d3pm import D3PM
from discrete_unet import *
from train_eval import train, test


modelConfig = {
        "state": "train",
        "epoch": 50,
        "batch_size": 64,
        "T": 1000,
        "channel": 32,
        "channel_mult": [1, 2],
        "attn": [],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 0.1,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_49_.pt",
        "sampled_dir": "./outputs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledDDPM.png",
        "nrow": 8,
        "num_classes": 2,
        "checkerboard_method": 4,
        "dataset": "checkerboard",
        "train_size": 1000,
        "checkerboard_noise": 0.1,
        "run_id": "0",
}

def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(**modelConfig)
    parser.add_argument("--show_process", action="store_true")
    args = parser.parse_args()
    args = vars(args)    

    for location in [args['save_weight_dir'], args['sampled_dir']]:
        if not os.path.exists(os.path.join(location, f'Run_{args["run_id"]}')):
            os.makedirs(os.path.join(location, f'Run_{args["run_id"]}'))
    
    args['save_weight_dir'] = os.path.join(args['save_weight_dir'], f'Run_{args["run_id"]}/')
    args['sampled_dir'] = os.path.join(args['sampled_dir'], f'Run_{args["run_id"]}/')

    if args.state == "train":
        train(args)
    elif args.state == "test":
        test(args)
    
if __name__ == '__main__':
    main()