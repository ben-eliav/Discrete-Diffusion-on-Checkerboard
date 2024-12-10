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


# modelConfig = {
#         "state": "train",
#         "epoch": 50,
#         "batch_size": 64,
#         "T": 1000,
#         "channel": 32,
#         "channel_mult": [1, 2],
#         "attn": [],
#         "num_res_blocks": 2,
#         "dropout": 0.15,
#         "lr": 5e-4,
#         "multiplier": 2.,
#         "beta_1": 1e-4,
#         "beta_T": 0.02,
#         "img_size": 32,
#         "grad_clip": 0.1,
#         "device": "cuda:0",
#         "training_load_weight": None,
#         "save_weight_dir": "./Checkpoints/",
#         "test_load_weight": "ckpt_49_.pt",
#         "sampled_dir": "./outputs/",
#         "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
#         "sampledImgName": "SampledDDPM.png",
#         "nrow": 8,
#         "num_classes": 2,
#         "checkerboard_method": 4,
#         "dataset": "checkerboard",
#         "train_size": 1000,
#         "checkerboard_noise": 0.1,
#         "run_id": "0",
# }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default="train")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--multiplier", type=float, default=2.)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--training_load_weight", type=str, default=None)
    parser.add_argument("--save_weight_dir", type=str, default="./Checkpoints/")
    parser.add_argument("--test_load_weight", type=str, default="ckpt_49_.pt")
    parser.add_argument("--sampled_dir", type=str, default="./outputs/")
    parser.add_argument("--sampledNoisyImgName", type=str, default="NoisyNoGuidenceImgs.png")
    parser.add_argument("--sampledImgName", type=str, default="SampledDDPM.png")
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--checkerboard_method", type=int, default=4)
    parser.add_argument("--noisy_points", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="checkerboard")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--checkerboard_noise", type=float, default=0.1)
    parser.add_argument("--run_id", type=str, default="0")
    parser.add_argument("--show_process", action="store_true")
    parser.add_argument('--show_original', action='store_true')

    args = parser.parse_args()
    args = vars(args)
    args['channel_mult'] = [1, 2]
    args['attn'] = []

    for location in [args['save_weight_dir'], args['sampled_dir']]:
        if not os.path.exists(os.path.join(location, f'Run_{args["run_id"]}')):
            os.makedirs(os.path.join(location, f'Run_{args["run_id"]}'))
        else:
            print(f"Directory {location} already exists and will be overwritten.")
    
    if args['device'] == 'cpu':
        print("############## WARNING: Running on CPU ##############")

    args['save_weight_dir'] = os.path.join(args['save_weight_dir'], f'Run_{args["run_id"]}/')
    args['sampled_dir'] = os.path.join(args['sampled_dir'], f'Run_{args["run_id"]}/')

    if args['state'] == "train":
        train(args)
    elif args['state'] == "test":
        test(args)
    
if __name__ == '__main__':
    main()