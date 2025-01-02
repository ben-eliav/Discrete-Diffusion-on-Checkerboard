import argparse
import os

from train_eval import train, test

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default="train", help="train or test")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--channel", type=int, default=32, help="Number of channels in the first layer of UNet")
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
    parser.add_argument("--sampledNoisyImgName", type=str, default="NoisyNoGuidenceImgs")
    parser.add_argument("--sampledImgName", type=str, default="SampledD3PM")
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes in the dataset - how to discretize the data")
    parser.add_argument("--checkerboard_method", type=int, default=2, help="use create_checkerboard[i+1] to create dataset")
    parser.add_argument('--checkerboard_squares', type=int, default=4, help='number of squares in a row of the checkerboard')
    parser.add_argument("--noisy_points", type=float, default=0.3, help="fraction of noisy points in the checkerboard dataset")
    parser.add_argument("--dataset", type=str, default="checkerboard", help="checkerboard, MNIST, or CIFAR10")
    parser.add_argument("--train_size", type=int, default=1000, help="number of samples in the training dataset, for checkerboard dataset")
    parser.add_argument("--test_batch", type=int, default=64, help="number of images sampled.")
    parser.add_argument("--run_id", type=str, default="0", help="used to create a directory to save weights and sampled images")
    parser.add_argument("--show_process", action="store_true", help="show sampled images during training")
    parser.add_argument('--show_original', action='store_true', help='show checkerboards from the original dataset')
    parser.add_argument('--display_distribution', action='store_true', help='display the distribution of the data (unet predictions)')
    parser.add_argument('--show_x0_pred', action='store_true', help='show the predicted x0 at every stage of the diffusion')

    args = parser.parse_args()
    args = vars(args)
    args['channel_mult'] = [1, 2]  # value to multiply the number of channels in each layer of UNet
    args['attn'] = []  # indices of layers in UNet to apply attention

    for key, location in {'save_weight_dir': args['save_weight_dir'],'sampled_dir': args['sampled_dir']}.items():
        if location == "None":
            args[key] = None
        else:
            args[key] = os.path.join(location, f'Run_{args["run_id"]}/')
            if not os.path.exists(args[key]):
                os.makedirs(args[key])
            elif args["state"] == "train":
                print(f"Directory {args[key]} already exists and will be overwritten.")
        
    
    if args['device'] == 'cpu':
        print("############## WARNING: Running on CPU ##############")

    if args['state'] == "train":
        train(args)
    elif args['state'] == "test":
        test(args)
    
if __name__ == '__main__':
    main()