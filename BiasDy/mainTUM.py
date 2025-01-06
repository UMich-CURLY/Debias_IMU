import torch
import os

import Interpolation as Interpolation
from network_for_BiadDy import bw_func_choose, ba_func_choose
from dataset import TUMDataset
import argparse

from learning import Gyro_train, Acc_train, Test, simple_visualization, plot_bias


    

def main():
    parser = argparse.ArgumentParser('Learning Bias Dynamics')
    parser.add_argument('--integral_method', type=str, default='euler') # 'euler', 'rk4', 'dopri5', ... see odeint_SO3.py
    parser.add_argument('--outputdir', type=str, default="results/TUM_matser")
    parser.add_argument('--network_type_Gyro', type=str, default='bw_func_net')
    parser.add_argument('--network_type_Acc', type=str, default='ba_func_net')
    parser.add_argument('--bw_model_path', type=str, default=None)
    parser.add_argument('--loss_window', type=int, default=16)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--lr_Gyro', type=float, default=5e-3)
    parser.add_argument('--weight_decay_Gyro', type=float, default=1e-6)
    parser.add_argument('--lr_Acc', type=float, default=7e-3)
    parser.add_argument('--weight_decay_Acc', type=float, default=1e-6)
    parser.add_argument('--epoch', type=int, default=1801)
    args = parser.parse_args()

    ## read the arguments
    integral_method = args.integral_method
    outputdir = args.outputdir
    network_type_Gyro = args.network_type_Gyro
    network_type_Acc = args.network_type_Acc
    lr_Gyro = args.lr_Gyro
    weight_decay_Gyro = args.weight_decay_Gyro
    lr_Acc = args.lr_Acc
    weight_decay_Acc = args.weight_decay_Acc
    loss_window = args.loss_window
    device = "cuda"
    epoch = args.epoch
    test_recompute = True

    dataset_parameters = {
        'dataset_name': 'TUM',
        'data_dir': 'data/TUM', # where are dataset located
        'data_cache_dir': 'data/TUM', # where to save the preprocessed data
        'train_seqs': [
            'dataset-room1',
            'dataset-room3',
            'dataset-room5',
        ],
        'val_seqs': [
            'dataset-room1',
            'dataset-room3',
            'dataset-room5',
        ],
        'test_seqs': [
            'dataset-room2',
            'dataset-room4',
            'dataset-room6',
        ],
        # time_for_train: 50.0 # 
        'dt': 0.005, # time step
        'percent_for_val': 0.2, # the last 0.2 of the data is used for validation for each sequence
        'loss_window': loss_window, # size of the loss window CDE-RNN
        'batch_size': 1000, # number of time windows in each batch
        'sg_window_bw': 15,
        'sg_order_bw': 3,
        'sg_window_ba': 15,
        'sg_order_ba': 3,
        }
    dataset_train = TUMDataset(**dataset_parameters, mode='train',recompute=False)
    dataset_val = TUMDataset(**dataset_parameters, mode='val')
    dataset_test = TUMDataset(**dataset_parameters, mode='test')

    ###### Gyro training ######
    Gyro_train(dataset_train, dataset_val, outputdir, network_type_Gyro, integral_method, device , lr_Gyro, weight_decay_Gyro, epoch=epoch)


    ###### Acc training ######
    bw_model_path = os.path.join(outputdir, "Gyro_weights", "model_epoch_1800.pt")
    bw_func = bw_func_choose(network_type_Gyro, device=device)
    bw_func.load_state_dict(torch.load(bw_model_path, weights_only=True)['func_bw_model'])
    Acc_train(dataset_train, dataset_val, outputdir, bw_func, network_type_Acc, integral_method, device, lr_Acc, weight_decay_Acc, epoch=epoch)


    ###### testing ######
    bw_func = bw_func_choose(network_type_Gyro, device=device)
    bw_func.load_state_dict(torch.load(bw_model_path, weights_only=True)['func_bw_model'])
    ba_model_path = os.path.join(outputdir, "Acc_weights", "model_epoch_1800.pt")
    ba_func = ba_func_choose(network_type_Acc, device=device)
    ba_func.load_state_dict(torch.load(ba_model_path, weights_only=True)['func_ba_model'])
    with torch.no_grad():
        Test(dataset_test, outputdir, bw_func, ba_func, integral_method, device='cpu', recompute=test_recompute)
    

    ##### simple evaluation and plot #####
    print("Gyro lr: ", lr_Gyro, "Gyro weight_decay: ", weight_decay_Gyro)
    seq = [f for f in os.listdir(os.path.join(outputdir, "Sequnces-results")) if os.path.isdir(os.path.join(outputdir, "Sequnces-results", f))]
    for s in seq:
        path_tmp = os.path.join(outputdir, "Sequnces-results", s, "FullTra_result.p")
        simple_visualization(path_tmp)
        plot_bias(path_tmp, dataset_test)
        





if __name__ == "__main__":
    main()

    

