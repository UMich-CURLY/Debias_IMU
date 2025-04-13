## check the reulst in difference time interval of integral

import torch
import os
import argparse
import matplotlib.pyplot as plt
import yaml
import re    

import Interpolation as Interpolation
from learning import  simple_visualization



def caculate(outputdir = "results/Euroc_16"):
    logdir_path = 'results/ablation.yaml'
    
    seq = [f for f in os.listdir(os.path.join(outputdir, "Sequnces-results")) if os.path.isdir(os.path.join(outputdir, "Sequnces-results", f))]
    # seq = []
    aoe_ave = []
    ate_ave = []
    for s in seq:
        path_tmp = os.path.join(outputdir, "Sequnces-results", s, "FullTra_result.p")
        aoe, ate = simple_visualization(path_tmp)
        aoe_ave.append(aoe)
        ate_ave.append(ate)
    aoe_ave = torch.tensor(aoe_ave)
    ate_ave = torch.tensor(ate_ave)
    print("AOE: ", aoe_ave.mean(), " ATE: ", ate_ave.mean())
    with open(outputdir + "/time_memory_record.txt", "r") as f:
        content = f.read()
    time_match = re.search(r'Time for training:\s*([\d\.]+)', content)
    memory_match = re.search(r'Max GPU Memory Allocated:\s*([\d\.]+)', content)
    if time_match and memory_match:
        train_time = float(time_match.group(1))
        gpu_memory = float(memory_match.group(1))

    with open(logdir_path, "a") as f:
        yaml.dump({outputdir: {"AOE": aoe_ave.mean().item(), "ATE": ate_ave.mean().item(), "train_time": train_time, "gpu_memory": gpu_memory}}, f)

        


        
def plot_ablation():
    logdir_path = 'results/ablation.yaml'
    with open(logdir_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    aoe_list = []
    vel_list = []
    train_time_list = []
    gpu_memory_list = []
    for k, v in data.items():
        match = re.search(r'Euroc_(\d+)', k)
        if match:
            length_N = int(match.group(1))
            aoe_list.append((length_N, v['AOE']))
            vel_list.append((length_N, v['ATE']))
            train_time_list.append((length_N, v['train_time']))
            gpu_memory_list.append((length_N, v['gpu_memory']))
    pass

    aoe_list = sorted(aoe_list, key=lambda x: x[0])
    vel_list = sorted(vel_list, key=lambda x: x[0])
    train_time_list = sorted(train_time_list, key=lambda x: x[0])
    gpu_memory_list = sorted(gpu_memory_list, key=lambda x: x[0])

    length_N = [x[0] for x in aoe_list]
    aoe = [x[1] for x in aoe_list]
    vel = [x[1] for x in vel_list]
    train_time = [x[1] for x in train_time_list]
    gpu_memory = [x[1] for x in gpu_memory_list]


    fig, ax = plt.subplots(1, 4, figsize=(16, 3))
    ax[0].plot(length_N, aoe, label="AOE", marker="o", color = 'C1', linewidth=2)   
    ax[0].set_title("AOE vs Integration Length N")
    ax[0].set_xlabel("Integration Length N", fontsize=12)
    ax[0].set_ylabel("AOE (deg)", fontsize=12)
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(length_N, vel, label="APE", marker="o", color = 'C1', linewidth=2)
    ax[1].set_title("APE (of velocity) vs Integration Length N")
    ax[1].set_xlabel("Integration Length N", fontsize=12)
    ax[1].set_ylabel("APE (of velocity) (m/s)", fontsize=12)
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(length_N, train_time, label="Training Time", marker="o", color = 'C1', linewidth=2)
    ax[2].set_title("Training Time vs Integration Length N")
    ax[2].set_xlabel("Integration Length N", fontsize=12)
    ax[2].set_ylabel("Training Time (s)", fontsize=12)
    ax[2].grid()
    ax[2].legend()
    ax[3].plot(length_N, gpu_memory, label="GPU Memory", marker="o", color = 'C1', linewidth=2)
    ax[3].set_title("GPU Memory vs Integration Length N")
    ax[3].set_xlabel("Integration Length N", fontsize=12)
    ax[3].set_ylabel("GPU Memory (MB)", fontsize=12)
    ax[3].grid()
    ax[3].legend()
    fig.tight_layout()
    fig.savefig("results/ablation.pdf")

    plt.show()

    # latex table
    print(f"AOE (deg)& {' & '.join([f'{x:.2f}' for x in aoe])} \\\\")
    print(f"APE (velocity) (m/s) & {' & '.join([f'{x:.2f}' for x in vel])} \\\\")
    print(f"Training Time (s)& {' & '.join([f'{x:.2f}' for x in train_time])} \\\\")
    print(f"GPU Memory (MB)& {' & '.join([f'{x:.2f}' for x in gpu_memory])} \\\\")

def main():
    parser = argparse.ArgumentParser('Learning Bias Dynamics')
    parser.add_argument('--outputdir', type=str, default="results/Euroc_16")
    parser.add_argument('--plot_only', action='store_true')
    args = parser.parse_args()

    results_save_dir = args.outputdir
    # args.plot_only = True
    if args.plot_only:
        plot_ablation()
        return
    caculate(results_save_dir)

if __name__ == "__main__":
    main()
    

