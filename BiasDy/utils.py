import torch
import yaml

def LoadYaml(filepath):
    with open(filepath) as file:
        config = yaml.safe_load(file)
    return config

def bmv(A, B):
    """
    Batch matrix vector multiplication
    :param A: shape (batch_size, m, n)
    :param B: shape (batch_size, n)
    :return: shape (batch_size, m)
    """
    return torch.matmul(A, B.unsqueeze(-1)).squeeze(-1)


def adjust_y_lim(*args):
    # unfold each tensor from inputs
    vels = []
    y_lims = [[0.,0],[0.,0],[0.,0]]
    for arg in args:
        vels.append(arg.cpu().detach().numpy())

    # find the max and min of the unfolded tensors
    
    for i in range(3):
        max_vel = max([vel[:,i].max() for vel in vels])
        min_vel = min([vel[:,i].min() for vel in vels])
        y_lims[i][0] = min_vel * 1.5
        y_lims[i][1] = max_vel * 1.5
    
        if max_vel > 50 and min_vel > -10 and min_vel < 0:
            y_lims[i][0] = -10
            y_lims[i][1] = max_vel * 1.2
        elif min_vel < -50 and max_vel < 10 and max_vel > 0:
            y_lims[i][1] = 10
            y_lims[i][0] = min_vel * 1.2
    return y_lims