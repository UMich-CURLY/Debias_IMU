import torch
import matplotlib.pyplot as plt
import lie_algebra as Lie

def ATE(p_gt: torch.Tensor, p_est: torch.Tensor, mask = None) -> torch.Tensor:
    """
    Absolute Trajectory Error
    :param p_gt: ground truth position: torch.tensor (batch_size, 3)
    :param p_est: estimated position: torch.tensor (batch_size, 3)
    :return: ATE: torch.tensor scalar
    """
    assert p_gt.shape == p_est.shape and p_gt.dim() == 2
    if mask is not None:
        assert mask.dim() == 1 and mask.shape[0] == p_gt.shape[0]
        validated_index = (mask == 0) # shape: (batch_size)
        return (p_gt[validated_index] - p_est[validated_index]).square().sum(dim=1).mean().sqrt()
    else:
        return (p_gt - p_est).square().sum(dim=1).mean().sqrt()

def AOE(R_gt: torch.Tensor, R_est: torch.Tensor, mask = None) -> torch.Tensor:
    """
    Absolute Orientation Error
    :param R_gt: ground truth orientation: torch.tensor (batch_size, 3, 3)
    :param R_est: estimated orientation: torch.tensor (batch_size, 3, 3)
    :param mask: mask for the orientation: torch.tensor (batch_size) 0: valid, 1: invalid
    :return: AOE: torch.tensor scalar
    """
    assert R_gt.shape == R_est.shape and R_gt.dim() == 3
    if mask is not None:
        assert mask.dim() == 1 and mask.shape[0] == R_gt.shape[0]
        validated_index = (mask == 0) # shape: (batch_size)
        return Lie.SO3log(R_gt[validated_index] @ R_est[validated_index].transpose(-1, -2)).square().sum(dim=(-1)).mean().sqrt() * 180 / torch.pi
    else:
        return Lie.SO3log(R_gt @ R_est.transpose(-1, -2)).square().sum(dim=(-1)).mean().sqrt() * 180 / torch.pi
    

def PlotVector3(*args, label_list = None, x_label = None, y_label = None, Plot=False, save_path = None, limits = None, **kwargs):
    """
    Plot N * 3 vectors
    args: x, y, x1, y1, x2, y2, ...
    (N,), (N,3), (N,), (N,3), ...
    """
        
    N = len(args)
    assert N % 2 == 0, "The number of input arguments should be even"
    N = N // 2
    assert label_list == None or N == len(label_list), "The length of label_list should be equal to the number of input arguments"

    fig, ax = plt.subplots(3, 1, **kwargs)
    for i in range(N):
        x = args[2 * i]
        y = args[2 * i + 1]
        assert x.shape[0] == y.shape[0] and y.shape[1] == 3
        for axis_index in range(3):
            if label_list is None:
                ax[axis_index].plot(x, y[:, axis_index], color = 'C' + str(i))
            else:
                ax[axis_index].plot(x, y[:, axis_index], label = label_list[i], color = 'C' + str(i))
                ax[axis_index].legend()
            if limits is not None:
                ax[axis_index].set_ylim(limits)
            ax[axis_index].grid(True)
            if y_label is not None:
                ax[axis_index].set_ylabel(y_label[axis_index])
            if axis_index == 2 and x_label is not None:
                ax[axis_index].set_xlabel(x_label)
    fig.tight_layout()
    if Plot:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close()
    return fig, ax
