import torch
import lie_algebra as Lie

def SO3log_loss(R_pred, R_gt, loss_type = torch.nn.MSELoss()):
   residual = Lie.SO3log(R_pred.transpose(-1,-2) @ R_gt)
   return loss_type(residual, torch.zeros_like(residual))
