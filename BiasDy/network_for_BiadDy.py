import torch
import torch.nn as nn
import lie_algebra as Lie

from utils import bmv

######### bias function #########
class bw_func_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128), 
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh(), 
            nn.Linear(128, 9),
        )
        self.net2 = nn.Sequential(
            nn.Linear(9, 512),
            nn.Tanh(),
            nn.Linear(512, 9),
        )
        self.linear2 = nn.Linear(9, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba
            imu_meas: shape (batch_size, 6) where 6 = w, a
            imu_meas_dot: shape (batch_size, 6) where 6 = w_dot, a_dot"""
        x = torch.cat([ y[..., 4:7], imu_meas[...,:3], imu_meas_dot[...,:3]], dim=-1) # order:  bw, w, w_dot
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)
    
def bw_func_choose(network_type, device = "cuda"):
    if network_type == 'bw_func_net':
        return bw_func_net().to(device)
    else:
        raise NotImplementedError
    
## ba_func
class ba_func_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256), 
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 9),
        )
        self.net2 = nn.Sequential(
            nn.Linear(9, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 9),
        )
        self.linear2 = nn.Linear(9, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
    def forward(self, y, imu_meas, imu_meas_dot, R0):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba
            imu_meas: shape (batch_size, 6) where 6 = w, a
            imu_meas_dot: shape (batch_size, 6) where 6 = w_dot, a_dot"""
        x = torch.cat([y[..., 13:16], imu_meas[...,3:],imu_meas_dot[...,3:]], dim=-1) # order:  ba, a, a_dot
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)

def ba_func_choose(network_type, device = "cuda"):
    if network_type == 'ba_func_net':
        return ba_func_net().to(device)
    else:
        raise NotImplementedError

    
 ## Base ODE
class VFSE23vbiasBase(torch.nn.Module):
    def __init__(self, u_func = None, u_dot_func = None,\
                biasfunc_w: bw_func_net = None, biasfunc_a: ba_func_net = None, biasfunc_v = None, \
                R0 = torch.eye(3), device = "cuda"):
        super().__init__()

        # biasfunc_w, biasfunc_a, biasfunc_v should be a torch.nn.Module
        assert isinstance(biasfunc_w, torch.nn.Module) or biasfunc_w is None
        assert isinstance(biasfunc_a, torch.nn.Module) or biasfunc_a is None
            
        self.biasfunc_w = biasfunc_w
        self.biasfunc_a = biasfunc_a

        self.u_func = u_func
        self.u_dot_func = u_dot_func
        self._device = device
        self.R0 = R0.to(device)
        self.g_const = torch.tensor([0, 0, -9.81]).to(device)

        self._set_bw_zero = False
        self._set_ba_zero = False

        gyro_Rot = 0.05*torch.randn(3, 3).to(device)
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)

    def state_dict(self, *args, **kwargs):
        original_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = [key for key in original_dict.keys() if key.startswith('biasfunc')]
        for key in keys_to_remove:
            original_dict.pop(key)
        return original_dict
    
    def set_u_func(self, u_func):
        self.u_func = u_func
    def set_u_dot_func(self, u_dot_func):
        self.u_dot_func = u_dot_func

    def callback_change_chart(self, R0):
        self.set_R0(R0)
    def set_R0(self, R0):
        self.R0 = R0.clone().to(self._device)

    def __call__(self, t, y: torch.Tensor):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba"""
        #################### !!!!Important!!! ######################
        ##  remenber to assign self.R0 before using this function ##
        ############################################################
        t_abs = y[..., 3] # shape (batch_size,)
        imu_meas = self.u_func(t_abs)
        w_tilde = imu_meas[..., :3]
        imu_meas_dot = self.u_dot_func(t_abs)
        if self._set_bw_zero:
            bw = torch.zeros_like(y[..., 4:7])
            bw_dot = torch.zeros_like(y[..., 4:7])
        else:
            bw = y[..., 4:7]
            bw_dot = self.biasfunc_w(y, imu_meas, imu_meas_dot, self.R0) #* self.alpha 
        if self._set_ba_zero:
            ba = torch.zeros_like(y[..., 13:16])
            ba_dot = torch.zeros_like(y[..., 13:16])
        else:
            ba = y[..., 13:16]
            ba_dot = self.biasfunc_a(y, imu_meas, imu_meas_dot, self.R0)
        xi_dot = bmv(Lie.SO3rightJacoInv(y[..., :3]),w_tilde - bw ) # shape (batch_size, 3)
        t_dot = torch.ones(*t_abs.shape, 1).to(y.device)  # shape (batch_size, 1)
        Rt = self.R0.to(y.device) @ Lie.SO3exp(y[..., :3])
        v_dot = bmv(Rt, imu_meas[...,3:] - ba) + self.g_const.expand_as(y[..., 13:16]).to(y.device) # shape (batch_size, 3)
        p_dot = y[..., 7:10]
        return torch.cat([xi_dot, t_dot, bw_dot, v_dot, p_dot, ba_dot], dim=-1)
    
    def set_bw_zero(self, flag):
        self._set_bw_zero = flag
    
    def set_ba_zero(self, flag):
        self._set_ba_zero = flag

    def freeze_func_bw(self):
        if self.biasfunc_w is not None:
            for param in self.biasfunc_w.parameters():
                param.requires_grad = False
    
    def freeze_func_ba(self):
        if self.biasfunc_a is not None:
            for param in self.biasfunc_a.parameters():
                param.requires_grad = False

    def unfreeze_func_bw(self):
        if self.biasfunc_w is not None:
            for param in self.biasfunc_w.parameters():
                param.requires_grad = True

    def unfreeze_func_ba(self):
        if self.biasfunc_a is not None:
            for param in self.biasfunc_a.parameters():
                param.requires_grad = True

