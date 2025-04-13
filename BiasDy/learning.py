import torch
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt

from Third_party.torchdiffeq import odeint as odeint
import Interpolation as Interpolation
from BiasDy_loss import SO3log_loss
from utils import bmv
import lie_algebra as Lie

from network_for_BiadDy import bw_func_net, ba_func_net, VFSE23vbiasBase, bw_func_choose, ba_func_choose

from SO3diffeq import odeint_SO3
from dataset import BaseDataset, EUROCDataset, TUMDataset, pdump, pload

from evalutation import AOE, ATE, PlotVector3
from utils import adjust_y_lim

def write_parameters(type_train: str, dataset: BaseDataset, output_dir: str, network_type: str, lr: float, weight_decay: float, epoch: int):
    if type_train not in ['Gyro', 'Acc']:
        raise ValueError("type should be 'Gyro' or 'Acc'")
    if type_train=='Gyro':
        file_name = "Gyro_parameters.yaml"
    else:
        file_name = "Acc_parameters.yaml"
    
    ## get training parameters and dataset parameters into yaml file
    train_parameters = {
        'dataset_name': dataset.dataset_name,
        'train_type': type_train,
        'network_type': network_type,
        'integral_method': 'euler',
        'loss_windows': dataset._loss_window,
        'lr': lr,
        'weight_decay': weight_decay,
        'epoch': epoch
    }
    dataset_parameters = {
        'dataset_name': dataset.dataset_name,
        'train_seqs': dataset.sequences,
        'sg_window_bw': dataset._sg_window_bw,
        'sg_order_bw': dataset._sg_order_bw,
        'sg_window_ba': dataset._sg_window_ba,
        'sg_order_ba': dataset._sg_order_ba,
    }
    
    output_dir = os.path.join(output_dir, "parameters", file_name)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as file:
        yaml.dump(train_parameters, file)
        yaml.dump(dataset_parameters, file)

def Gyro_train(dataset_train: BaseDataset, dataset_val: BaseDataset, outpur_dir: str, bw_func_name: str,  integral_method = "euler", device = "cuda", lr = 0.005, weight_decay=1e-6, epoch = 1801):  
    ######### training parameters #########
    num_iterations = epoch
    val_freq = 100
    save_freq = 100
    torch.manual_seed(3407) # 3407
    
    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False} # whether to mask some missing data
    mask_flag = MASKFLAG[dataset_train.dataset_name]
    
    ######### define save path #########
    model_save_path = os.path.join(outpur_dir, "Gyro_weights")
    os.makedirs(model_save_path, exist_ok=True)
    log_writer_path = os.path.join(outpur_dir, "Gyro_logs")
    
    ######### define model #########
    bw_func = bw_func_choose(bw_func_name, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func,device=device)
    best_loss = float('inf')
    writter = SummaryWriter(log_writer_path)
    model = model.to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(True)
    model.freeze_func_ba()  

    ######### save parameters #########
    write_parameters("Gyro", dataset_train, outpur_dir, bw_func_name, lr, weight_decay, epoch)
    
    ######### define loss #########
    criterion = SO3log_loss
    

    ######### training #########
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    for epoch in range(num_iterations):
        optimizer.zero_grad()
        model.train()
        
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, Spline_time, induces = dataset_train.get_data_for_biasDy(index, device, mask_flag=mask_flag)
        y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
        """y0_batch: [batch_size, 16] order: [xi0, t0, bw0, v0, p0, ba0]"""
        Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
        model.set_u_func(Spline.evaluate)
        model.set_u_dot_func(Spline.derivative)
        model.set_R0(R0_batch) # don't forget to set R0!
        _, R_sol = odeint_SO3(model, y0_batch, R0_batch, t_odeint, rtol=1e-7, atol=1e-9, method=integral_method) # default int method: dopri5

        train_loss = 1e6 * criterion(R_sol, X_gt_batch[...,:3,:3]) 
        train_loss.backward()
        optimizer.step()
        
        writter.add_scalar("train_loss", train_loss.item(), epoch)
        writter.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        if epoch % val_freq == 0:
            print(f"epoch: {epoch}, train_loss: {train_loss.item()}")
            with torch.no_grad():
                model.eval()
                loss_total = 0
                for index in range(dataset_val.length()):
                    # read data
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, Spline_time, induces = dataset_val.get_data_for_biasDy(index, device, mask_flag=mask_flag)
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    """y0_batch: [batch_size, 16] order: [xi0, t0, bw0, v0, p0, ba0]"""
                    Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
                    model.set_u_func(Spline.evaluate)
                    model.set_u_dot_func(Spline.derivative)
                    model.set_R0(R0_batch) # don't forget to set R0!
                    _, R_sol = odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method) # default int method: dopri5
                    loss = 1e6 * torch.nn.MSELoss()(R_sol, X_gt_batch[...,:3,:3])
                    loss_total += loss.item()
                writter.add_scalar("val_loss", loss_total, epoch)
                print(f"epoch: {epoch}, val_loss: {loss_total}")

            if loss_total < best_loss:
                best_loss = loss_total
                torch.save({'func_bw_model': model.biasfunc_w.state_dict(),'VF_model':model.state_dict(), 'optimizer': optimizer.state_dict(),\
                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_loss': best_loss}, model_save_path + "/best_val_model.pt")

        if epoch % save_freq == 0:
            torch.save({'func_bw_model': model.biasfunc_w.state_dict(), 'VF_model':model.state_dict(),'optimizer': optimizer.state_dict(), \
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'valloss': loss_total}, model_save_path + "/model_epoch_" + str(epoch) + ".pt")
    writter.close()

def Acc_train(dataset_train: EUROCDataset, dataset_val: EUROCDataset, outpur_dir: str, bw_model: bw_func_net, ba_model_name: str, integral_method = "euler", device = "cuda", lr = 0.005, weight_decay=1e-6, epoch = 1801):
    """ bw_model: the bias dynamics model for gyro, should already be trained.
    """
    ######### training parameters #########
    num_iterations = epoch
    val_freq = 100
    save_freq = 100
    torch.manual_seed(3407) # 3407

    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False} # whether to mask some missing data
    mask_flag = MASKFLAG[dataset_train.dataset_name]
    
    ######### define save path #########
    model_save_path = os.path.join(outpur_dir, "Acc_weights")
    os.makedirs(model_save_path, exist_ok=True)
    log_writer_path = os.path.join(outpur_dir, "Acc_logs")
    
    ######### define model #########
    ba_model = ba_func_choose(ba_model_name, device)
    model = VFSE23vbiasBase(biasfunc_w = bw_model, biasfunc_a = ba_model, device = device)
    best_loss = float('inf')
    writter = SummaryWriter(log_writer_path)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.freeze_func_bw()

    ######### save parameters #########
    write_parameters("Acc", dataset_train, outpur_dir, ba_model_name, lr, weight_decay, epoch)
    
    ######### define loss #########
    """MSELoss for acceleration"""
    
    ######### training #########
    gpu_memory = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    for epoch in range(num_iterations):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        model.train()
        
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, Spline_time, induces = dataset_train.get_data_for_biasDy(index, device, mask_flag=mask_flag)
        y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
        """y0_batch: [batch_size, 16] order: [xi0, t0, bw0, v0, p0, ba0]"""
        Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
        model.set_u_func(Spline.evaluate)
        model.set_u_dot_func(Spline.derivative)
        model.set_R0(R0_batch) # don't forget to set R0!
        # R_gt_func = Interpolation.SO3LinearInterpolation(t_odeint, X_gt_batch[...,:3,:3].transpose(0,1), device=device)
        # model.set_batch_R_func(R_gt_func.evaluate)
        solution, _ = odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method) # solution has the same order as y0_batch, [batch_size, N, 16]

        train_loss = 1e6 * torch.nn.MSELoss()(solution[...,7:13], torch.cat([X_gt_batch[...,:3,3], X_gt_batch[...,:3,4]], dim=-1))
        train_loss.backward()
        optimizer.step()
        
        writter.add_scalar("train_loss", train_loss.item(), epoch)
        writter.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()

        if epoch % val_freq == 0:
            print(f"epoch: {epoch}, train_loss: {train_loss.item()}")
            with torch.no_grad():
                model.eval()
                loss_total = 0
                for index in range(dataset_val.length()):
                    # read data
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, Spline_time, induces = dataset_train.get_data_for_biasDy(index, device, mask_flag=mask_flag)
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    """y0_batch: [batch_size, 16] order: [xi0, t0, bw0, v0, p0, ba0]"""
                    Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
                    model.set_u_func(Spline.evaluate)
                    model.set_u_dot_func(Spline.derivative)
                    model.set_R0(R0_batch) # don't forget to set R0!
                    solution, _ = odeint_SO3(model, y0_batch, R0_batch, t_odeint,  method=integral_method) # default int method: dopri5
                    loss = 1e6 * torch.nn.MSELoss()(solution[...,7:13], torch.cat([X_gt_batch[...,:3,3], X_gt_batch[...,:3,4]], dim=-1))
                    loss_total += loss.item()
                writter.add_scalar("val_loss", loss_total, epoch)
                print(f"epoch: {epoch}, val_loss: {loss_total}")

            if loss_total < best_loss:
                best_loss = loss_total
                torch.save({'func_ba_model': model.biasfunc_a.state_dict(), 'optimizer': optimizer.state_dict(),\
                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_loss': best_loss}, model_save_path + "/best_val_model.pt")

        if epoch % save_freq == 0:
            torch.save({'func_ba_model': model.biasfunc_a.state_dict(), 'optimizer': optimizer.state_dict(), \
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'valloss': loss_total}, model_save_path + "/model_epoch_" + str(epoch) + ".pt")
        torch.cuda.synchronize()
        gpu_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
    writter.close()
    return gpu_memory
    


def Test(dataset_test: EUROCDataset, output_dir: str, bw_model: bw_func_net, ba_model: ba_func_net, integral_method = "euler", device = "cpu", recompute = False):
    def calculate_raw_imu(dataset: EUROCDataset, index, device, path: str):
        if not os.path.exists(path):
            print("No results using raw data, calculate it now!")
            t_imu, u_imu, X_gt, t_gt = dataset.get_full_trajectory(index)
            u_imu = u_imu.to(device)
            X_gt = X_gt.to(device)
            t_gt = t_gt.to(device)
            X_raw = torch.zeros_like(X_gt)
            X_raw[0] = X_gt[0]
            g_const = torch.tensor([0, 0, -9.81]).to(device)
            for i in range(1, X_gt.shape[0]):
                dt = t_gt[i] - t_gt[i-1]
                X_raw[i,:3,:3] = X_raw[i-1,:3,:3] @ Lie.SO3exp(u_imu[i-1,:3] * dt)
                X_raw[i,:3,3] = X_raw[i-1,:3,3] + (bmv(X_raw[i-1,:3,:3], u_imu[i-1,3:]) +g_const )* dt
                X_raw[i,:3,4] = X_raw[i-1,:3,4] + X_raw[i-1,:3,3] * dt
            datadic = {'X_raw': X_raw}
            pdump(datadic, path)
        else:
            datadic = pload(path)
            X_raw = datadic['X_raw']
    
    model = VFSE23vbiasBase(biasfunc_w=bw_model.to(device), biasfunc_a=ba_model.to(device), device=device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.eval()

    for index in range(dataset_test.length()):
        path_tmp = os.path.join(output_dir, "Sequnces-results", dataset_test.sequences[index])
        os.makedirs(path_tmp, exist_ok=True)
        if not recompute and os.path.exists(os.path.join(path_tmp, "FullTra_result.p")):
            continue
        print("Testing trajectory: ", dataset_test.sequences[index])
        # read imu data
        coeff, time = dataset_test.get_coeff(index)
        Spline = Interpolation.CubicHermiteSpline(time, coeff, device)
        
        # read initial condition 
        _, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index,device=device) # shape [N, 5, 5], [N]
        R0 = X_gt[0,:3,:3].clone()
        bias_w_gt,_ = dataset_test.get_w_bias(index)
        bias_a_gt,_ = dataset_test.get_a_bias(index)
        bw0 = bias_w_gt[0]
        ba0 = bias_a_gt[0]
        Vec3zeros = torch.zeros(3, device=device)
        # y0 = torch.cat([Vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0,:3,3], X_gt[0,:3,4], ba0, Vec3zeros], dim=-1).unsqueeze(0) # [16,1]
        y0 = torch.cat([Vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0,:3,3], X_gt[0,:3,4], ba0], dim=-1).unsqueeze(0) # [16,1]
        
        # integrate
        model.set_u_func(Spline.evaluate)
        model.set_u_dot_func(Spline.derivative)
        model.set_R0(R0) # don't forget to set R0
        sol_pred, R_pred = odeint_SO3(model, y0, R0, t_gt, method=integral_method)
        sol_pred = sol_pred.squeeze(1) # oder: [xi, t, bw, v, p, ba]
        R_pred = R_pred.squeeze(1)
        net_imu = u_imu - torch.cat([sol_pred[:,4:7], sol_pred[:,13:16]], dim=-1)
        datadic = {'sol_pred': sol_pred, 'R_pred': R_pred, 'X_gt': X_gt, 't_gt': t_gt,'net_imu':net_imu, 'imu_raw': u_imu, 'timestamp_start': dataset_test.get_start_timestamp(index)}
        pdump(datadic, path_tmp, "FullTra_result.p")

    for index in range(dataset_test.length()):
        # calculate raw imu data integration
        path_raw_int = os.path.join(output_dir, "raw_int", dataset_test.sequences[index] + ".p")
        os.makedirs(os.path.dirname(path_raw_int), exist_ok=True)
        calculate_raw_imu(dataset_test, index, device, path_raw_int)


def simple_visualization(dict_path):
    dict = pload(dict_path)
    X_gt = dict['X_gt']
    t_gt = dict['t_gt']
    R_pred = dict['R_pred']
    sol_pred = dict['sol_pred'] # oder: [xi, t, bw, v, p, ba]
    u_imu_raw = dict['imu_raw']
    u_imu_net = dict['net_imu']
    X_pred = Lie.SEn3fromSO3Rn(R_pred, sol_pred[:,7:13])

    ## RPY
    rpy_gt = Lie.SO3_to_RPY(X_gt[:,:3,:3]) * 180 / torch.pi
    rpy_pred = Lie.SO3_to_RPY(R_pred) * 180 / torch.pi
    save_path = os.path.join(os.path.dirname(dict_path), "RPY.pdf")
    PlotVector3(t_gt, rpy_gt, t_gt, rpy_pred, label_list = ['Ground Truth', 'Proposed'], x_label = 'Time (s)', y_label = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'], Plot=False, save_path = save_path, figsize=(10, 6))

    err_SO3 = Lie.SO3log(X_gt[:,:3,:3] @ R_pred.transpose(-1, -2)) * 180 / torch.pi
    save_path = os.path.join(os.path.dirname(dict_path), "SO3error.pdf")
    PlotVector3(t_gt, err_SO3, x_label = 'time', y_label = ['log-x', 'log-y', 'log-z'], Plot=False, save_path = save_path, figsize=(20, 8))

    ## velocity
    vel_gt = X_gt[:,:3,3]
    vel_pred = sol_pred[:,7:10]
    save_path = os.path.join(os.path.dirname(dict_path), "velocity.pdf")
    PlotVector3(t_gt, vel_gt, t_gt, vel_pred, label_list = ['gt', 'pred'], x_label = 'time', y_label = ['vel-x', 'vel-y', 'vel-z'], Plot=False, save_path = save_path, figsize=(20, 8))
    err_vel = vel_gt - vel_pred
    save_path = os.path.join(os.path.dirname(dict_path), "velocity_error.pdf")
    PlotVector3(t_gt, err_vel, x_label = 'time', y_label = ['err-x', 'err-y', 'err-z'], Plot=False, save_path = save_path, figsize=(20, 8))

    ## imu
    save_path = os.path.join(os.path.dirname(dict_path), "w_imu.pdf")
    PlotVector3(t_gt, u_imu_raw[:,:3], t_gt, u_imu_net[:,:3], label_list = ['raw', 'net'], x_label = 'time', y_label = ['w-x', 'w-y', 'w-z'], Plot=False, save_path = save_path, figsize=(20, 8))
    save_path = os.path.join(os.path.dirname(dict_path), "a_imu.pdf")
    PlotVector3(t_gt, u_imu_raw[:,3:], t_gt, u_imu_net[:,3:], label_list = ['raw', 'net'], x_label = 'time', y_label = ['a-x', 'a-y', 'a-z'], Plot=False, save_path = save_path, figsize=(20, 8))


    aoe = AOE(X_gt[:,:3,:3], R_pred)
    ate = ATE(X_gt[:,:3,3], sol_pred[:,7:10])
    print(f"sequence: {os.path.basename(os.path.dirname(dict_path)):20} AOE: {aoe:.3f} ATE vel: {ate:.3f}")

    return aoe, ate
    

def plot_bias(dict_path, dateset: BaseDataset):
    seq_name = os.path.basename(os.path.dirname(dict_path))

    dict = pload(dict_path)
    X_gt = dict['X_gt']
    t_gt = dict['t_gt']
    R_pred = dict['R_pred']
    sol_pred = dict['sol_pred'] # oder: [xi, t, bw, v, p, ba]
    u_imu_raw = dict['imu_raw']
    u_imu_net = dict['net_imu']

    index_seq = dateset.find_sequence_index(seq_name)
    bw_gt, t_bias = dateset.get_w_bias(index_seq)
    ba_gt, _ = dateset.get_a_bias(index_seq)

    bw_pred = sol_pred[:,4:7]
    ba_pred = sol_pred[:,13:16]

    save_path = os.path.join(os.path.dirname(dict_path), "bw.pdf")
    PlotVector3(t_gt, bw_gt, t_gt, bw_pred, label_list = ['gt', 'pred'], x_label = 'time', y_label = ['bw-x', 'bw-y', 'bw-z'], Plot=False, save_path = save_path, figsize=(20, 8))
    save_path = os.path.join(os.path.dirname(dict_path), "ba.pdf")
    PlotVector3(t_gt, ba_gt, t_gt, ba_pred, label_list = ['gt', 'pred'], x_label = 'time', y_label = ['ba-x', 'ba-y', 'ba-z'], Plot=False, save_path = save_path, figsize=(20, 8))



def trainGyroAxb(dataset_train: BaseDataset, dataset_val: BaseDataset, dataset_test: BaseDataset, save_path: str):
    train_flag = True
    device = "cuda"
    torch.manual_seed(3407)
    A = torch.nn.Parameter((torch.eye(3) + 0.01 * torch.randn(3, 3)).to(device), requires_grad=True)
    b = torch.nn.Parameter(0.01 * torch.randn(3).to(device), requires_grad=True)
    
    writer = SummaryWriter(os.path.join(save_path, "GyroAxb_logs"))
    optimizer = torch.optim.Adam([A, b], lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    model_save_path = os.path.join(save_path, "GyroAxbweights")
    os.makedirs(model_save_path, exist_ok=True)
    if train_flag:
        for epoch in range(1801):
            optimizer.zero_grad()
            index = torch.randint(0, dataset_train.length(), (1,)).item()
            (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, _, _, s = dataset_train.get_data_for_biasDy(index, device)
            t_imu, u_imu, X_gt, t_gt = dataset_train.get_full_trajectory(index, device)
            u_imu_batch = torch.stack([u_imu[s + i] for i in range(dataset_train._loss_window)], dim=0) # (loss_window, batch_size, 3)
            w = u_imu_batch[..., :3].to(device)
            w_hat = (A @ w.unsqueeze(-1)).squeeze(-1) + b
            R_hat = X_gt_batch[..., :3, :3].clone()
            R_t = R_hat[0].clone()
            for i in range(dataset_train._loss_window - 1):
                R_t = R_t @ Lie.SO3exp(w_hat[i] * dataset_train.dt)
                R_hat[i + 1] = R_t
            loss = 1e6 * SO3log_loss(R_hat, X_gt_batch[..., :3, :3])
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar("loss", loss.item(), epoch)
            if epoch % 100 == 0:
                # print(f"epoch: {epoch}, loss: {loss.item()}")
                with torch.no_grad():
                    loss_total = 0
                    for index in range(dataset_val.length()):
                        # read data
                        (_, X_gt_batch, _), _, _, _, s = dataset_val.get_data_for_biasDy(index, device)
                        t_imu, u_imu, X_gt, t_gt = dataset_train.get_full_trajectory(index, device)
                        u_imu_batch = torch.stack([u_imu[s + i] for i in range(dataset_train._loss_window)], dim=0) # (loss_window, batch_size, 3)
                        w = u_imu_batch[..., :3].to(device)
                        w_hat = (A @ w.unsqueeze(-1)).squeeze(-1) + b
                        R_hat = X_gt_batch[..., :3, :3].clone()
                        R_t = R_hat[0].clone()
                        for i in range(dataset_train._loss_window - 1):
                            R_t = R_t @ Lie.SO3exp(w_hat[i] * dataset_train.dt)
                            R_hat[i + 1] = R_t
                        loss = 1e6 * SO3log_loss(R_hat, X_gt_batch[..., :3, :3])
                        loss_total += loss.item()
                    writer.add_scalar("val_loss", loss_total, epoch)
                    print(f"epoch: {epoch}, val_loss: {loss_total}")

                torch.save({"A": A, "b": b}, os.path.join(model_save_path, f"model_epoch_{epoch}.pt"))
    writer.close()
    ## write A and b into a yaml file
    with open(os.path.join(model_save_path, "A_b.yaml"), "w") as f:
        yaml.dump({"A": A.cpu().detach().numpy().tolist(), "b": b.cpu().detach().numpy().tolist()}, f, default_flow_style=False)
    return A, b

def trainAccAxb(dataset_train: BaseDataset, dataset_val: BaseDataset, dataset_test: BaseDataset, save_path: str, Gyroweights_path: str):
    
    train_flag = True
    device = "cuda"
    torch.manual_seed(3407)
    A_gyro = torch.load(os.path.join(Gyroweights_path), weights_only=True)["A"].to(device)
    b_gyro = torch.load(os.path.join(Gyroweights_path), weights_only=True)["b"].to(device)

    A = torch.nn.Parameter((torch.eye(3) + 0.01 * torch.randn(3, 3)).to(device), requires_grad=True)
    b = torch.nn.Parameter(0.01 * torch.randn(3).to(device), requires_grad=True)
    
    writer = SummaryWriter(os.path.join(save_path, "AccAxb_logs"))
    optimizer = torch.optim.Adam([A, b], lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    model_save_path = os.path.join(save_path, "AccAxbweights")
    os.makedirs(model_save_path, exist_ok=True)
    if train_flag:
        for epoch in range(1801):
            optimizer.zero_grad()
            index = torch.randint(0, dataset_train.length(), (1,)).item()
            (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, _, _, s = dataset_train.get_data_for_biasDy(index, device)
            t_imu, u_imu, X_gt, t_gt = dataset_train.get_full_trajectory(index, device)
            u_imu_batch = torch.stack([u_imu[s + i] for i in range(dataset_train._loss_window)], dim=0) # (loss_window, batch_size, 3)
            w_hat = (A_gyro @ u_imu_batch[..., :3].unsqueeze(-1)).squeeze(-1) + b_gyro
            a = u_imu_batch[..., 3:].to(device)
            a_hat = (A @ a.unsqueeze(-1)).squeeze(-1) + b
            v_hat = X_gt_batch[..., :3, 3].clone()
            p_hat = X_gt_batch[..., :3, 4].clone()
            R_hat = X_gt_batch[..., :3, :3].clone()
            R_t = R_hat[0].clone()
            v_t = v_hat[0].clone()
            p_t = p_hat[0].clone()
            g_const = torch.tensor([0, 0, -9.81]).to(device)
            for i in range(dataset_train._loss_window - 1):
                p_t = p_t + v_t * dataset_train.dt
                v_t = v_t + bmv(R_t, a_hat[i]) * dataset_train.dt + g_const * dataset_train.dt
                R_t = R_t @ Lie.SO3exp(w_hat[i] * dataset_train.dt)
                p_hat[i + 1] = p_t
                v_hat[i + 1] = v_t
                R_hat[i + 1] = R_t
            loss = 1e6 * torch.nn.MSELoss()(p_hat, X_gt_batch[..., :3, 4]) + 1e6 * torch.nn.MSELoss()(v_hat, X_gt_batch[..., :3, 3])
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar("loss", loss.item(), epoch)
            if epoch % 100 == 0:
                with torch.no_grad():
                    loss_total = 0
                    for index in range(dataset_val.length()):
                        # read data
                        (_, X_gt_batch, _), _, _, _, s = dataset_val.get_data_for_biasDy(index, device)
                        t_imu, u_imu, X_gt, t_gt = dataset_train.get_full_trajectory(index, device)
                        u_imu_batch = torch.stack([u_imu[s + i] for i in range(dataset_train._loss_window)], dim=0)
                        w_hat = (A_gyro @ u_imu_batch[..., :3].unsqueeze(-1)).squeeze(-1) + b_gyro
                        a = u_imu_batch[..., 3:].to(device)
                        a_hat = (A @ a.unsqueeze(-1)).squeeze(-1) + b
                        v_hat = X_gt_batch[..., :3, 3].clone()
                        p_hat = X_gt_batch[..., :3, 4].clone()
                        R_hat = X_gt_batch[..., :3, :3].clone()
                        R_t = R_hat[0].clone()
                        v_t = v_hat[0].clone()
                        p_t = p_hat[0].clone()
                        g_const = torch.tensor([0, 0, -9.81]).to(device)
                        for i in range(dataset_train._loss_window - 1):
                            p_t = p_t + v_t * dataset_train.dt
                            v_t = v_t + bmv(R_t, a_hat[i]) * dataset_train.dt + g_const * dataset_train.dt
                            R_t = R_t @ Lie.SO3exp(w_hat[i] * dataset_train.dt)
                            p_hat[i + 1] = p_t
                            v_hat[i + 1] = v_t
                            R_hat[i + 1] = R_t
                        loss = 1e6 * torch.nn.MSELoss()(p_hat, X_gt_batch[..., :3, 4]) + 1e6 * torch.nn.MSELoss()(v_hat, X_gt_batch[..., :3, 3])
                        loss_total += loss.item()
                    writer.add_scalar("val_loss", loss_total, epoch)
                    print(f"epoch: {epoch}, val_loss: {loss_total}")
                    torch.save({"A": A, "b": b}, os.path.join(model_save_path, f"model_epoch_{epoch}.pt"))
    writer.close()
    ## write A and b into a yaml file
    with open(os.path.join(model_save_path, "A_b.yaml"), "w") as f:
        yaml.dump({"A": A.cpu().detach().numpy().tolist(), "b": b.cpu().detach().numpy().tolist()}, f, default_flow_style=False)
    
def testAxb(dataset_test: BaseDataset, base_dir: str, recompute: bool = False):
    ## test A and b
    device = 'cpu'
    gyro_weights_path = os.path.join(base_dir, "GyroAxbweights", "model_epoch_1800.pt")
    acc_weights_path = os.path.join(base_dir, "AccAxbweights", "model_epoch_1800.pt")
    A_gyro = torch.load(gyro_weights_path, weights_only=True)["A"].to(device)
    b_gyro = torch.load(gyro_weights_path, weights_only=True)["b"].to(device)
    A_acc = torch.load(acc_weights_path, weights_only=True)["A"].to(device)
    b_acc = torch.load(acc_weights_path, weights_only=True)["b"].to(device)
    with torch.no_grad():
        for index in range(dataset_test.length()):
            seq_name = dataset_test.sequences[index]
            print(f"Testing sequence: {seq_name}")
            results_path_Axb = os.path.join(base_dir, "Axb_results",seq_name, "Axb_results.p")
            os.makedirs(os.path.dirname(results_path_Axb), exist_ok=True)
            if not os.path.exists(results_path_Axb) or recompute:
                # print("Calculating Axb results")
                t_imu, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index)
                u_imu = u_imu.to(device)
                w_hat = (A_gyro @ u_imu[..., :3].unsqueeze(-1)).squeeze(-1) + b_gyro
                a_hat = (A_acc @ u_imu[..., 3:].unsqueeze(-1)).squeeze(-1) + b_acc
                X_gt = X_gt.to(device)
                t_gt = t_gt.to(device)
                X_Axb = torch.zeros_like(X_gt)
                X_Axb[0] = X_gt[0]
                g_const = torch.tensor([0, 0, -9.81]).to(device)
                for i in range(1, X_gt.shape[0]):
                    dt = t_gt[i] - t_gt[i-1]
                    X_Axb[i,:3,:3] = X_Axb[i-1,:3,:3] @ Lie.SO3exp(w_hat[i-1] * dt)
                    X_Axb[i,:3,3] = X_Axb[i-1,:3,3] + (bmv(X_Axb[i-1,:3,:3], a_hat[i-1]) + g_const ) * dt
                    X_Axb[i,:3,4] = X_Axb[i-1,:3,4] + X_Axb[i-1,:3,3] * dt
                datadic = {'X_Axb': X_Axb, 'X_gt': X_gt, 't_gt': t_gt, 'u_Axb': torch.cat([w_hat, a_hat], dim=-1)}
                pdump(datadic, results_path_Axb)
            else:
                datadic = pload(results_path_Axb)
                X_Axb = datadic['X_Axb']
                X_gt = datadic['X_gt']
                t_gt = datadic['t_gt']
            ## RPY
            rpy_gt = Lie.SO3_to_RPY(X_gt[:,:3,:3]) * 180 / torch.pi
            rpy_Axb = Lie.SO3_to_RPY(X_Axb[:,:3,:3]) * 180 / torch.pi
            save_path_tmp = os.path.join(os.path.dirname(results_path_Axb), "RPY.pdf")
            PlotVector3(t_gt, rpy_gt, t_gt, rpy_Axb, label_list = ['Ground Truth', 'Proposed'], x_label = 'Time (s)', y_label = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'], Plot=False, save_path = save_path_tmp, figsize=(10, 6))

            err_SO3 = Lie.SO3log(X_gt[:,:3,:3] @ X_Axb[:,:3,:3].transpose(-1, -2)) * 180 / torch.pi
            save_path_tmp = os.path.join(os.path.dirname(results_path_Axb), "SO3error.pdf")
            PlotVector3(t_gt, err_SO3, x_label = 'time', y_label = ['log-x (deg)', 'log-y (deg)', 'log-z (deg)'], Plot=False, save_path = save_path_tmp, figsize=(20, 8))

            ## velocity
            vel_gt = X_gt[:,:3,3]
            vel_Axb = X_Axb[:,:3,3]
            save_path_tmp = os.path.join(os.path.dirname(results_path_Axb), "Velocity.pdf")
            PlotVector3(t_gt, vel_gt, t_gt, vel_Axb, label_list = ['Ground Truth', 'Proposed'], x_label = 'Time (s)', y_label = ['x (m/s)', 'y (m/s)', 'z (m/s)'], Plot=False, save_path = save_path_tmp, figsize=(10, 6))
            err_vel = vel_gt - vel_Axb
            save_path_tmp = os.path.join(os.path.dirname(results_path_Axb), "Velocity_error.pdf")
            PlotVector3(t_gt, err_vel, x_label = 'time', y_label = ['x (m/s)', 'y (m/s)', 'z (m/s)'], Plot=False, save_path = save_path_tmp, figsize=(20, 8))

def compare_visualization(dataset_test: BaseDataset, outputdir: str, other_results: str = None):
    seq = dataset_test.sequences
    for s in seq:
        with torch.no_grad():
            path_tmp = os.path.join(outputdir, "Axb_results", s, "Axb_results.p")
            dict_Axb = pload(path_tmp)
            X_Axb = dict_Axb['X_Axb']
            X_gt = dict_Axb['X_gt']
            t_gt = dict_Axb['t_gt']
            path_tmp = os.path.join(outputdir, "raw_int", s + ".p")
            dict = pload(path_tmp)
            X_raw = dict['X_raw']
            path_tmp = os.path.join(outputdir, "Sequnces-results", s, "FullTra_result.p")
            dict = pload(path_tmp)
            R_pred = dict['R_pred']
            v_pred = dict['sol_pred'][:, 7:10] # order: [xi, t, bw, v, p, ba]
            p_pred = dict['sol_pred'][:, 10:13]

            ## RPY
            AOE_Other = float('inf')
            if other_results is not None:
                raise NotImplementedError
            rpy_gt = Lie.SO3_to_RPY(X_gt[:,:3,:3]) * 180 / torch.pi
            rpy_pred = Lie.SO3_to_RPY(R_pred) * 180 / torch.pi
            rpy_raw = Lie.SO3_to_RPY(X_raw[:,:3,:3]) * 180 / torch.pi
            rpy_Axb = Lie.SO3_to_RPY(X_Axb[:,:3,:3]) * 180 / torch.pi
            if other_results is not None:
                print(f"Seq name: {s:20}AOE: raw: {AOE(X_gt[:,:3,:3], X_raw[:,:3,:3]):.3f} (deg), Axb: {AOE(X_gt[:,:3,:3], X_Axb[:,:3,:3]):.3f} (deg),  pred: {AOE(X_gt[:,:3,:3], R_pred):.3f} (deg), other: {AOE_Other:.3f} (deg)")
            else:
                print(f"Seq name: {s:20}AOE: raw: {AOE(X_gt[:,:3,:3], X_raw[:,:3,:3]):.3f} (deg), Axb: {AOE(X_gt[:,:3,:3], X_Axb[:,:3,:3]):.3f} (deg),  pred: {AOE(X_gt[:,:3,:3], R_pred):.3f} (deg)")

            y_label = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
            x_label = 'Time (s)'
            y_limits = [None, None, None]
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
            for axis_index in range(3):
                ax[axis_index].plot(t_gt, rpy_gt[:, axis_index], color = 'C0', label = 'Ground Truth', linewidth=2)
                ax[axis_index].plot(t_gt, rpy_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linestyle='-.')
                # ax[axis_index].plot(t_gt, rpy_Axb[:, axis_index], color = 'C3', label = 'Linear Model')
                ax[axis_index].plot(t_gt, rpy_pred[:, axis_index], color = 'C1', label = 'Proposed')
                ax[axis_index].grid(True)
                if y_limits[axis_index] != None:
                    ax[axis_index].set_ylim(y_limits[axis_index])
                # ax[axis_index].set_ylim(limits)
                ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
            ax[axis_index].set_xlabel(x_label, fontsize=13)
            ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
            fig.align_labels()
            # fig.suptitle('EUROC MH_O4 Euler Angles', fontsize=16)
            
            fig.tight_layout()
            if True: # s == '05_random':
                save_path = os.path.join(outputdir, "compare", s + "_RPY.pdf")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
            
            ## plot SO3 error
            error_pred = Lie.SO3log(X_gt[:,:3,:3] @ R_pred.transpose(-1, -2)) * 180 / torch.pi
            error_Axb = Lie.SO3log(X_gt[:,:3,:3] @ X_Axb[:,:3,:3].transpose(-1, -2)) * 180 / torch.pi
            error_raw = Lie.SO3log(X_gt[:,:3,:3] @ X_raw[:,:3,:3].transpose(-1, -2)) * 180 / torch.pi
            y_limits = [[-10, 10], [-10, 10], [-10, 10]]
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
            for axis_index in range(3):
                ax[axis_index].plot(t_gt, torch.zeros_like(t_gt), color = 'C0', linestyle='--')
                ax[axis_index].plot(t_gt, error_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linestyle='-.', linewidth=2)
                ax[axis_index].plot(t_gt, error_Axb[:, axis_index], color = 'C3', label = 'Linear Model', linewidth=2)
                ax[axis_index].plot(t_gt, error_pred[:, axis_index], color = 'C1', label = 'Proposed', linewidth=2)
                if other_results is not None:
                    raise NotImplementedError
                ax[axis_index].grid(True)
                if y_limits[axis_index] != None:
                    ax[axis_index].set_ylim(y_limits[axis_index])
                ax[axis_index].set_ylabel(f'log-{axis_index} (deg)', fontsize=13)
            ax[axis_index].set_xlabel(x_label, fontsize=13)
            ax[0].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.40), fontsize=13)
            fig.align_labels()
            fig.tight_layout()
            if True: #s == '05_random':
                save_path = os.path.join(outputdir, "compare", s + "_SO3error.pdf")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
            
            ## plot velocity
            vel_gt = X_gt[:,:3,3]
            vel_pred = v_pred
            vel_raw = X_raw[:,:3,3]
            vel_Axb = X_Axb[:,:3,3]
            # y_limits = [[-10, 10], [-10, 10], [-0.5, 0.5]]
            y_label = ['v-x (m/s)', 'v-y (m/s)', 'v-z (m/s)']
            y_limits = adjust_y_lim(vel_pred, vel_Axb, vel_gt)
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
            for axis_index in range(3):
                ax[axis_index].plot(t_gt, vel_gt[:, axis_index], color = 'C0', label = 'Ground Truth', linewidth=2)
                ax[axis_index].plot(t_gt, vel_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linestyle='-.', linewidth=2)
                ax[axis_index].plot(t_gt, vel_Axb[:, axis_index], color = 'C3', label = 'Linear Model', linewidth=2)
                ax[axis_index].plot(t_gt, vel_pred[:, axis_index], color = 'C1', label = 'Proposed', linewidth=2)
                ax[axis_index].grid(True)
                if y_limits[axis_index] != None:
                    ax[axis_index].set_ylim(y_limits[axis_index])
                ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
            ax[axis_index].set_xlabel(x_label, fontsize=13)
            ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
            fig.align_labels()
            fig.tight_layout()
            if True:
                save_path = os.path.join(outputdir, "compare", s + "_velocity.pdf")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
            
            ## plot velocity error
            err_vel_pred = vel_gt - vel_pred
            err_vel_Axb = vel_gt - vel_Axb
            err_vel_raw = vel_gt - vel_raw
            # y_limits = [None, None, None]
            tmp = torch.cat([err_vel_pred, err_vel_Axb], dim=0)
            y_limits = [[torch.min(tmp[:,i]).item()*1.5, torch.max(tmp[:,i]).item() * 1.5] for i in range(3)]
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
            for axis_index in range(3):
                ax[axis_index].plot(t_gt, err_vel_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linestyle='-.')
                ax[axis_index].plot(t_gt, err_vel_Axb[:, axis_index], color = 'C3', label = 'Linear Model')
                ax[axis_index].plot(t_gt, err_vel_pred[:, axis_index], color = 'C1', label = 'Proposed')
                ax[axis_index].grid(True)
                if y_limits[axis_index] != None:
                    ax[axis_index].set_ylim(y_limits[axis_index])
                ax[axis_index].set_ylabel(f'v_err-{axis_index} (m/s)', fontsize=13)
            ax[axis_index].set_xlabel(x_label, fontsize=13)
            ax[0].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.40), fontsize=13)
            fig.align_labels()
            fig.tight_layout()

            if True:
                save_path = os.path.join(outputdir, "compare", s + "_velocity_error.pdf")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)

            ### calculate APE
            APE_raw = ATE(X_gt[...,:3,4], X_raw[...,:3,4])
            APE_Axb = ATE(X_gt[...,:3,4], X_Axb[...,:3,4])
            APE_pred = ATE(X_gt[...,:3,4], p_pred)
            print(f"Seq name: {s:20}APE: raw: {APE_raw:.3f} (m), Axb: {APE_Axb:.3f} (m),  pred: {APE_pred:.3f} (m)")
    print("Please use EVO or OPENVINS to calculate the metric for formal evaluation.")