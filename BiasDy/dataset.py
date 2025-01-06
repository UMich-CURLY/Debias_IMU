import numpy as np
import pickle
import os
import torch
import yaml
from scipy.signal import savgol_filter

import lie_algebra as Lie
import Interpolation 

torch.manual_seed(256)

class BaseDataset():
    def __init__(self, data_name: str, data_cache_dir: str, train_seqs: list, val_seqs: list, test_seqs: list, mode: str,percent_for_val: float, batch_size: int, loss_window: int, dt: float):
        """input:
        - data_cache_dir: directory to save the preprocessed data
        - train_seqs: list of training sequences, example: ['MH_01_easy', 'MH_03_medium',...] 
        - val_seqs: list of validation sequences, example: ['MH_01_easy', 'MH_03_medium',...]
        - test_seqs: list of test sequences, example: ['MH_02_easy', 'MH_04_difficult',...]
        - mode: 'train', 'val' or 'test'
        - percent_for_val: time of trajecoty for validation, seconds, example: 10.0, means 
                        the last 10 seconds of each sequence will be used for validation, the rest for training
        - batch_size: number of time windows in each batch
        - N: size of trajectory during training
        - TODO: add adaptive time step for integration, now is fixed to 0.005
        """
        super().__init__()

        self.dataset_name = data_name
        self.data_cache_dir = data_cache_dir
        self._mode = mode
        sequences_dict = {'train': train_seqs,'val': val_seqs,'test': test_seqs,}
        self._sequences_dict = sequences_dict
        self.sequences = sequences_dict[self._mode]
        self._length = len(self.sequences)

        self._batch_size = batch_size
        self._loss_window = loss_window

        # IMU sampling time
        self.dt = dt # (s)
        self._percent_for_val = percent_for_val

        self.path_normalize_factors = os.path.join(data_cache_dir, 'nf.p')
        self.mean_u = None
        self.std_u = None
        self.bw_const = None

    def get_loss_window(self):
        return self._loss_window
            
    def get_mask_missing(self, i: int)->torch.Tensor:
        imudict = self.load_imu(i)
        mask_missing = imudict['mask_missing']
        return mask_missing # (time,) 0: valid, 1: missing

    def get_data_for_biasDy(self, i: int, device = 'cpu', mask_flag = False)->tuple[torch.Tensor, ...]:
        imudict = self.load_imu(i)
        gtdict = self.load_gt(i)
        coeff = imudict['coeff']
        Spline_time = imudict['t_imu']
        bias_w = imudict['w_bias'] # [time, 3]
        bias_a = imudict['a_bias'] # [time, 3]
        N_max = imudict['u_imu'].shape[0]   
        index_train_max = round(N_max * (1 - self._percent_for_val))
        assert (imudict['t_imu'][1] -imudict['t_imu'][1] -  self.dt) < 0.01, "dt not match"
        if self._mode == 'train':
            s = torch.randint(0, index_train_max - self._loss_window, (self._batch_size, ))
            if mask_flag:
                mask_missing = imudict['mask_missing'] # (time,) 0: valid, 1: missing
                valid_indices = (mask_missing[s] == 0) & (mask_missing[s + self._loss_window] == 0)
                s = s[valid_indices]
            # # debug 
            # s = torch.randint(0, N_max - self._loss_window, (self._batch_size, ))
            R_gt_batch = torch.stack([gtdict['R_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3, 3]
            v_gt_batch = torch.stack([gtdict['v_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            p_gt_batch = torch.stack([gtdict['p_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            X_gt_batch = Lie.SEn3fromSO3Rn(R_gt_batch,torch.cat((v_gt_batch, p_gt_batch), dim=-1))
            t_gt_batch = torch.stack([gtdict['t_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size]
            bw_gt_batch = torch.stack([bias_w[s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            ba_gt_batch = torch.stack([bias_a[s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            t_odeint = torch.arange(0, self._loss_window) * self.dt # [time_loss]
            assert t_odeint.shape[0] == R_gt_batch.shape[0], "t_odeint and R_gt_batch shape not match"

        elif self._mode == 'val':
            s = torch.arange(index_train_max, N_max - self._loss_window, self._loss_window)
            if mask_flag:
                mask_missing = imudict['mask_missing'] # (time,) 0: valid, 1: missing
                valid_indices = (mask_missing[s] == 0) & (mask_missing[s + self._loss_window] == 0)
                s = s[valid_indices]
            R_gt_batch = torch.stack([gtdict['R_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3, 3]
            v_gt_batch = torch.stack([gtdict['v_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            p_gt_batch = torch.stack([gtdict['p_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            X_gt_batch = Lie.SEn3fromSO3Rn(R_gt_batch,torch.cat((v_gt_batch, p_gt_batch), dim=-1))
            t_gt_batch = torch.stack([gtdict['t_gt'][s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size]
            bw_gt_batch = torch.stack([bias_w[s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            ba_gt_batch = torch.stack([bias_a[s + i] for i in range(self._loss_window)], dim=0) # [time_loss, batch_size, 3]
            t_odeint = torch.arange(0, self._loss_window) * self.dt # [time_loss]
        else:
            raise ValueError("get_data_for_biasDy is only for training or validation, if you want to get the full sequence, please use get_full_trajectory")
        a, b, c, d = coeff
        a = a.to(device)
        b = b.to(device)
        c = c.to(device)
        d = d.to(device)
        coeff = (a, b, c, d)

        bwa_gt_batch = torch.cat((bw_gt_batch, ba_gt_batch), dim=-1) # [time_loss, batch_size, 6]

        return (t_gt_batch.to(device), X_gt_batch.to(device), bwa_gt_batch.to(device)), t_odeint.to(device), coeff, Spline_time.to(device), s
    
    def get_coeff(self, i: int)->torch.Tensor:
        imudict = self.load_imu(i)
        Spline_time = imudict['t_imu']
        return imudict['coeff'], Spline_time

    def get_w_bias(self, i: int)->torch.Tensor:
        imudict = self.load_imu(i)
        w_bias = imudict['w_bias']
        t_bias = imudict['t_imu']
        return w_bias, t_bias
    
    def get_a_bias(self, i: int)->torch.Tensor:
        imudict = self.load_imu(i)
        a_bias = imudict['a_bias']
        t_bias = imudict['t_imu']
        return a_bias, t_bias

    def get_train_data(self) -> torch.Tensor:
        assert self._mode == 'train', "get_train_data is only for training mode"
        return self._get_data()
        
    
    def get_val_data(self) -> torch.Tensor:
        assert self._mode == 'val', "get_val_data is only for validation mode"
        return self._get_data()
    
    def get_full_trajectory(self, i: int, device = "cpu")->tuple[torch.Tensor, ...]:
        imudict = self.load_imu(i)
        gtdict = self.load_gt(i)
        t_imu = imudict['t_imu']
        u_imu = imudict['u_imu']
        R_gt = gtdict['R_gt']
        v_gt = gtdict['v_gt']
        p_gt = gtdict['p_gt']
        X_gt = Lie.SEn3fromSO3Rn(R_gt,torch.cat((v_gt, p_gt), dim=-1))
        t_gt = gtdict['t_gt']
        return t_imu.to(device), u_imu.to(device), X_gt.to(device), t_gt.to(device)

    def get_start_timestamp(self, i: int)->float:
        imudict = self.load_imu(i)
        return imudict['t_start']

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        b0 = self.uni.sample(u[:, 0].shape).cuda()
        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] =  b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def length(self):
        return self._length

    def load_imu(self, i):
        return pload(self.data_cache_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.data_cache_dir, self.sequences[i] + '_gt.p')

    def load_coeff(self, i):
        return pload(self.data_cache_dir, self.sequences[i] + '_coeff.p')

    def read_data(self, data_dir):
        raise NotImplementedError
    
    def find_sequence_index(self, sequence):
        return self.sequences.index(sequence)
    
    def find_sequence_mode(self, sequence):
        if sequence in self._sequences_dict['train']:
            return 'train'
        elif sequence in self._sequences_dict['val']:
            return 'val'
        elif sequence in self._sequences_dict['test']:
            return 'test'
        else:
            raise ValueError("Sequence not found in any mode")

    @staticmethod
    def interpolate(x, t, t_int):
            """
            Interpolate ground truth at the sensor timestamps
            """

            # vector interpolation
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if i in [4, 5, 6, 7]:
                    continue
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            # quaternion interpolation
            t_int = torch.Tensor(t_int - t[0])
            t = torch.Tensor(t - t[0])
            qs = Lie.quat_normlize(torch.Tensor(x[:, 4:8]))
            def qinterp(qs, t, t_int):
                idxs = np.searchsorted(t, t_int)
                idxs0 = idxs-1
                idxs0[idxs0 < 0] = 0
                idxs1 = idxs
                idxs1[idxs1 == t.shape[0]] = t.shape[0] - 1
                q0 = qs[idxs0]
                q1 = qs[idxs1]
                tau = torch.zeros_like(t_int)
                dt = (t[idxs1]-t[idxs0])[idxs0 != idxs1]
                tau[idxs0 != idxs1] = (t_int-t[idxs0])[idxs0 != idxs1]/dt
                return Lie.quat_slerp(q0, q1, tau)
            x_int[:, 4:8] = qinterp(qs, t, t_int).numpy()
            return x_int
    
    @staticmethod
    def apply_sg_smoother(data, window_length, polyorder, deriv = 0):
        """
        Apply Savitzky-Golay filter to the data
        data: numpy array of shape (n,6)
        """
        dada_soothed = np.zeros_like(data)
        n = data.shape[-1]
        for i in range(n):
            dada_soothed[:, i] = savgol_filter(data[:, i], window_length, polyorder, deriv=deriv)
        return dada_soothed
    

    @staticmethod
    def construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)->tuple[torch.Tensor, ...]:
        """
        t: [time_loss, batch_size]
        X: [time_loss, batch_size, 5, 5]
        bwa: [time_loss, batch_size, 6]

        return y0, R0 
        if type == 'SO3', y0 = [xi0, t0, bw0] (batch_size, 7)
        if type == 'SE3', y0 = [xi0, t0, bw0, v0, p0, ba0] (batch_size, 16)
        """
        R0 = X_gt_batch[0, :, :3, :3].clone()
        t0 = t_gt_batch[0].clone()
        xi0 = torch.zeros_like(X_gt_batch[0, :, :3, 3])
        v0 = X_gt_batch[0, :, :3, 3]
        p0 = X_gt_batch[0, :, :3, 4]
        bw0 = bwa_gt_batch[0, :, :3]
        ba0 = bwa_gt_batch[0, :, 3:]
        y0 = torch.cat([xi0, t0.unsqueeze(-1), bw0, v0, p0, ba0], dim=-1) # [batch_size, 16]
        return y0, R0
    

class EUROCDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, 
                dataset_name: str, 
                data_dir: str,
                data_cache_dir: str, 
                train_seqs: list, 
                val_seqs: list, 
                test_seqs: list, 
                mode: str,
                percent_for_val: float,  
                batch_size: int, 
                loss_window: int, 
                dt: float, 
                sg_window_bw: int,
                sg_order_bw: int, 
                sg_window_ba: int, 
                sg_order_ba: int,  
                recompute = False):
        super().__init__(dataset_name, data_cache_dir, train_seqs, val_seqs, test_seqs, mode, percent_for_val, batch_size, loss_window, dt)
        # convert raw data to pre loaded data
        self._sg_window_bw = sg_window_bw # window size for bias w smoother
        self._sg_order_bw = sg_order_bw # order for bias w smoother
        self._sg_window_ba = sg_window_ba # window size for bias a smoother
        self._sg_order_ba = sg_order_ba # order for bias a smoother
        self.read_data(data_dir, recompute)

    def read_data(self, data_dir, recompute = False):
        """Read the data from the dataset"""

        f = os.path.join(self.data_cache_dir, self.sequences[0] + '.p')
        if os.path.exists(f) and not recompute:
            # print("Data already converted!")
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "state_groundtruth_estimate0", "data.csv")
            return path_imu, path_gt

        sequences = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)
            timestamp_start = imu[idx0_imu, 0]

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            

            ## pre-process imu data
            seq_mode = self.find_sequence_mode(sequence) # should be 'train', 'val', or 'test'
            if seq_mode == 'train' or seq_mode == 'val':
                #### change ts to equal spacing
                ts = self.dt * np.arange(imu.shape[0])
                tmp = imu[:, 0]/1e9
                tmp = tmp - tmp[0]
                imu_ori = imu.copy()
                imu = np.zeros_like(imu_ori)
                imu[:, 0] = ts
                for i in range(6):
                    imu[:, 1+i] = np.interp(ts, tmp, imu_ori[:, 1+i])
                # interpolate gt
                t_gt = (gt[:, 0]- imu_ori[0, 0])/1e9
                gt = self.interpolate(gt, t_gt, ts)
            if seq_mode == 'test':
                t_start = imu[0, 0]
                ts = (imu[:, 0] - t_start)/1e9
                t_gt = (gt[:, 0] - t_start)/1e9
                gt = self.interpolate(gt, t_gt, ts)
        

            # set the first position as the origin
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # convert from numpy
            q_gt = torch.from_numpy(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = Lie.quat_to_SO3(q_gt.cuda()).cpu()            
            p_gt = torch.from_numpy(p_gt).double()
            v_gt = torch.from_numpy(gt[:, 8:11]).double()
            imu = torch.from_numpy(imu[:, 1:]).double()
            ts = torch.from_numpy(ts).double()

            # calculate w bias
            w_bias = imu[:-1, :3] - Lie.SO3log(Rot_gt[:-1].transpose(-1,-2) @ Rot_gt[1:]) / (ts[1:] - ts[:-1]).unsqueeze(-1) 
            w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
            w_bias = torch.from_numpy(self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw))
            # calculate a bias
            g_const = torch.tensor([0, 0, -9.81])
            a_bias = imu[:-1, 3:6] - (Rot_gt[:-1].transpose(-1,-2) @ ((v_gt[1:] - v_gt[:-1]) / (ts[1:] - ts[:-1]).unsqueeze(-1)-g_const.unsqueeze(0)).unsqueeze(-1)).squeeze(-1)
            a_bias = torch.cat((a_bias, a_bias[-1].unsqueeze(0)), dim=0)
            a_bias = torch.from_numpy(self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba))

            # IMU spline 
            a, b, c, d = Interpolation.get_cubic_hermite_coeffs(ts, imu)
            a = a.float()
            b = b.float()
            c = c.float()
            d = d.float()
            imu_coeff = (a, b, c, d)

            # save for all training
            mondict = {
                't_imu': ts.float(),
                'u_imu': imu.float(),
                'w_bias': w_bias.float(),
                'a_bias': a_bias.float(),
                'coeff': imu_coeff,
                't_start': timestamp_start,
            }
            pdump(mondict, self.data_cache_dir, sequence + ".p")
            # save ground truth
            mondict = {
                't_gt': ts.float(),
                'q_gt': q_gt.float(),
                'R_gt': Rot_gt.float(),
                'v_gt': v_gt.float(),
                'p_gt': p_gt.float(),
            }
            pdump(mondict, self.data_cache_dir, sequence + "_gt.p")

class TUMDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, 
                dataset_name: str, 
                data_dir: str,
                data_cache_dir: str, 
                train_seqs: list, 
                val_seqs: list, 
                test_seqs: list, 
                mode: str,
                percent_for_val: float,  
                batch_size: int, 
                loss_window: int, 
                dt: float, 
                sg_window_bw: int, 
                sg_order_bw: int, 
                sg_window_ba: int, 
                sg_order_ba: int,  
                recompute = False):
        super().__init__(dataset_name, data_cache_dir, train_seqs, val_seqs, test_seqs, mode, percent_for_val, batch_size, loss_window, dt)
        # convert raw data to pre loaded data
        self._sg_window_bw = sg_window_bw
        self._sg_order_bw = sg_order_bw
        self._sg_window_ba = sg_window_ba
        self._sg_order_ba = sg_order_ba
        self.read_data(data_dir, recompute)

    def read_data(self, data_dir, recompute = False):
        """Read the data from the dataset"""

        f = os.path.join(self.data_cache_dir, 'dataset-room1.p')
        if os.path.exists(f) and not recompute:
            # print("Data already converted!")
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "mocap0", "data.csv")
            return path_imu, path_gt
        
        # only get folder name
        sequences = [seq for seq in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, seq))]
        # sequences = os.listdir(data_dir)
        # sequences = self.sequences
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)
            timestamp_start = imu[idx0_imu, 0]

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]

            ## pre-process imu data
            seq_mode = self.find_sequence_mode(sequence) # should be 'train', 'val', or 'test'
            if seq_mode == 'train' or seq_mode == 'val':
                #### change ts to equal spacing
                ts = self.dt * np.arange(imu.shape[0])
                tmp = imu[:, 0]/1e9
                tmp = tmp - tmp[0]
                imu_ori = imu.copy()
                imu = np.zeros_like(imu_ori)
                imu[:, 0] = ts
                for i in range(6):
                    imu[:, 1+i] = np.interp(ts, tmp, imu_ori[:, 1+i])
                # interpolate gt
                t_gt = (gt[:, 0]- imu_ori[0, 0])/1e9
                gt = self.interpolate(gt, t_gt, ts)
            elif seq_mode == 'test':
                t_start = imu[0, 0]
                ts = (imu[:, 0] - t_start)/1e9
                t_gt = (gt[:, 0] - t_start)/1e9
                gt = self.interpolate(gt, t_gt, ts)
            

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = torch.from_numpy(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = Lie.quat_to_SO3(q_gt.cuda()).cpu()
            p_gt = torch.from_numpy(p_gt).double()
            v_gt = self.apply_sg_smoother(gt[:, 1:4], 21, 5, deriv=1) / self.dt
            v_gt = torch.from_numpy(v_gt).double()
            imu = torch.from_numpy(imu[:, 1:]).double()
            ts = torch.from_numpy(ts).double()

            # calculate w bias
            w_bias = imu[:-1, :3] - Lie.SO3log(Rot_gt[:-1].transpose(-1,-2) @ Rot_gt[1:]) / (ts[1:] - ts[:-1]).unsqueeze(-1) 
            w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
            w_bias = torch.from_numpy(self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw))
            # calculate a bias
            g_const = torch.tensor([0, 0, -9.81])
            a_bias = imu[:-1, 3:6] - (Rot_gt[:-1].transpose(-1,-2) @ ((v_gt[1:] - v_gt[:-1]) / (ts[1:] - ts[:-1]).unsqueeze(-1)-g_const.unsqueeze(0)).unsqueeze(-1)).squeeze(-1)
            a_bias = torch.cat((a_bias, a_bias[-1].unsqueeze(0)), dim=0)
            a_bias = torch.from_numpy(self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba))

            # IMU spline 
            a, b, c, d = Interpolation.get_cubic_hermite_coeffs(ts, imu)
            a = a.float()
            b = b.float()
            c = c.float()
            d = d.float()
            imu_coeff = (a, b, c, d)

            ## add mask to the data, some gt data is missing
            mask_missing = torch.zeros_like(ts)
            tmp = np.searchsorted(t_gt, ts) ## shape: (ts.shape[0],), find the index of t_gt that is closest to ts[i]
            diff_t = ts - t_gt[tmp]
            mask_missing[np.abs(diff_t) > 0.01] = 1

            # save for all training
            mondict = {
                # 'xs': dxi_ij.float(),
                't_imu': ts.float(),
                'u_imu': imu.float(),
                'w_bias': w_bias.float(),
                'a_bias': a_bias.float(),
                'coeff': imu_coeff,
                't_start': timestamp_start,
                'mask_missing': mask_missing.float(),
            }
            pdump(mondict, self.data_cache_dir, sequence + ".p")
            # save ground truth
            mondict = {
                't_gt': ts.float(),
                'q_gt': q_gt.float(),
                'R_gt': Rot_gt.float(),
                'v_gt': v_gt.float(),
                'p_gt': p_gt.float(),
            }
            pdump(mondict, self.data_cache_dir, sequence + "_gt.p")

class FetchDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, 
                dataset_name: str, 
                data_dir: str,
                data_cache_dir: str, 
                train_seqs: list, 
                val_seqs: list, 
                test_seqs: list, 
                mode: str,
                percent_for_val: float,  
                batch_size: int, 
                loss_window: int, 
                dt: float, 
                sg_window_bw: int, 
                sg_order_bw: int, 
                sg_window_ba: int, 
                sg_order_ba: int,  
                recompute = False):
        super().__init__(dataset_name, data_cache_dir, train_seqs, val_seqs, test_seqs, mode, percent_for_val, batch_size, loss_window, dt)
        # convert raw data to pre loaded data
        self._sg_window_bw = sg_window_bw
        self._sg_order_bw = sg_order_bw
        self._sg_window_ba = sg_window_ba
        self._sg_order_ba = sg_order_ba
        self.read_data(data_dir, recompute)

    def read_data(self, data_dir, recompute = False):
        """Read the data from the dataset"""

        f = os.path.join(self.data_cache_dir, '01_square.p')
        if os.path.exists(f) and not recompute:
            # print("Data already converted!")
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "mocap0", "data.csv")
            return path_imu, path_gt
        
        # only get folder name
        sequences = [seq for seq in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, seq))]
        # sequences = os.listdir(data_dir)
        # sequences = self.sequences
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)
            timestamp_start = imu[idx0_imu, 0]

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]

            ## pre-process imu data
            seq_mode = self.find_sequence_mode(sequence) # should be 'train', 'val', or 'test'
            if seq_mode == 'train' or seq_mode == 'val':
                #### change ts to equal spacing
                ts = self.dt * np.arange(imu.shape[0])
                tmp = imu[:, 0]/1e9
                tmp = tmp - tmp[0]
                imu_ori = imu.copy()
                imu = np.zeros_like(imu_ori)
                imu[:, 0] = ts
                for i in range(6):
                    imu[:, 1+i] = np.interp(ts, tmp, imu_ori[:, 1+i])
                # interpolate gt
                t_gt = (gt[:, 0]- imu_ori[0, 0])/1e9
                gt = self.interpolate(gt, t_gt, ts)
            elif seq_mode == 'test':
                t_start = imu[0, 0]
                ts = (imu[:, 0] - t_start)/1e9
                t_gt = (gt[:, 0] - t_start)/1e9
                gt = self.interpolate(gt, t_gt, ts)
            

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = torch.from_numpy(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = Lie.quat_to_SO3(q_gt.cuda()).cpu()
            p_gt = torch.from_numpy(p_gt).double()
            v_gt = self.apply_sg_smoother(gt[:, 1:4], 21, 5, deriv=1) / self.dt
            v_gt = torch.from_numpy(v_gt).double()
            imu = torch.from_numpy(imu[:, 1:]).double()
            ts = torch.from_numpy(ts).double()

            # calculate w bias
            w_bias = imu[:-1, :3] - Lie.SO3log(Rot_gt[:-1].transpose(-1,-2) @ Rot_gt[1:]) / (ts[1:] - ts[:-1]).unsqueeze(-1) 
            w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
            w_bias = torch.from_numpy(self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw))
            # calculate a bias
            g_const = torch.tensor([0, 0, -9.81])
            a_bias = imu[:-1, 3:6] - (Rot_gt[:-1].transpose(-1,-2) @ ((v_gt[1:] - v_gt[:-1]) / (ts[1:] - ts[:-1]).unsqueeze(-1)-g_const.unsqueeze(0)).unsqueeze(-1)).squeeze(-1)
            a_bias = torch.cat((a_bias, a_bias[-1].unsqueeze(0)), dim=0)
            a_bias = torch.from_numpy(self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba))

            # IMU spline 
            a, b, c, d = Interpolation.get_cubic_hermite_coeffs(ts, imu)
            a = a.float()
            b = b.float()
            c = c.float()
            d = d.float()
            imu_coeff = (a, b, c, d)

            # save for all training
            mondict = {
                # 'xs': dxi_ij.float(),
                't_imu': ts.float(),
                'u_imu': imu.float(),
                'w_bias': w_bias.float(),
                'a_bias': a_bias.float(),
                'coeff': imu_coeff,
                't_start': timestamp_start,
            }
            pdump(mondict, self.data_cache_dir, sequence + ".p")
            # save ground truth
            mondict = {
                't_gt': ts.float(),
                'q_gt': q_gt.float(),
                'R_gt': Rot_gt.float(),
                'v_gt': v_gt.float(),
                'p_gt': p_gt.float(),
            }
            pdump(mondict, self.data_cache_dir, sequence + "_gt.p")
            
def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

        
        