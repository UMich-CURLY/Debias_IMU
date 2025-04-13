import os
import numpy as np
import torch
import lie_algebra as Lie
import matplotlib.pyplot as plt

from scipy.linalg import expm, logm

def read_vio_est_file(file_path):
    mat = np.genfromtxt(file_path, delimiter=' ', skip_header=1)
    t = mat[:, 0]
    p = mat[:, 1:4]
    q = mat[:, 4:8]
    Rot = Lie.quat_to_SO3(q = torch.from_numpy(q), order = 'xyzw')
    Rot = Rot.numpy()
    return t, p, Rot

def read_vio_gt_file(file_path):
    mat = np.genfromtxt(file_path, delimiter=' ', skip_header=1)
    t = mat[:, 0]
    p = mat[:, 1:4]
    q = mat[:, 4:8]
    Rot = Lie.quat_to_SO3(q = torch.from_numpy(q), order = 'xyzw')
    Rot = Rot.numpy()
    return t, p, Rot

def get_SE3_alignment(p_gt, p_est, Rot_gt, Rot_est):
    """ p_gt, p_est: (3) 
        Rot_gt, Rot_est: (3, 3) 
    """
    T_gt = np.eye(4)
    T_est = np.eye(4)
    T_gt[:3, :3] = Rot_gt
    T_gt[:3, 3] = p_gt
    T_est[:3, :3] = Rot_est
    T_est[:3, 3] = p_est
    T_mat = T_gt @ np.linalg.inv(T_est)
    return T_mat

def simple_align_with_gt(t_est, p_est, Rot_est, t_gt, p_gt, Rot_gt):
    # align the trajectory
    start_time = max(t_est[0], t_gt[0])
    
    start_index_est = np.where(t_est >= start_time)[0][0]
    start_index_gt = np.where(t_gt >= start_time)[0][0]
    
    # align the trajectory
    # T_gt = np.eye(4)
    # T_est = np.eye(4)
    # T_gt[:3, :3] = Rot_gt[start_index_gt]
    # T_gt[:3, 3] = p_gt[start_index_gt]
    # T_est[:3, :3] = Rot_est[start_index_est]
    # T_est[:3, 3] = p_est[start_index_est]

    # T_mat = T_gt @ np.linalg.inv(T_est) 
    T_mat = get_SE3_alignment(p_gt[start_index_gt], p_est[start_index_est], Rot_gt[start_index_gt], Rot_est[start_index_est])
    
    p_est_align = np.einsum('ij, bj -> bi', T_mat[:3, :3], p_est) + T_mat[:3, 3]
    Rot_est_align = np.einsum('ij, bjk -> bik', T_mat[:3, :3], Rot_est)
    return p_est_align, Rot_est_align

def sub_seq_from_time(t_start, t_end, t, p, Rot):
    start_index = np.where(t >= t_start)[0][0]
    end_index = np.where(t <= t_end)[0][-1]
    return t[start_index:end_index] - t_start, p[start_index:end_index], Rot[start_index:end_index]

def SO3_interp(t_eval, t_original, R_original):
    """
    Interpolates rotations in SO(3) using linear interpolation on the Lie algebra.
    
    Parameters:
        t_eval (array-like): Timestamps at which to evaluate the interpolated rotations.
        t_original (array-like): Sorted timestamps corresponding to the original rotations.
        R_original (array-like): Array of shape (N, 3, 3) of original rotation matrices.
        
    Returns:
        R_interp (np.ndarray): Array of shape (len(t_eval), 3, 3) with the interpolated rotations.
        
    The function uses the formula:
        R(t) = R0 * expm( alpha * logm(R0.T @ R1) )
    where alpha = (t - t0) / (t1 - t0) and t is between t0 and t1.
    For t outside the original range, the nearest rotation is returned.
    """
    t_eval = np.asarray(t_eval)
    t_original = np.asarray(t_original)
    R_original = np.asarray(R_original)
    
    R_interp = []
    
    for t in t_eval:
        # Clamp t to the available time range
        if t <= t_original[0]:
            R_interp.append(R_original[0])
        elif t >= t_original[-1]:
            R_interp.append(R_original[-1])
        else:
            # Find index i such that t_original[i] <= t < t_original[i+1]
            i = np.searchsorted(t_original, t) - 1
            t0 = t_original[i]
            t1 = t_original[i + 1]
            alpha = (t - t0) / (t1 - t0)
            assert 0 <= alpha <= 1
            
            R0 = R_original[i]
            R1 = R_original[i + 1]
            
            # Compute the relative rotation using the matrix logarithm
            delta_R = logm(np.dot(R0.T, R1))
            # Interpolate in the tangent space and map back to SO(3)
            R_t = np.dot(R0, expm(alpha * delta_R))
            R_interp.append(R_t)
    
    return np.array(R_interp)

def interpolate_vec(t_eval, t_original, p_original):
    p_eval = np.zeros((t_eval.shape[0], p_original.shape[-1]))
    for i in range(p_original.shape[-1]):
        p_eval[:, i] = np.interp(t_eval, t_original, p_original[:, i])
    return p_eval

def interpolate_Rp(t_eval, t_original, p_original, R_original):
    p_eval = interpolate_vec(t_eval, t_original, p_original)
    R_eval = SO3_interp(t_eval, t_original, R_original)
    return p_eval, R_eval



def main():
    dir_name = 'results/Euroc_master/VIO'
    os.makedirs(dir_name, exist_ok=True)

    sequences = ['MH_02_easy', 'MH_04_difficult', 'V2_02_medium', 'V1_03_difficult', 'V1_01_easy']
    # sequences = []
    methods = ['Axb', 'MB', 'proposed', 'rawimu']

    
    for seq in sequences:
        gt_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/src/open_vins/ov_data/euroc_mav/{seq}.txt"
        t_gt, p_gt, Rot_gt = read_vio_gt_file(gt_file)
        R_vio_list = []
        p_vio_list = []
        t_est_vio_list = []
        print(f"Processing {seq}")
        for method in methods:
            est_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/results/Euroc/VIO_stereo/{method}/{seq}/00_estimate.txt"
            t_est, p_est, Rot_est = read_vio_est_file(est_file)
            
            p_est_align, Rot_est_align = simple_align_with_gt(t_est, p_est, Rot_est, t_gt, p_gt, Rot_gt)
            R_vio_list.append(Rot_est_align)
            p_vio_list.append(p_est_align)
            t_est_vio_list.append(t_est)

        R_vio_Axb = R_vio_list[0]
        p_vio_Axb = p_vio_list[0]
        t_est_vio_Axb = t_est_vio_list[0]
        R_vio_MB = R_vio_list[1]
        p_vio_MB = p_vio_list[1]
        t_est_vio_MB = t_est_vio_list[1]
        R_vio_proposed = R_vio_list[2]
        p_vio_proposed = p_vio_list[2]
        t_est_vio_proposed = t_est_vio_list[2]
        R_vio_raw = R_vio_list[3]
        p_vio_raw = p_vio_list[3]
        t_est_vio_raw = t_est_vio_list[3]

        start_time = max(t_est_vio_Axb[0], t_gt[0], t_est_vio_MB[0], t_est_vio_proposed[0], t_est_vio_raw[0])
        end_time = min(t_est_vio_Axb[-1], t_gt[-1], t_est_vio_MB[-1], t_est_vio_proposed[-1], t_est_vio_raw[-1])

        t_est_vio_Axb, p_vio_Axb, R_vio_Axb = sub_seq_from_time(start_time, end_time, t_est_vio_Axb, p_vio_Axb, R_vio_Axb)
        t_est_vio_MB, p_vio_MB, R_vio_MB = sub_seq_from_time(start_time, end_time, t_est_vio_MB, p_vio_MB, R_vio_MB)
        t_est_vio_proposed, p_vio_proposed, R_vio_proposed = sub_seq_from_time(start_time, end_time, t_est_vio_proposed, p_vio_proposed, R_vio_proposed)
        t_est_vio_raw, p_vio_raw, R_vio_raw = sub_seq_from_time(start_time, end_time, t_est_vio_raw, p_vio_raw, R_vio_raw)
        t_gt, p_gt, Rot_gt = sub_seq_from_time(start_time, end_time, t_gt, p_gt, Rot_gt)
        
        # plot position
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        y_label = ['p-x (m)', 'p-y (m)', 'p-z (m)']
        x_label = 'Time (s)'
        from utils import adjust_y_lim
        y_limits = adjust_y_lim(torch.from_numpy(p_vio_proposed), torch.from_numpy(p_vio_Axb), torch.from_numpy(p_gt))
        for axis_index in range(3):
            ax[axis_index].plot(t_gt, p_gt[:, axis_index], color = 'C0', label = 'Ground Truth', linewidth=2)
            ax[axis_index].plot(t_est_vio_raw, p_vio_raw[:, axis_index], color = 'C2', label = 'Raw IMU')
            ax[axis_index].plot(t_est_vio_Axb, p_vio_Axb[:, axis_index], color = 'C3', label = 'Linear Model')
            ax[axis_index].plot(t_est_vio_proposed, p_vio_proposed[:, axis_index], color = 'C1', label = 'Proposed')
            ax[axis_index].grid(True)
            if y_limits[axis_index] != None:
                ax[axis_index].set_ylim(y_limits[axis_index])
            ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
        ax[axis_index].set_xlabel(x_label, fontsize=13)
        ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_positionVIO.pdf")
        # plot xy plane
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(p_gt[:, 0], p_gt[:, 1], color = 'C0', label = 'Ground Truth', linewidth=2)
        ax.plot(p_vio_raw[:, 0], p_vio_raw[:, 1], color = 'C2', label = 'Raw IMU')
        ax.plot(p_vio_Axb[:, 0], p_vio_Axb[:, 1], color = 'C3', label = 'Linear Model')
        ax.plot(p_vio_proposed[:, 0], p_vio_proposed[:, 1], color = 'C1', label = 'Proposed')
        ax.grid(True)
        ax.set_xlabel('p-x (m)', fontsize=13)
        ax.set_ylabel('p-y (m)', fontsize=13)
        ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_position_xyVIO.pdf")


        # plot batch position xy plane
        # p_Axb = interpolate_vec(t_gt, t_est_vio_Axb, p_vio_Axb)
        # p_MB = interpolate_vec(t_gt, t_est_vio_MB, p_vio_MB)
        # p_proposed = interpolate_vec(t_gt, t_est_vio_proposed, p_vio_proposed)
        # p_raw = interpolate_vec(t_gt, t_est_vio_raw, p_vio_raw)

        def position_xy_local():
            length_plot = 500
            s = np.random.randint(0, p_gt.shape[0] - length_plot, 3)
            # interp in local windwos
            fig, ax = plt.subplots(1, 3, figsize=(10, 3))
            for i in range(3):
                # align the trajectory
                p_Axb_local, Rot_Axb_local = interpolate_Rp(t_gt[s[i]:s[i] + length_plot], t_est_vio_Axb, p_vio_Axb, R_vio_Axb)
                p_MB_local, Rot_MB_local = interpolate_Rp(t_gt[s[i]:s[i] + length_plot], t_est_vio_MB, p_vio_MB, R_vio_MB)
                p_proposed_local, Rot_proposed_local = interpolate_Rp(t_gt[s[i]:s[i] + length_plot], t_est_vio_proposed, p_vio_proposed, R_vio_proposed)
                p_raw_local, Rot_raw_local = interpolate_Rp(t_gt[s[i]:s[i] + length_plot], t_est_vio_raw, p_vio_raw, R_vio_raw)

                T_mat_Axb = get_SE3_alignment(p_gt[s[i]], p_Axb_local[0], Rot_gt[s[i]], Rot_Axb_local[0])
                T_mat_MB = get_SE3_alignment(p_gt[s[i]], p_MB_local[0], Rot_gt[s[i]], Rot_MB_local[0])
                T_mat_proposed = get_SE3_alignment(p_gt[s[i]], p_proposed_local[0], Rot_gt[s[i]], Rot_proposed_local[0])
                T_mat_raw = get_SE3_alignment(p_gt[s[i]], p_raw_local[0], Rot_gt[s[i]], Rot_raw_local[0])

                p_Axb_align_local = np.einsum('ij, bj -> bi', T_mat_Axb[:3, :3], p_Axb_local) + T_mat_Axb[:3, 3]
                p_MB_align_local = np.einsum('ij, bj -> bi', T_mat_MB[:3, :3], p_MB_local) + T_mat_MB[:3, 3]
                p_proposed_align_local = np.einsum('ij, bj -> bi', T_mat_proposed[:3, :3], p_proposed_local) + T_mat_proposed[:3, 3]
                p_raw_align_local = np.einsum('ij, bj -> bi', T_mat_raw[:3, :3], p_raw_local) + T_mat_raw[:3, 3]

                ax[i].plot(p_gt[s[i]:s[i] + length_plot, 0], p_gt[s[i]:s[i] + length_plot, 1], color = 'C0', label = 'Ground Truth', linewidth=2)
                ax[i].plot(p_raw_align_local[:, 0], p_raw_align_local[:, 1], color = 'C2', label = 'Raw IMU')
                ax[i].plot(p_Axb_align_local[:, 0], p_Axb_align_local[:, 1], color = 'C3', label = 'Linear Model')
                ax[i].plot(p_proposed_align_local[:, 0], p_proposed_align_local[:, 1], color = 'C1', label = 'Proposed')
                ax[i].grid(True)
                ax[i].set_xlabel('p-x (m)')#, fontsize=13)
                ax[i].set_ylabel('p-y (m)')#, fontsize=13)
                ax[i].set_aspect('equal', adjustable='datalim')# axis equal
                ax[i].legend()
            # ax[1].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13, columnspacing=0.8)
            fig.tight_layout()
            fig.savefig(f"{dir_name}/{seq}_position_xy_localVIO.pdf")
            plt.show()
        # position_xy_local()
        

        # plot SO3 error
        p_gt_Axb, R_gt_Axb = interpolate_Rp(t_est_vio_Axb, t_gt, p_gt, Rot_gt)
        p_gt_MB, R_gt_MB = interpolate_Rp(t_est_vio_MB, t_gt, p_gt, Rot_gt)
        p_gt_proposed, R_gt_proposed = interpolate_Rp(t_est_vio_proposed, t_gt, p_gt, Rot_gt)
        p_gt_raw, R_gt_raw = interpolate_Rp(t_est_vio_raw, t_gt, p_gt, Rot_gt)
    
        error_Axb = Lie.SO3log(torch.from_numpy(R_gt_Axb) @ torch.from_numpy(R_vio_Axb).transpose(-1, -2)) * 180 / np.pi
        error_MB = Lie.SO3log(torch.from_numpy(R_gt_MB) @ torch.from_numpy(R_vio_MB).transpose(-1, -2)) * 180 / np.pi
        error_proposed = Lie.SO3log(torch.from_numpy(R_gt_proposed) @ torch.from_numpy(R_vio_proposed).transpose(-1, -2)) * 180 / np.pi
        error_raw = Lie.SO3log(torch.from_numpy(R_gt_raw) @ torch.from_numpy(R_vio_raw).transpose(-1, -2)) * 180 / np.pi
        y_label = ['log-x (deg)', 'log-y (deg)', 'log-z (deg)']
        x_label = 'Time (s)'
        y_limits = [[-2, 2], [-2, 2], [-2, 2]]
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for axis_index in range(3):
            ax[axis_index].plot(t_est_vio_raw, error_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linewidth=2)
            ax[axis_index].plot(t_est_vio_Axb, error_Axb[:, axis_index], color = 'C3', label = 'Linear Model', linewidth=2)
            ax[axis_index].plot(t_est_vio_proposed, error_proposed[:, axis_index], color = 'C1', label = 'Proposed', linewidth=2)
            ax[axis_index].plot(t_est_vio_MB, error_MB[:, axis_index], color = 'C4', label = 'MB', linewidth=2)
            ax[axis_index].plot(t_gt, np.zeros_like(t_gt), color = 'C0', linestyle='--')
            ax[axis_index].grid(True)
            if y_limits[axis_index] != None:
                ax[axis_index].set_ylim(y_limits[axis_index])
            ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
        ax[axis_index].set_xlabel(x_label, fontsize=13)
        ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_SO3errorVIO.pdf")
        # plot position error 
        error_Axb = p_vio_Axb - p_gt_Axb
        error_MB = p_vio_MB - p_gt_MB
        error_proposed = p_vio_proposed - p_gt_proposed
        error_raw = p_vio_raw - p_gt_raw
        y_label = ['p-x (m)', 'p-y (m)', 'p-z (m)']
        x_label = 'Time (s)'
        y_limits = [[-2, 2], [-2, 2], [-1, 1]]
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for axis_index in range(3):
            ax[axis_index].plot(t_est_vio_raw, error_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linewidth=2)
            ax[axis_index].plot(t_est_vio_Axb, error_Axb[:, axis_index], color = 'C3', label = 'Linear Model', linewidth=2)
            ax[axis_index].plot(t_est_vio_proposed, error_proposed[:, axis_index], color = 'C1', label = 'Proposed', linewidth=2)
            ax[axis_index].plot(t_est_vio_MB, error_MB[:, axis_index], color = 'C4', label = 'MB', linewidth=2)
            ax[axis_index].plot(t_gt, np.zeros_like(t_gt), color = 'C0', linestyle='--')
            ax[axis_index].grid(True)
            if y_limits[axis_index] != None:
                ax[axis_index].set_ylim(y_limits[axis_index])
            ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
        ax[axis_index].set_xlabel(x_label, fontsize=13)
        ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_position_errorVIO.pdf")
        plt.close('all')
        # plt.show()
    
    


    
        

    ## TUM VI
    dir_name = 'results/TUM/VIO'
    os.makedirs(dir_name, exist_ok=True)

    sequences = ['dataset-room2', 'dataset-room4', 'dataset-room6']
    methods = ['Axb', 'MBrossard', 'proposed', 'rawimu']

    for seq in sequences:
        gt_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/src/open_vins/ov_data/tum_vi_test/{seq}.txt"
        t_gt, p_gt, Rot_gt = read_vio_gt_file(gt_file)
        R_vio_list = []
        p_vio_list = []
        t_est_vio_list = []
        print(f"Processing {seq}")
        for method in methods:
            est_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/results/Tum/VIO_stereo/{method}/{seq}/00_estimate.txt"
            t_est, p_est, Rot_est = read_vio_est_file(est_file)
            
            p_est_align, Rot_est_align = simple_align_with_gt(t_est, p_est, Rot_est, t_gt, p_gt, Rot_gt)
            R_vio_list.append(Rot_est_align)
            p_vio_list.append(p_est_align)
            t_est_vio_list.append(t_est)

        R_vio_Axb = R_vio_list[0]
        p_vio_Axb = p_vio_list[0]
        t_est_vio_Axb = t_est_vio_list[0]
        R_vio_MB = R_vio_list[1]
        p_vio_MB = p_vio_list[1]
        t_est_vio_MB = t_est_vio_list[1]
        R_vio_proposed = R_vio_list[2]
        p_vio_proposed = p_vio_list[2]
        t_est_vio_proposed = t_est_vio_list[2]
        R_vio_raw = R_vio_list[3]
        p_vio_raw = p_vio_list[3]
        t_est_vio_raw = t_est_vio_list[3]

        start_time = max(t_est_vio_Axb[0], t_gt[0], t_est_vio_MB[0], t_est_vio_proposed[0], t_est_vio_raw[0])
        end_time = min(t_est_vio_Axb[-1], t_gt[-1], t_est_vio_MB[-1], t_est_vio_proposed[-1], t_est_vio_raw[-1])

        t_est_vio_Axb, p_vio_Axb, R_vio_Axb = sub_seq_from_time(start_time, end_time, t_est_vio_Axb, p_vio_Axb, R_vio_Axb)
        t_est_vio_MB, p_vio_MB, R_vio_MB = sub_seq_from_time(start_time, end_time, t_est_vio_MB, p_vio_MB, R_vio_MB)
        t_est_vio_proposed, p_vio_proposed, R_vio_proposed = sub_seq_from_time(start_time, end_time, t_est_vio_proposed, p_vio_proposed, R_vio_proposed)
        t_est_vio_raw, p_vio_raw, R_vio_raw = sub_seq_from_time(start_time, end_time, t_est_vio_raw, p_vio_raw, R_vio_raw)
        t_gt, p_gt, Rot_gt = sub_seq_from_time(start_time, end_time, t_gt, p_gt, Rot_gt)
        
        # plot position
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        y_label = ['p-x (m)', 'p-y (m)', 'p-z (m)']
        x_label = 'Time (s)'
        from utils import adjust_y_lim
        y_limits = adjust_y_lim(torch.from_numpy(p_vio_proposed), torch.from_numpy(p_vio_Axb), torch.from_numpy(p_gt))
        for axis_index in range(3):
            ax[axis_index].plot(t_gt, p_gt[:, axis_index], color = 'C0', label = 'Ground Truth', linewidth=2)
            # ax[axis_index].plot(t_est_vio_raw, p_vio_raw[:, axis_index], color = 'C2', label = 'Raw IMU')
            ax[axis_index].plot(t_est_vio_Axb, p_vio_Axb[:, axis_index], color = 'C3', label = 'Linear Model')
            ax[axis_index].plot(t_est_vio_proposed, p_vio_proposed[:, axis_index], color = 'C1', label = 'Proposed')
            ax[axis_index].grid(True)
            if y_limits[axis_index] != None:
                ax[axis_index].set_ylim(y_limits[axis_index])
            ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
        ax[axis_index].set_xlabel(x_label, fontsize=13)
        ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_positionVIO.pdf")
        # plot xy plane
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(p_gt[:, 0], p_gt[:, 1], color = 'C0', label = 'Ground Truth', linewidth=2)
        ax.plot(p_vio_raw[:, 0], p_vio_raw[:, 1], color = 'C2', label = 'Raw IMU')
        ax.plot(p_vio_Axb[:, 0], p_vio_Axb[:, 1], color = 'C3', label = 'Linear Model')
        ax.plot(p_vio_proposed[:, 0], p_vio_proposed[:, 1], color = 'C1', label = 'Proposed')
        ax.grid(True)
        ax.set_xlabel('p-x (m)', fontsize=13)
        ax.set_ylabel('p-y (m)', fontsize=13)
        ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_position_xyVIO.pdf")

        

        # plot SO3 error
        p_gt_Axb, R_gt_Axb = interpolate_Rp(t_est_vio_Axb, t_gt, p_gt, Rot_gt)
        p_gt_MB, R_gt_MB = interpolate_Rp(t_est_vio_MB, t_gt, p_gt, Rot_gt)
        p_gt_proposed, R_gt_proposed = interpolate_Rp(t_est_vio_proposed, t_gt, p_gt, Rot_gt)
        p_gt_raw, R_gt_raw = interpolate_Rp(t_est_vio_raw, t_gt, p_gt, Rot_gt)
    
        error_Axb = Lie.SO3log(torch.from_numpy(R_gt_Axb) @ torch.from_numpy(R_vio_Axb).transpose(-1, -2)) * 180 / np.pi
        error_MB = Lie.SO3log(torch.from_numpy(R_gt_MB) @ torch.from_numpy(R_vio_MB).transpose(-1, -2)) * 180 / np.pi
        error_proposed = Lie.SO3log(torch.from_numpy(R_gt_proposed) @ torch.from_numpy(R_vio_proposed).transpose(-1, -2)) * 180 / np.pi
        error_raw = Lie.SO3log(torch.from_numpy(R_gt_raw) @ torch.from_numpy(R_vio_raw).transpose(-1, -2)) * 180 / np.pi
        y_label = ['log-x (deg)', 'log-y (deg)', 'log-z (deg)']
        x_label = 'Time (s)'
        y_limits = [[-5, 5], [-5, 5], [-5, 5]]
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for axis_index in range(3):
            ax[axis_index].plot(t_est_vio_raw, error_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linestyle='-.')
            ax[axis_index].plot(t_est_vio_Axb, error_Axb[:, axis_index], color = 'C3', label = 'Linear Model', linewidth=2)
            ax[axis_index].plot(t_est_vio_proposed, error_proposed[:, axis_index], color = 'C1', label = 'Proposed', linewidth=2)
            ax[axis_index].plot(t_est_vio_MB, error_MB[:, axis_index], color = 'C4', label = 'MB', linewidth=2)
            ax[axis_index].plot(t_gt, np.zeros_like(t_gt), color = 'C0', linestyle='--')
            ax[axis_index].grid(True)
            if y_limits[axis_index] != None:
                ax[axis_index].set_ylim(y_limits[axis_index])
            ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
        ax[axis_index].set_xlabel(x_label, fontsize=13)
        ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_SO3errorVIO.pdf")
        # plot position error 
        error_Axb = p_vio_Axb - p_gt_Axb
        error_MB = p_vio_MB - p_gt_MB
        error_proposed = p_vio_proposed - p_gt_proposed
        error_raw = p_vio_raw - p_gt_raw
        y_label = ['p-x (m)', 'p-y (m)', 'p-z (m)']
        x_label = 'Time (s)'
        y_limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for axis_index in range(3):
            ax[axis_index].plot(t_est_vio_raw, error_raw[:, axis_index], color = 'C2', label = 'Raw IMU', linestyle='-.')
            ax[axis_index].plot(t_est_vio_Axb, error_Axb[:, axis_index], color = 'C3', label = 'Linear Model', linewidth=2)
            ax[axis_index].plot(t_est_vio_proposed, error_proposed[:, axis_index], color = 'C1', label = 'Proposed', linewidth=2)
            ax[axis_index].plot(t_est_vio_MB, error_MB[:, axis_index], color = 'C4', label = 'MB', linewidth=2)
            ax[axis_index].plot(t_gt, np.zeros_like(t_gt), color = 'C0', linestyle='--')
            ax[axis_index].grid(True)
            if y_limits[axis_index] != None:
                ax[axis_index].set_ylim(y_limits[axis_index])
            ax[axis_index].set_ylabel(y_label[axis_index], fontsize=13)
        ax[axis_index].set_xlabel(x_label, fontsize=13)
        ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.40), fontsize=13)
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(f"{dir_name}/{seq}_position_errorVIO.pdf")
        plt.close('all')

if __name__ == '__main__':
    main()