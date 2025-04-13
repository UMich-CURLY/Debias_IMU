import numpy as np
import sys
import os
import pandas as pd

from evo import entry_points
import glob

from evo import entry_points



def euroc_convert_to_evo_format(filename_old, filename_new):
    mat = np.genfromtxt(filename_old, delimiter=' ')
    mat = mat[:,:8]
    np.savetxt(filename_new, mat, delimiter=' ')

def main():
    flag_eva_tra = True
    flag_compare = True
    
    ### EUROC ###
    methods = ['Axb', 'MBrossard', 'Proposed', 'Raw']
    vio_dir_name = ['Axb', 'MB', 'proposed', 'rawimu']
    sequences = ['MH_02_easy', 'MH_04_difficult', 'V1_01_easy', 'V1_03_difficult', 'V2_02_medium']
    # sequences = ['MH_02_easy']
    # sequences = []
    rpe_len_list = [5, 10, 15, 20]

    for seq in sequences:
        dir_name = f"evo_evaluation/cache/{seq}"
        os.makedirs(dir_name, exist_ok=True)
        tmp_save_dir = f"evo_evaluation/cache/{seq}/tmp"
        os.makedirs(tmp_save_dir, exist_ok=True)
        for method_index, method in enumerate(methods):
            if flag_eva_tra:
                gt_file = "/home/ben/Documents/fork_others/openvins_docker/worksapce/src/open_vins/ov_data/euroc_mav/" + seq +".csv"
                test_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/results/Euroc/SE3fromNet/{method}/{seq}/00_estimate.txt"
                est_file_cashe = tmp_save_dir + "/" + method + "_" + seq + ".txt"
                euroc_convert_to_evo_format(test_file, est_file_cashe)
                vio_est_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/results/Euroc/VIO_stereo/{vio_dir_name[method_index]}/{seq}/00_estimate.txt"
                vio_est_file_cashe = tmp_save_dir + "/vio_" + vio_dir_name[method_index] + "_" + seq + ".txt"
                euroc_convert_to_evo_format(vio_est_file, vio_est_file_cashe)

                ## pure integration
                # APE
                if os.path.exists(f"{dir_name}/ape_rot_{method}_{seq}.zip"):
                    os.remove(f"{dir_name}/ape_rot_{method}_{seq}.zip")
                sys.argv = ["evo_ape", 'euroc', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/ape_rot_{method}_{seq}.zip",
                    "-r", "angle_deg", "--align_origin"]
                entry_points.ape()
                if os.path.exists(f"{dir_name}/ape_trans_{method}_{seq}.zip"):
                    os.remove(f"{dir_name}/ape_trans_{method}_{seq}.zip")
                sys.argv = ["evo_ape", 'euroc', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/ape_trans_{method}_{seq}.zip",
                    "-r", "trans_part", "--align_origin"]
                entry_points.ape()
                # RPE
                for len_rpe in rpe_len_list:
                    if os.path.exists(f"{dir_name}/rpe_rot_{method}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/rpe_rot_{method}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'euroc', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/rpe_rot_{method}_{len_rpe}_{seq}.zip",
                        "-r", "angle_deg", #"--align_origin", 
                        f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()
                    if os.path.exists(f"{dir_name}/rpe_trans_{method}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/rpe_trans_{method}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'euroc', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/rpe_trans_{method}_{len_rpe}_{seq}.zip",
                        "-r", "trans_part", "--align_origin", f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()

                ## VIO
                # APE
                if os.path.exists(f"{dir_name}/vio_ape_rot_{vio_dir_name[method_index]}_{seq}.zip"):
                    os.remove(f"{dir_name}/vio_ape_rot_{vio_dir_name[method_index]}_{seq}.zip")
                sys.argv = ["evo_ape", 'euroc', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_ape_rot_{vio_dir_name[method_index]}_{seq}.zip",
                    "-r", "angle_deg", "--align_origin"]
                entry_points.ape()
                # assert False
                if os.path.exists(f"{dir_name}/vio_ape_trans_{vio_dir_name[method_index]}_{seq}.zip"):
                    os.remove(f"{dir_name}/vio_ape_trans_{vio_dir_name[method_index]}_{seq}.zip")
                sys.argv = ["evo_ape", 'euroc', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_ape_trans_{vio_dir_name[method_index]}_{seq}.zip",
                    "-r", "trans_part", "--align_origin"]
                entry_points.ape()
                # RPE
                for len_rpe in rpe_len_list:
                    if os.path.exists(f"{dir_name}/vio_rpe_rot_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/vio_rpe_rot_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'euroc', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_rpe_rot_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip",
                        "-r", "angle_deg", "--align_origin", 
                        f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()
                    if os.path.exists(f"{dir_name}/vio_rpe_trans_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/vio_rpe_trans_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'euroc', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_rpe_trans_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip",
                        "-r", "trans_part", "--align_origin", f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()
                          
                                 
        ## save pure integration results
        if flag_compare:
            os.remove(f"{dir_name}/ape_rot_table.csv") if os.path.exists(f"{dir_name}/ape_rot_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/ape_rot_*.zip'), "--save_table", f"{dir_name}/ape_rot_table.csv"]
            entry_points.res()
            os.remove(f"{dir_name}/ape_trans_table.csv") if os.path.exists(f"{dir_name}/ape_trans_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/ape_trans_*.zip'), "--save_table", f"{dir_name}/ape_trans_table.csv"]
            entry_points.res()
            for len_rpe in rpe_len_list:
                os.remove(f"{dir_name}/rpe_rot_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/rpe_rot_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/rpe_rot_*_{len_rpe}_*.zip'),  "--save_table", f"{dir_name}/rpe_rot_{len_rpe}_table.csv"]
                entry_points.res()
                os.remove(f"{dir_name}/rpe_trans_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/rpe_trans_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/rpe_trans_*_{len_rpe}_*.zip'),  "--save_table", f"{dir_name}/rpe_trans_{len_rpe}_table.csv"]
                entry_points.res()
            ## save VIO results
            os.remove(f"{dir_name}/vio_ape_rot_table.csv") if os.path.exists(f"{dir_name}/vio_ape_rot_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_ape_rot_*.zip'),  "--save_table", f"{dir_name}/vio_ape_rot_table.csv"]
            entry_points.res()
            os.remove(f"{dir_name}/vio_ape_trans_table.csv") if os.path.exists(f"{dir_name}/vio_ape_trans_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_ape_trans_*.zip'),  "--save_table", f"{dir_name}/vio_ape_trans_table.csv"]
            entry_points.res()
            for len_rpe in rpe_len_list:
                os.remove(f"{dir_name}/vio_rpe_rot_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/vio_rpe_rot_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_rpe_rot_*_{len_rpe}_*.zip'), "--save_table", f"{dir_name}/vio_rpe_rot_{len_rpe}_table.csv"]
                entry_points.res()
                os.remove(f"{dir_name}/vio_rpe_trans_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/vio_rpe_trans_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_rpe_trans_*_{len_rpe}_*.zip'), "--save_table", f"{dir_name}/vio_rpe_trans_{len_rpe}_table.csv"]
                entry_points.res()
                                           
    
    def read_csv(filename):
        df = pd.read_csv(filename)
        df = df.set_index(df.columns[0]) 
        desired_order = ['Raw', 'Axb', 'MBrossard', 'Proposed']
        desired_order_alt = ['rawimu', 'Axb', 'MB', 'proposed']
        rmse_values = []
        for keyword in desired_order:
            row = df[df.index.str.contains(keyword)]
            if not row.empty:
                rmse_values.append(row.iloc[0]['rmse'])
            else:
                row = df[df.index.str.contains(desired_order_alt[desired_order.index(keyword)])]
                if not row.empty:
                    rmse_values.append(row.iloc[0]['rmse'])
                else:
                    raise ValueError(f"Keyword '{keyword}' not found in the index.")
        rmse_array = np.array(rmse_values).reshape((4))
        return rmse_array
    
    def format_array2str(array):
        """array: 4"""
        min_val = np.min(array)
        array_str = []
        for i in range(array.shape[0]):
            if array[i] == min_val:
                array_str.append(f"\\textbf{{{array[i]:.2f}}}")
            else:
                array_str.append(f"{array[i]:.2f}")
        return array_str

    # Latex APE table
    save_str = []
    print("EUROC APE pure integration and VIO")
    save_str.append("EUROC APE pure integration and VIO")
    sequences = ['MH_02_easy', 'MH_04_difficult', 'V1_01_easy', 'V1_03_difficult', 'V2_02_medium']
    seq_name_latex = ['MH\\_02', 'MH\\_04', 'V1\\_01', 'V1\\_03', 'V2\\_02']
    aoe_avergae = []
    ape_avergae = []
    aoe_vio_avergae = []
    ape_vio_avergae = []
    for seq in sequences:
        tmp = f"evo_evaluation/cache/{seq}"
        aoe = read_csv(f"{tmp}/ape_rot_table.csv") # [4]
        aoe_avergae.append(aoe)
        aoe = format_array2str(aoe)
        ape = read_csv(f"{tmp}/ape_trans_table.csv") # [4]
        ape_avergae.append(ape)
        ape = format_array2str(ape)
        aoe_vio = read_csv(f"{tmp}/vio_ape_rot_table.csv")
        aoe_vio_avergae.append(aoe_vio)
        aoe_vio = format_array2str(aoe_vio)
        ape_vio = read_csv(f"{tmp}/vio_ape_trans_table.csv")
        ape_vio_avergae.append(ape_vio)
        ape_vio = format_array2str(ape_vio)
        
        print(f"& {seq_name_latex[sequences.index(seq)]} & & {aoe[0]}/- & {aoe[1]}/{ape[1]} & {aoe[2]}/{ape[2]} & {aoe[3]}/{ape[3]} & & {aoe_vio[0]}/{ape_vio[0]} & {aoe_vio[1]}/{ape_vio[1]} &  {aoe_vio[2]}/{ape_vio[2]} & {aoe_vio[3]}/{ape_vio[3]} \\\\")
        save_str.append(f"& {seq_name_latex[sequences.index(seq)]} & & {aoe[0]}/- & {aoe[1]}/{ape[1]} & {aoe[2]}/{ape[2]} & {aoe[3]}/{ape[3]} & & {aoe_vio[0]}/{ape_vio[0]} & {aoe_vio[1]}/{ape_vio[1]} &  {aoe_vio[2]}/{ape_vio[2]} & {aoe_vio[3]}/{ape_vio[3]} \\\\")
    aoe_avergae = np.array(aoe_avergae)
    ape_avergae = np.array(ape_avergae)
    aoe_vio_avergae = np.array(aoe_vio_avergae)
    ape_vio_avergae = np.array(ape_vio_avergae)
    aoe_avergae = format_array2str(np.mean(aoe_avergae, axis=0))
    ape_avergae = format_array2str(np.mean(ape_avergae, axis=0))
    aoe_vio_avergae = format_array2str(np.mean(aoe_vio_avergae, axis=0))
    ape_vio_avergae = format_array2str(np.mean(ape_vio_avergae, axis=0))
    print(f"& \\textbf{{Average}} & & {aoe_avergae[0]}/- & {aoe_avergae[1]}/{ape_avergae[1]} & {aoe_avergae[2]}/{ape_avergae[2]} & {aoe_avergae[3]}/{ape_avergae[3]} & & {aoe_vio_avergae[0]}/{ape_vio_avergae[0]} & {aoe_vio_avergae[1]}/{ape_vio_avergae[1]} & {aoe_vio_avergae[2]}/{ape_vio_avergae[2]} & {aoe_vio_avergae[3]}/{ape_vio_avergae[3]} \\\\")
    save_str.append(f"& \\textbf{{Average}} & & {aoe_avergae[0]}/- & {aoe_avergae[1]}/{ape_avergae[1]} & {aoe_avergae[2]}/{ape_avergae[2]} & {aoe_avergae[3]}/{ape_avergae[3]} & & {aoe_vio_avergae[0]}/{ape_vio_avergae[0]} & {aoe_vio_avergae[1]}/{ape_vio_avergae[1]} & {aoe_vio_avergae[2]}/{ape_vio_avergae[2]} & {aoe_vio_avergae[3]}/{ape_vio_avergae[3]} \\\\")
    
    save_str.append("\n")
    print("EUROC RPE pure integration and VIO")
    save_str.append("EUROC RPE pure integration and VIO")
    for len_rpe in rpe_len_list:
        roe_tmp = []
        rpe_tmp = []
        vio_roe_tmp = []
        vio_rpe_tmp = []
        for seq in sequences:
            tmp = f"evo_evaluation/cache/{seq}"
            roe = read_csv(f"{tmp}/rpe_rot_{len_rpe}_table.csv") # [4]
            roe_tmp.append(roe)
            rpe = read_csv(f"{tmp}/rpe_trans_{len_rpe}_table.csv") # [4]
            rpe_tmp.append(rpe)
            vio_roe = read_csv(f"{tmp}/vio_rpe_rot_{len_rpe}_table.csv")
            vio_roe_tmp.append(vio_roe)
            vio_rpe = read_csv(f"{tmp}/vio_rpe_trans_{len_rpe}_table.csv")
            vio_rpe_tmp.append(vio_rpe)
                                            
        roe_tmp = np.array(roe_tmp) # [5, 4]
        roe_tmp = np.mean(roe_tmp, axis=0) # [4]
        rpe_tmp = np.array(rpe_tmp)
        rpe_tmp = np.mean(rpe_tmp, axis=0)
        vio_roe_tmp = np.array(vio_roe_tmp) 
        vio_roe_tmp = np.mean(vio_roe_tmp, axis=0)
        vio_rpe_tmp = np.array(vio_rpe_tmp)
        vio_rpe_tmp = np.mean(vio_rpe_tmp, axis=0)
        roe_tmp = format_array2str(roe_tmp)
        rpe_tmp = format_array2str(rpe_tmp)
        vio_roe_tmp = format_array2str(vio_roe_tmp)
        vio_rpe_tmp = format_array2str(vio_rpe_tmp)

        print(f"& {len_rpe} & & {roe_tmp[0]}/{rpe_tmp[0]} & {roe_tmp[1]}/{rpe_tmp[1]} & {roe_tmp[2]}/{rpe_tmp[2]} & {roe_tmp[3]}/{rpe_tmp[3]} & & {vio_roe_tmp[0]}/{vio_rpe_tmp[0]} & {vio_roe_tmp[1]}/{vio_rpe_tmp[1]} & {vio_roe_tmp[2]}/{vio_rpe_tmp[2]} & {vio_roe_tmp[3]}/{vio_rpe_tmp[3]} \\\\")
        save_str.append(f"& {len_rpe} & & {roe_tmp[0]}/{rpe_tmp[0]} & {roe_tmp[1]}/{rpe_tmp[1]} & {roe_tmp[2]}/{rpe_tmp[2]} & {roe_tmp[3]}/{rpe_tmp[3]} & & {vio_roe_tmp[0]}/{vio_rpe_tmp[0]} & {vio_roe_tmp[1]}/{vio_rpe_tmp[1]} & {vio_roe_tmp[2]}/{vio_rpe_tmp[2]} & {vio_roe_tmp[3]}/{vio_rpe_tmp[3]} \\\\")

    with open('evo_evaluation/results.txt', 'w') as f:
        for item in save_str:
            f.write("%s\n" % item)

    

    ### TUM ###
    methods = ['Axb', 'MBrossard', 'Proposed', 'Raw']
    vio_dir_name = ['Axb', 'MBrossard', 'proposed', 'rawimu']
    sequences = ['dataset-room2', 'dataset-room4', 'dataset-room6']
    rpe_len_list = [5, 10, 15, 20]

    for seq in sequences:
        dir_name = f"evo_evaluation/cache/{seq}"
        os.makedirs(dir_name, exist_ok=True)
        tmp_save_dir = f"evo_evaluation/cache/{seq}/tmp"
        os.makedirs(tmp_save_dir, exist_ok=True)
        for method_index, method in enumerate(methods):
            if flag_eva_tra:
                gt_file = "/home/ben/Documents/fork_others/openvins_docker/worksapce/src/open_vins/ov_data/tum_vi_test/" + seq +".txt"
                test_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/results/Tum/SE3fromNet/{method}/{seq}/00_estimate.txt"
                est_file_cashe = tmp_save_dir + "/" + method + "_" + seq + ".txt"
                euroc_convert_to_evo_format(test_file, est_file_cashe)
                vio_est_file = f"/home/ben/Documents/fork_others/openvins_docker/worksapce/results/Tum/VIO_stereo/{vio_dir_name[method_index]}/{seq}/00_estimate.txt"
                vio_est_file_cashe = tmp_save_dir + "/vio_" + vio_dir_name[method_index] + "_" + seq + ".txt"
                euroc_convert_to_evo_format(vio_est_file, vio_est_file_cashe)

                ## pure integration
                # APE
                if os.path.exists(f"{dir_name}/ape_rot_{method}_{seq}.zip"):
                    os.remove(f"{dir_name}/ape_rot_{method}_{seq}.zip")
                sys.argv = ["evo_ape", 'tum', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/ape_rot_{method}_{seq}.zip",
                    "-r", "angle_deg", "--align_origin"]
                entry_points.ape()
                if os.path.exists(f"{dir_name}/ape_trans_{method}_{seq}.zip"):
                    os.remove(f"{dir_name}/ape_trans_{method}_{seq}.zip")
                sys.argv = ["evo_ape", 'tum', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/ape_trans_{method}_{seq}.zip",
                    "-r", "trans_part", "--align_origin"]
                entry_points.ape()
                # RPE
                for len_rpe in rpe_len_list:
                    if os.path.exists(f"{dir_name}/rpe_rot_{method}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/rpe_rot_{method}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'tum', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/rpe_rot_{method}_{len_rpe}_{seq}.zip",
                        "-r", "angle_deg", #"--align_origin", 
                        f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()
                    if os.path.exists(f"{dir_name}/rpe_trans_{method}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/rpe_trans_{method}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'tum', gt_file, est_file_cashe, "-v", "--save_results", f"{dir_name}/rpe_trans_{method}_{len_rpe}_{seq}.zip",
                        "-r", "trans_part", "--align_origin", f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()

                ## VIO
                # APE
                if os.path.exists(f"{dir_name}/vio_ape_rot_{vio_dir_name[method_index]}_{seq}.zip"):
                    os.remove(f"{dir_name}/vio_ape_rot_{vio_dir_name[method_index]}_{seq}.zip")
                sys.argv = ["evo_ape", 'tum', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_ape_rot_{vio_dir_name[method_index]}_{seq}.zip",
                    "-r", "angle_deg", "--align_origin"]
                entry_points.ape()
                # assert False
                if os.path.exists(f"{dir_name}/vio_ape_trans_{vio_dir_name[method_index]}_{seq}.zip"):
                    os.remove(f"{dir_name}/vio_ape_trans_{vio_dir_name[method_index]}_{seq}.zip")
                sys.argv = ["evo_ape", 'tum', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_ape_trans_{vio_dir_name[method_index]}_{seq}.zip",
                    "-r", "trans_part", "--align_origin"]
                entry_points.ape()
                # RPE
                for len_rpe in rpe_len_list:
                    if os.path.exists(f"{dir_name}/vio_rpe_rot_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/vio_rpe_rot_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'tum', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_rpe_rot_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip",
                        "-r", "angle_deg", "--align_origin", 
                        f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()
                    if os.path.exists(f"{dir_name}/vio_rpe_trans_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip"):
                        os.remove(f"{dir_name}/vio_rpe_trans_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip")
                    sys.argv = ["evo_rpe", 'tum', gt_file, vio_est_file_cashe, "-v", "--save_results", f"{dir_name}/vio_rpe_trans_{vio_dir_name[method_index]}_{len_rpe}_{seq}.zip",
                        "-r", "trans_part", "--align_origin", f"-d={len_rpe}", "-u=m", "--pairs_from_reference", "--all_pairs"]
                    entry_points.rpe()
                          
                                 
        ## save pure integration results
        if flag_compare:
            os.remove(f"{dir_name}/ape_rot_table.csv") if os.path.exists(f"{dir_name}/ape_rot_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/ape_rot_*.zip'), "--save_table", f"{dir_name}/ape_rot_table.csv"]
            entry_points.res()
            os.remove(f"{dir_name}/ape_trans_table.csv") if os.path.exists(f"{dir_name}/ape_trans_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/ape_trans_*.zip'), "--save_table", f"{dir_name}/ape_trans_table.csv"]
            entry_points.res()
            for len_rpe in rpe_len_list:
                os.remove(f"{dir_name}/rpe_rot_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/rpe_rot_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/rpe_rot_*_{len_rpe}_*.zip'),  "--save_table", f"{dir_name}/rpe_rot_{len_rpe}_table.csv"]
                entry_points.res()
                os.remove(f"{dir_name}/rpe_trans_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/rpe_trans_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/rpe_trans_*_{len_rpe}_*.zip'),  "--save_table", f"{dir_name}/rpe_trans_{len_rpe}_table.csv"]
                entry_points.res()
            ## save VIO results
            os.remove(f"{dir_name}/vio_ape_rot_table.csv") if os.path.exists(f"{dir_name}/vio_ape_rot_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_ape_rot_*.zip'),  "--save_table", f"{dir_name}/vio_ape_rot_table.csv"]
            entry_points.res()
            os.remove(f"{dir_name}/vio_ape_trans_table.csv") if os.path.exists(f"{dir_name}/vio_ape_trans_table.csv") else None
            sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_ape_trans_*.zip'),  "--save_table", f"{dir_name}/vio_ape_trans_table.csv"]
            entry_points.res()
            for len_rpe in rpe_len_list:
                os.remove(f"{dir_name}/vio_rpe_rot_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/vio_rpe_rot_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_rpe_rot_*_{len_rpe}_*.zip'), "--save_table", f"{dir_name}/vio_rpe_rot_{len_rpe}_table.csv"]
                entry_points.res()
                os.remove(f"{dir_name}/vio_rpe_trans_{len_rpe}_table.csv") if os.path.exists(f"{dir_name}/vio_rpe_trans_{len_rpe}_table.csv") else None
                sys.argv = ["evo_res",  *glob.glob(f'{dir_name}/vio_rpe_trans_*_{len_rpe}_*.zip'), "--save_table", f"{dir_name}/vio_rpe_trans_{len_rpe}_table.csv"]
                entry_points.res()
                                           

    # Latex APE table
    save_str = []
    print("TUM APE pure integration and VIO")
    save_str.append("TUM APE pure integration and VIO")
    sequences = ['dataset-room2', 'dataset-room4', 'dataset-room6']
    seq_name_latex = ['Room2', 'Room4', 'Room6']
    aoe_avergae = []
    ape_avergae = []
    aoe_vio_avergae = []
    ape_vio_avergae = []
    for seq in sequences:
        tmp = f"evo_evaluation/cache/{seq}"
        aoe = read_csv(f"{tmp}/ape_rot_table.csv") # [4]
        aoe_avergae.append(aoe)
        aoe = format_array2str(aoe)
        ape = read_csv(f"{tmp}/ape_trans_table.csv") # [4]
        ape_avergae.append(ape)
        ape = format_array2str(ape)
        aoe_vio = read_csv(f"{tmp}/vio_ape_rot_table.csv")
        aoe_vio_avergae.append(aoe_vio)
        aoe_vio = format_array2str(aoe_vio)
        ape_vio = read_csv(f"{tmp}/vio_ape_trans_table.csv")
        ape_vio_avergae.append(ape_vio)
        ape_vio = format_array2str(ape_vio)
        
        print(f"& {seq_name_latex[sequences.index(seq)]} & & {aoe[0]}/- & {aoe[1]}/{ape[1]} & {aoe[2]}/{ape[2]} & {aoe[3]}/{ape[3]} & & failed & {aoe_vio[1]}/{ape_vio[1]} &  {aoe_vio[2]}/{ape_vio[2]} & {aoe_vio[3]}/{ape_vio[3]} \\\\")
        save_str.append(f"& {seq_name_latex[sequences.index(seq)]} & & {aoe[0]}/- & {aoe[1]}/{ape[1]} & {aoe[2]}/{ape[2]} & {aoe[3]}/{ape[3]} & & failed & {aoe_vio[1]}/{ape_vio[1]} &  {aoe_vio[2]}/{ape_vio[2]} & {aoe_vio[3]}/{ape_vio[3]} \\\\")
    aoe_avergae = np.array(aoe_avergae)
    ape_avergae = np.array(ape_avergae)
    aoe_vio_avergae = np.array(aoe_vio_avergae)
    ape_vio_avergae = np.array(ape_vio_avergae)
    aoe_avergae = format_array2str(np.mean(aoe_avergae, axis=0))
    ape_avergae = format_array2str(np.mean(ape_avergae, axis=0))
    aoe_vio_avergae = format_array2str(np.mean(aoe_vio_avergae, axis=0))
    ape_vio_avergae = format_array2str(np.mean(ape_vio_avergae, axis=0))
    print(f"& \\textbf{{Average}} & & {aoe_avergae[0]}/- & {aoe_avergae[1]}/{ape_avergae[1]} & {aoe_avergae[2]}/{ape_avergae[2]} & {aoe_avergae[3]}/{ape_avergae[3]} & & failed & {aoe_vio_avergae[1]}/{ape_vio_avergae[1]} & {aoe_vio_avergae[2]}/{ape_vio_avergae[2]} & {aoe_vio_avergae[3]}/{ape_vio_avergae[3]} \\\\")
    save_str.append(f"& \\textbf{{Average}} & & {aoe_avergae[0]}/- & {aoe_avergae[1]}/{ape_avergae[1]} & {aoe_avergae[2]}/{ape_avergae[2]} & {aoe_avergae[3]}/{ape_avergae[3]} & & failed & {aoe_vio_avergae[1]}/{ape_vio_avergae[1]} & {aoe_vio_avergae[2]}/{ape_vio_avergae[2]} & {aoe_vio_avergae[3]}/{ape_vio_avergae[3]} \\\\")
    
    save_str.append("\n")
    print("TUM RPE pure integration and VIO")
    save_str.append("TUM RPE pure integration and VIO")
    for len_rpe in rpe_len_list:
        roe_tmp = []
        rpe_tmp = []
        vio_roe_tmp = []
        vio_rpe_tmp = []
        for seq in sequences:
            tmp = f"evo_evaluation/cache/{seq}"
            roe = read_csv(f"{tmp}/rpe_rot_{len_rpe}_table.csv") # [4]
            roe_tmp.append(roe)
            rpe = read_csv(f"{tmp}/rpe_trans_{len_rpe}_table.csv") # [4]
            rpe_tmp.append(rpe)
            vio_roe = read_csv(f"{tmp}/vio_rpe_rot_{len_rpe}_table.csv")
            vio_roe_tmp.append(vio_roe)
            vio_rpe = read_csv(f"{tmp}/vio_rpe_trans_{len_rpe}_table.csv")
            vio_rpe_tmp.append(vio_rpe)
                                            
        roe_tmp = np.array(roe_tmp) # [5, 4]
        roe_tmp = np.mean(roe_tmp, axis=0) # [4]
        rpe_tmp = np.array(rpe_tmp)
        rpe_tmp = np.mean(rpe_tmp, axis=0)
        vio_roe_tmp = np.array(vio_roe_tmp) 
        vio_roe_tmp = np.mean(vio_roe_tmp, axis=0)
        vio_rpe_tmp = np.array(vio_rpe_tmp)
        vio_rpe_tmp = np.mean(vio_rpe_tmp, axis=0)
        roe_tmp = format_array2str(roe_tmp)
        rpe_tmp = format_array2str(rpe_tmp)
        vio_roe_tmp = format_array2str(vio_roe_tmp)
        vio_rpe_tmp = format_array2str(vio_rpe_tmp)

        print(f"& {len_rpe} & & {roe_tmp[0]}/{rpe_tmp[0]} & {roe_tmp[1]}/{rpe_tmp[1]} & {roe_tmp[2]}/{rpe_tmp[2]} & {roe_tmp[3]}/{rpe_tmp[3]} & & failed & {vio_roe_tmp[1]}/{vio_rpe_tmp[1]} & {vio_roe_tmp[2]}/{vio_rpe_tmp[2]} & {vio_roe_tmp[3]}/{vio_rpe_tmp[3]} \\\\")
        save_str.append(f"& {len_rpe} & & {roe_tmp[0]}/{rpe_tmp[0]} & {roe_tmp[1]}/{rpe_tmp[1]} & {roe_tmp[2]}/{rpe_tmp[2]} & {roe_tmp[3]}/{rpe_tmp[3]} & & failed & {vio_roe_tmp[1]}/{vio_rpe_tmp[1]} & {vio_roe_tmp[2]}/{vio_rpe_tmp[2]} & {vio_roe_tmp[3]}/{vio_rpe_tmp[3]} \\\\")

    with open('evo_evaluation/results.txt', 'a') as f:
        f.write("\n")
        for item in save_str:
            f.write("%s\n" % item)



if __name__ == '__main__':
    main()