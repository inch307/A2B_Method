import duchi
import cfo
import topm
import numpy as np
import bound_tuning
import pm
import noutput
import re
import glob
import pickle

import optimize_avg
import optimize_worst
import os

def find_or_generate_best_n(eps_k, obj, n_range=(2, 16)):
    if obj == 'avg':
        results_dir = 'results_average_free_a/'
        var_key = 'avg_var'
        file_pattern = f"opt_results_eps{eps_k}_N*.pkl"
        optimizer_func = optimize_avg.optimize
    elif obj == 'worst':
        results_dir = 'results_worst_case_free_a/'
        var_key = 'worst_var'
        file_pattern = f"opt_results_worst_case_eps{eps_k}_N*.pkl"
        optimizer_func = optimize_worst.optimize
    else:
        print(f" Error: Unknown objective '{obj}'. Please use 'avg' or 'worst'.")
        return None

    os.makedirs(results_dir, exist_ok=True)

    search_path = os.path.join(results_dir, file_pattern)
    pkl_files = glob.glob(search_path)

    if not pkl_files:
        print(f"No result files found for eps_k = {eps_k}. Starting file generation...")
        print(f"   Generating for N in range [{n_range[0]}, {n_range[1]}]...")

        for n_val in range(n_range[0], n_range[1] + 1):
            print(f"   Running optimization for N={n_val}...")
            try:
                optimizer_func(eps_k, n_val)
            except Exception as e:
                print(f"   Error during optimization for N={n_val}: {e}")
        
        print("File generation complete. Re-searching for the best N...")
        pkl_files = glob.glob(search_path)

    if not pkl_files:
        print(f" Still no result files found after attempting generation for eps_k = {eps_k}.")
        return None

    best_N = None
    min_variance = float('inf')

    for pkl_path in pkl_files:
        try:
            match = re.search(r'_N(\d+)\.pkl$', pkl_path)
            if not match:
                continue
            
            N = int(match.group(1))
            if N > n_range[1]: 
                continue

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            variance = data['metadata'].get(var_key, np.nan)
            
            if not np.isnan(variance) and variance < min_variance:
                min_variance = variance
                best_N = N
        except Exception as e:
            print(f"  - Error processing file {pkl_path}: {e}")

    print("\n" + "="*40)
    print(" Final Result ")
    if best_N is not None:
        print(f"For eps_k = {eps_k} (objective: '{obj}'):")
        print(f"   Optimal N is {best_N}")
        print(f"   With minimum variance: {min_variance:.6f}")
    else:
        print(f"Could not find a valid result for eps_k = {eps_k}.")
    print("="*40)
    
    return best_N

def get_duchi(d, eps, rng, eps_ratio):
    duchi_mechanism = duchi.Duchi(d, eps * eps_ratio, rng, 1., k=1)
    direct_encoding = cfo.DE(d, eps * (1-eps_ratio), 3, rng)
    return duchi_mechanism, direct_encoding, duchi_mechanism.Duchi_batch

def get_to(d, eps, rng, eps_ratio):
    three_output_mechanism = topm.TOPM(d, eps * eps_ratio, rng, 1., k=1)
    direct_encoding = cfo.DE(d, eps * (1-eps_ratio), 3, rng)
    return three_output_mechanism, direct_encoding, three_output_mechanism.TO_batch

def get_pmsub(d, eps, rng, eps_ratio):
    pmsub_mechanism = topm.TOPM(d, eps * eps_ratio, rng, 1., k=1)
    direct_encoding = cfo.DE(d, eps * (1-eps_ratio), 3, rng)
    return pmsub_mechanism, direct_encoding, pmsub_mechanism.PM_batch

def get_topm(d, eps, rng, eps_ratio):
    topm_mechanism = topm.TOPM(d, eps * eps_ratio, rng, 1., k=1)
    direct_encoding = cfo.DE(d, eps * (1-eps_ratio), 3, rng)
    return topm_mechanism, direct_encoding, topm_mechanism.HM_batch

def get_pm(d, eps, rng, eps_ratio):
    pm_mechanism = pm.PM(d, eps*eps_ratio, rng, 1., k=1)
    direct_encoding = cfo.DE(d, eps * (1-eps_ratio), 3, rng)
    return pm_mechanism, direct_encoding, pm_mechanism.PM_batch

def get_no(d, eps, rng, eps_ratio, args):
    eps_k = eps * eps_ratio
    if args.N is None:
        print(args.N)
        best_N = find_or_generate_best_n(eps_k, args.obj)
        args.N = best_N
        print(best_N)
    
    print(f'args N is {args.N}')
    if args.mech == 'noutput':
        if args.obj == 'avg':
            pkl_path = f'results_average_free_a/opt_results_eps{eps_k}_N{args.N}.pkl'
        elif args.obj == 'worst':
            pkl_path = f'results_worst_case_free_a/opt_results_worst_case_eps{eps_k}_N{args.N}.pkl'
        try:
            no_mechanism = noutput.Noutput(pkl_path)
            print(f"LDPMechanism: {pkl_path}")
        except FileNotFoundError as e:
            if args.obj == 'avg':
                optimize_avg.optimize(eps_k, args.N)
            elif args.obj == 'worst':
                optimize_worst.optimize(eps_k, args.N)
            no_mechanism = noutput.Noutput(pkl_path)
    direct_encoding = cfo.DE(d, eps * (1-eps_ratio), 3, rng)
    return no_mechanism, direct_encoding, no_mechanism.perturb_batch

def get_bt(eps, eps_ratio, l ,r, alpha, lr, zeta, tau):
    bt = bound_tuning.BoundTuning(eps * (1-eps_ratio), l=l, r=r, alpha=alpha, lr=lr, zeta=zeta, tau=tau)
    return bt

def perturb(data, de_mechanism, numeric_perturb_func, bt): 
    transformed_data, clipped_lower_idx, clipped_upper_idx, no_clip_idx = bt.transform(data)
    clipped_flag = np.zeros((len(data), 3))
    clipped_flag[clipped_lower_idx, 0] = 1.
    clipped_flag[clipped_upper_idx, 1] = 1.
    clipped_flag[no_clip_idx, 2] = 1.
    clipped_flag = np.argmax(clipped_flag, axis=1)

    true_mean = np.mean(data)
    theta_hat = np.mean(bt.inverse_transform(numeric_perturb_func(transformed_data)))

    unique, counts = np.unique(clipped_flag, return_counts=True)
    true_hist = counts / np.sum(counts)
    est_hist = de_mechanism.batch(clipped_flag)
        # print(f'true: {true_hist}, est: {est_hist}')
    return true_mean, theta_hat, true_hist, est_hist

def perturb_no_de(data, numeric_perturb_func, bt): 
    transformed_data, clipped_lower_idx, clipped_upper_idx, no_clip_idx = bt.transform(data)
    clipped_flag = np.zeros((len(data), 3))
    clipped_flag[clipped_lower_idx, 0] = 1.
    clipped_flag[clipped_upper_idx, 1] = 1.
    clipped_flag[no_clip_idx, 2] = 1.
    clipped_flag = np.argmax(clipped_flag, axis=1)

    true_mean = np.mean(data)
    theta_hat = np.mean(bt.inverse_transform(numeric_perturb_func(transformed_data)))
    return true_mean, theta_hat