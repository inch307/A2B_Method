# EXP.py
import jax
import jax.numpy as jnp
# Scipy와의 호환성을 위해 64비트 정밀도 사용 설정
jax.config.update("jax_enable_x64", True)

import numpy as np
import os
import math

import data_load
import argparse
import pickle

import ldp

def get_mech(mech_name, k, eps, rng, eps_ratio, args):
    if mech_name == 'duchi':
        mech, de, perturb_func = ldp.get_duchi(k, eps, rng, eps_ratio)
    elif mech_name == 'to':
        mech, de, perturb_func = ldp.get_to(k, eps, rng, eps_ratio)
    elif mech_name == 'pmsub':
        mech, de, perturb_func = ldp.get_pmsub(k, eps, rng, eps_ratio)
    elif mech_name == 'topm':
        mech, de, perturb_func = ldp.get_topm(k, eps, rng, eps_ratio)
    elif mech_name == 'pm':
        mech, de, perturb_func = ldp.get_pm(k, eps, rng, eps_ratio)
    elif mech_name == 'noutput':
        mech, de, perturb_func = ldp.get_no(k, eps, rng, eps_ratio, args)
    else:
        raise ValueError(f'Mechanism config error: {mech_name}')
    
    return mech, de, perturb_func

parser = argparse.ArgumentParser()
parser.add_argument('--sam', type=int, default=100000)
parser.add_argument('--seed', type = int, default=2025)
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--dataset', default='trunc_normal')
parser.add_argument('--mech', default='topm')
parser.add_argument('--N', type=int)
parser.add_argument('--obj', default='avg', help='Obj (avg or worst) for N-output mechanism')
parser.add_argument('--log_dir', default='logs')

parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.3) 
parser.add_argument('--eps_ratio', type=float, default=0.7)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--zeta', type=float, default=0.3)
parser.add_argument('--tau', type=float, default=2)

args = parser.parse_args()
rng = np.random.default_rng(seed=args.seed)
np.random.seed(args.seed)

log_data = {}
log_data_no_bt = {}

if args.dataset == 'truncNormal':
    data, mean, var = data_load.load_trunc_normal_data(0, 1, -1, 1, (args.sam,))
else:
    data, mean, var = data_load.data_load(args.dataset)
data_min = min(data)
data_max = max(data)
data_center = (data_min + data_max) / 2
data = rng.permutation(data)
num_data = len(data)

log_dir = os.path.join(args.log_dir, f'bt/{args.dataset}/{args.mech}/{args.lr}/{args.alpha}/{args.eps_ratio}/{args.steps}/{args.zeta}/{args.tau}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Directory created: {log_dir}")
else:
    print(f"Directory already exists: {log_dir}")

base_log_dir = os.path.join(args.log_dir, f'base/{args.dataset}/{args.mech}')
if not os.path.exists(base_log_dir):
    os.makedirs(base_log_dir)
    print(f"Directory created: {base_log_dir}")
else:
    print(f"Directory already exists: {base_log_dir}")

eps_lst = [1.0, 2.0, 3.0, 4.0, 5.0]
scales = [1/2, 2/3, 1.0, 1.5, 2.0]

mechanism = args.mech
log_data['args'] = vars(args)
log_data['steps'] = {}
log_data['lrs'] = {}

for eps in eps_lst:
    log_data[eps] = {}
    log_data_no_bt[eps] = {}
    
    steps, lr = args.steps, args.lr
    log_data['steps'][eps] = steps
    log_data['lrs'][eps] = lr
    split_size = math.ceil(len(data) / steps)
    splits = [data[j * split_size:(j+1) * split_size] for j in range(steps)]
    alpha = args.alpha
    eta = lr

    log_data[eps] = {}
    log_data_no_bt[eps] = {}
    rng = np.random.default_rng(seed=args.seed)
    np.random.seed(args.seed)

    for scale in scales:
        log_data[eps][scale] = {'mse': 0, 'split_mse': [], 'domain_distance': 0, 'split_domain_distance': []}
        log_data_no_bt[eps][scale] = {'mse': 0}
        l = data_center - (data_center - data_min) * scale
        r = data_center + (data_max - data_center) * scale

        split_means = np.zeros((args.num, steps), dtype=np.float64)
        split_est_means = np.zeros((args.num, steps), dtype=np.float64)
        split_domain_err = np.zeros((args.num, steps), dtype=np.float64)
        split_est_domains = np.zeros((args.num, steps, 2), dtype=np.float64)

        marginals_left = np.zeros((args.num, steps), dtype=np.float64)
        marginals_right = np.zeros((args.num, steps), dtype=np.float64)

        original_domain_distance = abs(l-data_min) + abs(r-data_max)
        log_data[eps][scale]['original_margianl_left'] = np.sum(data < l) / num_data
        log_data[eps][scale]['original_margianl_right'] = np.sum(data > r) / num_data

        for i in range(args.num):
            mech, de, perturb_func = get_mech(mechanism, 1, eps, rng, args.eps_ratio, args)
            bt = ldp.get_bt(eps, args.eps_ratio, l, r, alpha, eta, args.zeta, args.tau)

            for s, split in enumerate(splits):
                true_mean, est_mean, true_hist, est_hist = ldp.perturb(split, de, perturb_func, bt)
                split_means[i, s] = true_mean
                split_est_means[i, s] = est_mean
                split_est_domains[i, s, 0] = bt.l
                split_est_domains[i, s, 1] = bt.r
                bt.fit(est_hist[0], est_hist[1], est_hist[2])
                split_domain_err[i, s] = abs(bt.l - data_min) + abs(bt.r - data_max)
      
                marginals_left[i, s] = np.sum(data < bt.l) / num_data
                marginals_right[i, s] = np.sum(data > bt.r) / num_data

        est_means = np.mean(split_est_means, axis=1)
        log_data[eps][scale]['mse'] = np.mean( (true_mean - est_means)**2 )

        log_data[eps][scale]['split_mse'] = np.mean((split_means - split_est_means)**2, axis=0)
        log_data[eps][scale]['split_domain_distance'] = np.mean(split_domain_err, axis=0)
        log_data[eps][scale]['domain_distance'] = np.mean(split_domain_err)
        log_data[eps][scale]['original_domain_distance'] = original_domain_distance
        log_data[eps][scale]['marginals_left'] = np.mean(marginals_left, axis=0)
        log_data[eps][scale]['marginals_right'] = np.mean(marginals_right, axis=0)

        est_means = np.zeros(args.num)
        for i in range(args.num):
            mech, de, perturb_func = get_mech(mechanism, 1, eps, rng, 1.0, args)
            bt = ldp.get_bt(eps, 1.0, l, r, alpha, eta, args.zeta, args.tau)
            true_mean, est_mean = ldp.perturb_no_de(split, perturb_func, bt)
            est_means[i] = est_mean

        log_data_no_bt[eps][scale]['mse'] = np.mean( (mean - est_means)**2 )

pickle_filename = os.path.join(log_dir, f'log.pkl') 
with open(pickle_filename, 'wb') as pkl_file:
    pickle.dump(log_data, pkl_file)

pickle_filename = os.path.join(base_log_dir, f'log.pkl') 
if not(os.path.exists(pickle_filename)):
    with open(pickle_filename, 'wb') as pkl_file:
        pickle.dump(log_data_no_bt, pkl_file)