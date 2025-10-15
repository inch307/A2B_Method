import numpy as np
from scipy.optimize import minimize, Bounds
import time
import argparse
import pickle
import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def unpack_vars_jax(v, n, use_a0, objective_type):
    v_main = v[:-1] if objective_type == 'worst' else v

    a_pos = v_main[:n] 
    
    if use_a0:
        num_outputs = 2 * n + 1
        a_full = jnp.concatenate([jnp.flip(-a_pos), jnp.array([0.0]), a_pos])
    else:
        num_outputs = 2 * n
        a_full = jnp.concatenate([jnp.flip(-a_pos), a_pos])

    p_v = v_main[n:].reshape((n + 1, num_outputs))
    p_T = p_v.T
    return a_full, p_T

def objective_avg_jax(v, n, use_a0):
    a, p_T = unpack_vars_jax(v, n, use_a0, 'average')
    a_sq, x = a**2, a @ p_T
    x_j, x_jm1 = x[1:], x[:-1]
    dx = x_j - x_jm1
    E_Y2 = a_sq @ p_T
    E_Y2_j, E_Y2_jm1 = E_Y2[1:], E_Y2[:-1]
    avg_variance_proxy = (E_Y2_j + E_Y2_jm1) / 2.0
    integral_of_P = avg_variance_proxy * dx
    integral_of_x_sq = (x_j**3 - x_jm1**3) / 3.0
    total = jnp.where(jnp.abs(dx) > 1e-20, integral_of_P - integral_of_x_sq, 0.0).sum()
    return total

def objective_worst_jax(v):
    return v[-1]

def get_candidate_variances_jax(v_main, n, use_a0):
    a, p_T = unpack_vars_jax(v_main, n, use_a0, 'average') 
    a_sq, x = a**2, a @ p_T
    E_Y2 = a_sq @ p_T
    variances_at_xj = E_Y2 - x**2
    x_j, x_jm1 = x[1:], x[:-1]
    dx = x_j - x_jm1
    E_Y2_j, E_Y2_jm1 = E_Y2[1:], E_Y2[:-1]
    x_star_num, x_star_den = E_Y2_j - E_Y2_jm1, 2 * dx
    x_star = jnp.where(jnp.abs(x_star_den) > 1e-12, x_star_num / x_star_den, jnp.inf)
    is_in_interval = (x_star >= x_jm1) & (x_star <= x_j)
    slope_E_Y2 = jnp.where(jnp.abs(dx) > 1e-12, x_star_num / dx, 0)
    E_Y2_at_x_star = slope_E_Y2 * (x_star - x_j) + E_Y2_j
    var_at_x_star = E_Y2_at_x_star - x_star**2
    valid_var_at_x_star = jnp.where(is_in_interval, var_at_x_star, -jnp.inf)
    return jnp.concatenate([variances_at_xj, valid_var_at_x_star])

# ✨ [수정] a_i < a_{i+1} 제약조건 추가
def constraints_jax(v, n, epsilon, use_a0, objective_type):
    MIN_GAP = 1e-7
    
    v_main = v[:-1] if objective_type == 'worst' else v
    z = v[-1] if objective_type == 'worst' else 0
    a_full, p_T = unpack_vars_jax(v_main, n, use_a0, 'average')
    p, x = p_T.T, a_full @ p_T
    exp_eps = jnp.exp(epsilon)
    
    eq_cons_list = [jnp.sum(p, axis=1) - 1, jnp.atleast_1d(x[n] - 1)]
    p_j0 = p[0, :]
    if use_a0: eq_cons_list.append(p_j0[n+1:] - jnp.flip(p_j0[:n]))
    else: eq_cons_list.append(p_j0[n:] - jnp.flip(p_j0[:n]))

    a_pos = v_main[:n]
    
    ineq_cons_list = [
        jnp.atleast_1d(a_pos[0] - MIN_GAP),     
        jnp.diff(a_pos) - MIN_GAP,           
        jnp.diff(x) - MIN_GAP              
    ]
    
    p_n = p[n, :]
    if use_a0:
        p_neg_i_n_ref, p_zero_n_ref = jnp.flip(p_n[:n]), p_n[n]
        for j in range(n + 1):
            p_j = p[j, :]; ineq_cons_list.extend([p_j[n+1:] - p_neg_i_n_ref, exp_eps * p_neg_i_n_ref - p_j[n+1:], jnp.atleast_1d(p_j[n] - p_zero_n_ref), jnp.atleast_1d(exp_eps * p_zero_n_ref - p_j[n])])
        for k in range(1, n + 1):
            p_k = p[k, :]; p_neg_i_k, p_zero_k = jnp.flip(p_k[:n]), p_k[n]
            ineq_cons_list.extend([p_neg_i_k - p_neg_i_n_ref, exp_eps * p_neg_i_n_ref - p_neg_i_k, jnp.atleast_1d(p_zero_k - p_zero_n_ref), jnp.atleast_1d(exp_eps * p_zero_n_ref - p_zero_k)])
    else:
        p_neg_i_n_ref = jnp.flip(p_n[:n])
        for j in range(n + 1):
            p_j = p[j, :]; ineq_cons_list.extend([p_j[n:] - p_neg_i_n_ref, exp_eps * p_neg_i_n_ref - p_j[n:]])
        for k in range(1, n + 1):
            p_k = p[k, :]; p_neg_i_k = jnp.flip(p_k[:n])
            ineq_cons_list.extend([p_neg_i_k - p_neg_i_n_ref, exp_eps * p_neg_i_n_ref - p_neg_i_k])

    if objective_type == 'worst':
        candidate_vars = get_candidate_variances_jax(v_main, n, use_a0)
        ineq_cons_list.append(z - candidate_vars)

    return jnp.concatenate(eq_cons_list), jnp.concatenate(ineq_cons_list)

jax_fns_cache = {}
def create_jax_functions(n, epsilon, use_a0, objective_type):
    cache_key = (n, epsilon, use_a0, objective_type)
    if cache_key in jax_fns_cache: return jax_fns_cache[cache_key]
    
    print(f"n={n}, obj={objective_type}...")
    if objective_type == 'average':
        obj_fn_pure = lambda v: objective_avg_jax(v, n, use_a0)
    else:
        obj_fn_pure = objective_worst_jax
    eq_cons_fn = lambda v: constraints_jax(v, n, epsilon, use_a0, objective_type)[0]
    ineq_cons_fn = lambda v: constraints_jax(v, n, epsilon, use_a0, objective_type)[1]
    
    obj_jac = jax.jit(jax.grad(obj_fn_pure))
    eq_cons_jac = jax.jit(jax.jacobian(eq_cons_fn))
    ineq_cons_jac = jax.jit(jax.jacobian(ineq_cons_fn))

    jax_fns_cache[cache_key] = {
        'objective': jax.jit(obj_fn_pure), 'objective_jac': obj_jac,
        'eq_cons': jax.jit(eq_cons_fn), 'eq_cons_jac': eq_cons_jac,
        'ineq_cons': jax.jit(ineq_cons_fn), 'ineq_cons_jac': ineq_cons_jac,
    }
    return jax_fns_cache[cache_key]


class SimpleCallbackLogger:
    def __init__(self, optimizer_instance, disp_interval=10):
        self.iteration = 0; self.optimizer = optimizer_instance
        self.last_fun_val = np.inf; self.disp_interval = disp_interval
        print("\n--- Optimization Log ---")
        print("{:>5} | {:>18} | {:>15}".format("Iter", "Objective Value", "Change"))
        print("-" * 45)
    def __call__(self, xk, *args, **kwargs):
        if self.iteration == 0 or (self.iteration + 1) % self.disp_interval == 0:
            fun_val = self.optimizer.objective_func(xk)
            change = self.last_fun_val - fun_val
            print("{:5d} | {:18.6f} | {:15.6e}".format(self.iteration, fun_val, change))
            self.last_fun_val = fun_val
        self.iteration += 1

class LDPMechanismOptimizer:
    def __init__(self, n, epsilon, use_a0, objective_type):
        self.n, self.epsilon, self.use_a0, self.objective_type = n, epsilon, use_a0, objective_type
        self.num_a_vars = n
        num_outputs = (2 * n + 1) if use_a0 else (2 * n)
        self.num_p_vars = (n + 1) * num_outputs
        self.fns = create_jax_functions(n, epsilon, use_a0, objective_type)

    def objective_func(self, v):
        return self.fns['objective'](v).item()

    def objective_jac(self, v):
        return np.array(self.fns['objective_jac'](v))

    def _create_constraints(self):
        return [
            {'type': 'eq', 'fun': self.fns['eq_cons'], 'jac': self.fns['eq_cons_jac']},
            {'type': 'ineq', 'fun': self.fns['ineq_cons'], 'jac': self.fns['ineq_cons_jac']}
        ]

    def unpack_vars(self, v):
        expected_len_with_z = self.num_a_vars + self.num_p_vars + 1
        if self.objective_type == 'worst' and len(v) == expected_len_with_z:
            v_main = v[:-1]
        else:
            v_main = v
        
        a_pos = v_main[:self.n]
        
        if self.use_a0:
            num_outputs = 2 * self.n + 1
            a_full = np.concatenate([-a_pos[::-1], [0], a_pos])
        else:
            num_outputs = 2 * self.n
            a_full = np.concatenate([-a_pos[::-1], a_pos])

        p_v = v_main[self.n:].reshape((self.n + 1, num_outputs))
        return a_full, p_v.T
    
    def run(self, initial_guess, options):
        if self.objective_type == 'worst':
            initial_worst_var = calculate_worst_case_variance(initial_guess, self)
            initial_guess = np.append(initial_guess, initial_worst_var)
            lv = [0.0] * self.num_a_vars + [0.0] * self.num_p_vars + [0.0]
            uv = [np.inf] * self.num_a_vars + [1.0] * self.num_p_vars + [np.inf]
        else:
            lv = [0.0] * self.num_a_vars + [0.0] * self.num_p_vars
            uv = [np.inf] * self.num_a_vars + [1.0] * self.num_p_vars
        
        b = Bounds(lv, uv)
        c = self._create_constraints()
        cb = SimpleCallbackLogger(self)
        
        return minimize(self.objective_func, initial_guess, method='SLSQP', 
                        jac=self.objective_jac,
                        bounds=b, constraints=c, options=options, callback=cb)

def create_initial_guess_fixed_gap(n, epsilon, use_a0, smoothing_factor=0.1):
    exp_eps = np.exp(epsilon)
    initial_a_1_base = (exp_eps + 1) / ((exp_eps - 1) * n) if n > 0 else (exp_eps + 1) / (exp_eps - 1)

    initial_a_pos = initial_a_1_base * np.arange(1, n + 1)

    A = n * initial_a_1_base
    t = np.exp(epsilon/3); p_core, p_tail = exp_eps/(exp_eps+t), t/(exp_eps+t)
    get_overlap = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0]))
    num_output = 2*n+1 if use_a0 else 2*n
    p_matrix = np.zeros((n + 1, num_output))
    i_range = range(-n,n+1) if use_a0 else list(range(-n,0))+list(range(1,n+1))
    for j in range(n + 1):
        x = j / n; l_x, r_x = (A+1)/2.*x-(A-1)/2., (A+1)/2.*x+(A-1)/2.
        w_c, w_t = r_x - l_x, 2*A - (r_x - l_x)
        pdf_c = p_core/w_c if w_c > 1e-9 else 0; pdf_t = p_tail/w_t if w_t > 1e-9 else 0
        for idx, i in enumerate(i_range):
            if i == -n: s, e = -A, (-n + 0.5)/n*A
            elif i == n: s, e = (n - 0.5)/n*A, A
            elif i == 0 and use_a0: s, e = -0.5/n*A, 0.5/n*A
            else: s, e = (i-0.5)/n*A, (i+0.5)/n*A
            p_matrix[j, idx] = get_overlap([s,e],[-A,l_x])*pdf_t+get_overlap([s,e],[l_x,r_x])*pdf_c+get_overlap([s,e],[r_x,A])*pdf_t
        row_sum = np.sum(p_matrix[j, :]);
        if row_sum > 1e-9: p_matrix[j, :] /= row_sum

    a_full_initial = np.concatenate([-initial_a_pos[::-1], [0], initial_a_pos]) if use_a0 else np.concatenate([-initial_a_pos[::-1], initial_a_pos])
    x_n_initial = np.dot(p_matrix[n, :], a_full_initial)
    if abs(x_n_initial) > 1e-9:
        scaling_factor = 1.0/x_n_initial
        initial_a_pos *= scaling_factor 
        
    uniform = np.full_like(p_matrix, 1./num_output); smoothed = (1-smoothing_factor)*p_matrix + smoothing_factor*uniform
    for j in range(n+1):
        row_sum = np.sum(smoothed[j,:]);
        if row_sum > 1e-9: smoothed[j,:] /= row_sum

    return np.concatenate([initial_a_pos, smoothed.flatten()])

def calculate_worst_case_variance(v, optimizer):
    v_main = v
    if optimizer.objective_type == 'worst' and len(v) == optimizer.num_a_vars + optimizer.num_p_vars + 1:
        v_main = v[:-1]
    n, a, p = optimizer.n, *optimizer.unpack_vars(v_main)
    a_sq = a**2
    x, E_Y2 = a @ p, a_sq @ p
    variances_at_xj = E_Y2 - x**2
    candidate_variances = list(variances_at_xj)
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j-1]
        dx = xj - xjm1
        if abs(dx) < 1e-12: continue
        E_Y2_j, E_Y2_jm1 = E_Y2[j], E_Y2[j-1]
        x_star_num, x_star_den = E_Y2_j - E_Y2_jm1, 2*dx
        if abs(x_star_den) < 1e-12: continue
        x_star = x_star_num / x_star_den
        if (x_star >= xjm1 - 1e-9) and (x_star <= xj + 1e-9):
            slope_E_Y2 = x_star_num / dx
            E_Y2_at_x_star = slope_E_Y2 * (x_star - xj) + E_Y2_j
            var_at_x_star = E_Y2_at_x_star - x_star**2
            candidate_variances.append(var_at_x_star)
    return np.max(candidate_variances)

def calculate_average_case_variance(v, optimizer):
    v_main = v
    if optimizer.objective_type == 'worst' and len(v) == optimizer.num_a_vars + optimizer.num_p_vars + 1:
        v_main = v[:-1]
    n, a, p = optimizer.n, *optimizer.unpack_vars(v_main)
    x = a @ p
    total = 0.0
    a_sq = a**2
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j - 1]
        dx = xj - xjm1
        if abs(dx) < 1e-20: continue
        pj, pjm1 = p[:, j], p[:, j - 1]
        avg_variance_proxy = (np.dot(a_sq, pj) + np.dot(a_sq, pjm1)) / 2.0
        integral_of_P = avg_variance_proxy * dx
        integral_of_x_sq = (xj**3 - xjm1**3) / 3.0
        total += (integral_of_P - integral_of_x_sq)
    return total

def display_and_save_results(result, optimizer, N, output_dir):
    v = result.x
    v_main = v[:-1] if optimizer.objective_type == 'worst' else v
    
    final_a, final_p = optimizer.unpack_vars(v_main)

    print(f"a = {np.array2string(final_a, precision=6, max_line_width=120)}")

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"opt_results_eps{optimizer.epsilon}_N{N}.pkl")
    

    results_data = {
        'metadata': {'N_total_a_points': N, 'n_param': optimizer.n, 'epsilon': optimizer.epsilon,
                     'use_a0': optimizer.use_a0, 'optimization_type': optimizer.objective_type, 
                     'success': result.success, 'message': result.message, 'avg_var': result.fun, 'worst_var':calculate_worst_case_variance(v, optimizer),
                     'iterations': result.nit},
        'a_values': final_a, 'p_matrix': final_p, 'scipy_result_object': result }
    
    with open(filename, 'wb') as f: pickle.dump(results_data, f)
    print(f"  Success: {result.success}\n  Message: {result.message}")
    
    if optimizer.objective_type == 'worst':
        print(f"  Final Objective Value (Worst-Case Var): {result.fun:.6f}")
        print(f"  Corresponding Average-Case Variance: {calculate_average_case_variance(v, optimizer):.6f}")
    else: # average
        print(f"  Final Objective Value (Avg Var): {result.fun:.6f}")
        print(f"  Corresponding Worst-Case Variance: {calculate_worst_case_variance(v, optimizer):.6f}")

def optimize(epsilon, N, ftol=1e-6, maxiter=3000, disp=False):
    my_options = {'ftol': ftol, 'maxiter': maxiter, 'disp': disp}
    
    if N <= 1: raise ValueError("N must be > 1.")
    
    use_a0 = (N % 2 != 0)
    n_param = (N - 1) // 2 if use_a0 else N // 2

    optimizer = LDPMechanismOptimizer(n_param, epsilon, use_a0, 'average')
    initial_guess = create_initial_guess_fixed_gap(n_param, epsilon, use_a0)

    result = optimizer.run(initial_guess, options=my_options)

    output_dir = f'results_average_free_a'
    
    if result.success:
        display_and_save_results(result, optimizer, N, output_dir)
    else:
        display_and_save_results(result, optimizer, N, output_dir)