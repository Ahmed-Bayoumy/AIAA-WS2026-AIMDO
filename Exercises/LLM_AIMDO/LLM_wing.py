import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import random
import pandas as pd
import scipy.special
from scipy.stats import norm
import time # For simulating simulation time

# --- Global Constants and Problem Definition ---
N_DESIGN_VARS = 3
DESIGN_VAR_NAMES = ["Aspect Ratio (AR)", "Taper Ratio (TR)", "Thickness/Chord (tc)"]
X_BOUNDS = np.array([[5.0, 15.0], [0.2, 0.8], [0.08, 0.15]]) # Bounds for AR, TR, tc

# Target and limits for constraints
CL_TARGET = 1.2
SIGMA_LIMIT = 800.0 # Relaxed stress limit
CL_TOLERANCE = 0.1 # Relaxed lift tolerance

# Simulated Noise for CFD/FEM
SIM_NOISE_STD_DRAG = 0.005
SIM_NOISE_STD_LIFT = 0.02
SIM_NOISE_STD_STRESS = 10.0

# MDO Loop Parameters
NUM_INITIAL_SAMPLES = 10
NUM_ITERATIONS = 20 # Number of infill points to find
NUM_CANDIDATE_POINTS_PER_ITER = 100 # For optimization agent to search for infill

# Parameters for Baselines
# Total simulations for MDO = NUM_INITIAL_SAMPLES + NUM_ITERATIONS
NUM_TOTAL_MDO_SIMS = NUM_INITIAL_SAMPLES + NUM_ITERATIONS

# Ensemble Runs
N_ENSEMBLE_RUNS = 20 # Number of times to repeat the entire MDO process

# GA Baseline: Generate enough samples for the GA to "explore"
NUM_GA_BASELINE_SAMPLES = NUM_TOTAL_MDO_SIMS + 50 

# OMADS.mads Baseline: It will run as a separate agentic MDO
# It will have its own NUM_INITIAL_SAMPLES and NUM_ITERATIONS, same as main MDO

# --- 0. Simplified Wing Design Analytical Models (Simulated CFD/FEM) ---

def f_drag(x): # x = [AR, TR, tc]
    AR, TR, tc = x
    return 0.02 + (0.01 * (1/AR)) + (0.05 * tc**2) + (0.01 * (1-TR))

def f_lift(x): # x = [AR, TR, tc]
    AR, TR, tc = x
    return 0.5 + (0.05 * AR) + (0.1 * TR) - (0.2 * tc)

def f_stress(x): # x = [AR, TR, tc]
    AR, TR, tc = x
    tc = max(tc, 0.01) # Ensure tc is not too small
    return 100 * (tc**-1) + (5 * AR) + (20 * (1-TR))

# --- Shared State / Blackboard for Agents ---
class MDOState:
    def __init__(self):
        self.design_history = pd.DataFrame(columns=DESIGN_VAR_NAMES + ['Cd_sim', 'Cl_sim', 'Sigma_sim', 'Feasible'])
        self.gp_models = {
            'Cd': None,
            'Cl': None,
            'Sigma': None
        }
        self.best_feasible_design = None
        self.iteration = 0
        self.current_goal = ""
        self.current_status = "Initialized"
        
        # Plotting data for the *last* run's scatter plot
        self.all_evaluated_points_for_plot = {'X': [], 'Cd': [], 'Feasible': []} 
        
        # Store pre-evaluated designs for GA baseline (generated per run)
        self.pre_evaluated_ga_baseline_designs = pd.DataFrame(columns=DESIGN_VAR_NAMES + ['Cd_sim', 'Cl_sim', 'Sigma_sim', 'Feasible'])
        
        # Internal history for a single run, returned at the end
        self._current_run_mdo_cd_history = []
        self._current_run_ga_cd_history = []
        self._current_run_omads_mads_cd_history = [] 

    def add_design_result(self, x, cd_sim, cl_sim, sigma_sim, feasible):
        new_row = pd.DataFrame([list(x) + [cd_sim, cl_sim, sigma_sim, feasible]], 
                               columns=self.design_history.columns)
        self.design_history = pd.concat([self.design_history, new_row], ignore_index=True)
        
        # Update plotting data for the *last* run's scatter plot
        self.all_evaluated_points_for_plot['X'].append(x)
        self.all_evaluated_points_for_plot['Cd'].append(cd_sim)
        self.all_evaluated_points_for_plot['Feasible'].append(feasible)
        
        # Update internal history for this run
        self._update_current_run_mdo_cd_history()

    def get_evaluated_designs(self):
        return self.design_history[DESIGN_VAR_NAMES].values, \
               self.design_history[['Cd_sim', 'Cl_sim', 'Sigma_sim']].values

    def _update_current_run_mdo_cd_history(self):
        feasible_designs = self.design_history[self.design_history['Feasible'] == True]
        if not feasible_designs.empty:
            current_best_cd = feasible_designs['Cd_sim'].min()
            if not self._current_run_mdo_cd_history or current_best_cd < self._current_run_mdo_cd_history[-1]:
                self._current_run_mdo_cd_history.append(current_best_cd)
            else:
                self._current_run_mdo_cd_history.append(self._current_run_mdo_cd_history[-1])
        else:
            self._current_run_mdo_cd_history.append(self._current_run_mdo_cd_history[-1] if self._current_run_mdo_cd_history else np.inf)

    # Update GA baseline for current simulation count
    def update_ga_baseline_for_current_sim(self, num_mdo_simulations):
        current_sim_count = num_mdo_simulations
        
        current_ga_samples = self.pre_evaluated_ga_baseline_designs.head(current_sim_count)
        
        feasible_ga_designs = current_ga_samples[current_ga_samples['Feasible'] == True]
        
        if not feasible_ga_designs.empty:
            current_best_ga_cd = feasible_ga_designs['Cd_sim'].min()
            if not self._current_run_ga_cd_history or current_best_ga_cd < self._current_run_ga_cd_history[-1]:
                self._current_run_ga_cd_history.append(current_best_ga_cd)
            else:
                self._current_run_ga_cd_history.append(self._current_run_ga_cd_history[-1])
        else:
            self._current_run_ga_cd_history.append(self._current_run_ga_cd_history[-1] if self._current_run_ga_cd_history else np.inf)

    # Update OMADS.mads baseline for current simulation count
    def update_omads_mads_baseline_for_current_sim(self):
        # This method is called after OMADS.mads has performed its own simulation
        feasible_mads_designs = self.design_history[self.design_history['Feasible'] == True]
        
        if not feasible_mads_designs.empty:
            current_best_mads_cd = feasible_mads_designs['Cd_sim'].min()
            if not self._current_run_omads_mads_cd_history or current_best_mads_cd < self._current_run_omads_mads_cd_history[-1]:
                self._current_run_omads_mads_cd_history.append(current_best_mads_cd)
            else:
                self._current_run_omads_mads_cd_history.append(self._current_run_omads_mads_cd_history[-1])
        else:
            self._current_run_omads_mads_cd_history.append(self._current_run_omads_mads_cd_history[-1] if self._current_run_omads_mads_cd_history else np.inf)

# --- 1. Planner Agent ---
class PlannerAgent:
    def __init__(self, state: MDOState):
        self.state = state
        self.llm_interface = LLMInterface()

    def plan_initial_phase(self):
        self.state.current_goal = "Generate initial samples for surrogate model training."
        self.state.current_status = "Initial Sampling"
        return "INITIAL_SAMPLING"

    def plan_optimization_loop(self):
        self.state.current_goal = f"Iteration {self.state.iteration + 1}: Find next infill point to minimize drag."
        self.state.current_status = "Optimization Loop"
        return "OPTIMIZATION_LOOP"

    def plan_review_phase(self):
        self.state.current_goal = "Review final results and present best designs."
        self.state.current_status = "Review Results"
        return "REVIEW_RESULTS"

# --- 2. Simulation Agent (Simulated CFD/FEM) ---
class SimulationAgent:
    def __init__(self, state: MDOState):
        self.state = state

    def run_simulation(self, x):
        cd_true = f_drag(x)
        cl_true = f_lift(x)
        sigma_true = f_stress(x)

        # Add simulated noise
        cd_sim = cd_true + np.random.normal(0, SIM_NOISE_STD_DRAG)
        cl_sim = cl_true + np.random.normal(0, SIM_NOISE_STD_LIFT)
        sigma_sim = sigma_true + np.random.normal(0, SIM_NOISE_STD_STRESS)
        
        cd_sim = max(0.01, cd_sim) 
        cl_sim = max(0.1, cl_sim)
        sigma_sim = max(1.0, sigma_sim)

        return cd_sim, cl_sim, sigma_sim

# --- 3. Surrogate Agent (Gaussian Process) ---
class SurrogateAgent:
    def __init__(self, state: MDOState):
        self.state = state
        kernel_cd = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        kernel_cl = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        kernel_sigma = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        self.gp_regressors = {
            'Cd': GaussianProcessRegressor(kernel=kernel_cd, n_restarts_optimizer=10, random_state=None), # Set random_state to None for stochasticity across runs
            'Cl': GaussianProcessRegressor(kernel=kernel_cl, n_restarts_optimizer=10, random_state=None),
            'Sigma': GaussianProcessRegressor(kernel=kernel_sigma, n_restarts_optimizer=10, random_state=None)
        }

    def update_models(self):
        X_data, Y_data = self.state.get_evaluated_designs()
        
        if X_data.shape[0] < 2:
            return

        self.gp_regressors['Cd'].fit(X_data, Y_data[:, 0])
        self.gp_regressors['Cl'].fit(X_data, Y_data[:, 1])
        self.gp_regressors['Sigma'].fit(X_data, Y_data[:, 2])
        self.state.gp_models = self.gp_regressors

    def predict(self, x_cand):
        if not self.state.gp_models['Cd']:
            return None, None, None, None, None, None

        cd_mean, cd_std = self.state.gp_models['Cd'].predict(x_cand.reshape(1, -1), return_std=True)
        cl_mean, cl_std = self.state.gp_models['Cl'].predict(x_cand.reshape(1, -1), return_std=True)
        sigma_mean, sigma_std = self.state.gp_models['Sigma'].predict(x_cand.reshape(1, -1), return_std=True)
        
        return cd_mean[0], cd_std[0], cl_mean[0], cl_std[0], sigma_mean[0], sigma_std[0]

# --- 4. Optimization Agent (Expected Improvement with Constraints) ---
class OptimizationAgent:
    def __init__(self, state: MDOState):
        self.state = state
        self.surrogate_agent = SurrogateAgent(state)

    def _expected_improvement(self, x_cand, best_cd_so_far):
        cd_mean, cd_std, _, _, _, _ = self.surrogate_agent.predict(x_cand)

        if cd_std == 0:
            return 0.0

        Z = (best_cd_so_far - cd_mean) / cd_std
        ei = cd_std * (norm.cdf(Z) * Z + norm.pdf(Z))
        return ei

    def _probability_of_feasibility(self, x_cand):
        _, _, cl_mean, cl_std, sigma_mean, sigma_std = self.surrogate_agent.predict(x_cand)
        
        if cl_std == 0 or sigma_std == 0:
            cl_feasible = abs(cl_mean - CL_TARGET) <= CL_TOLERANCE
            sigma_feasible = sigma_mean <= SIGMA_LIMIT
            return float(cl_feasible and sigma_feasible)

        cl_prob_upper = norm.cdf((CL_TARGET + CL_TOLERANCE - cl_mean) / cl_std)
        cl_prob_lower = norm.cdf((CL_TARGET - CL_TOLERANCE - cl_mean) / cl_std)
        prob_cl_feasible = cl_prob_upper - cl_prob_lower

        prob_sigma_feasible = norm.cdf((SIGMA_LIMIT - sigma_mean) / sigma_std)

        return prob_cl_feasible * prob_sigma_feasible

    def _acquisition_function(self, x_cand, best_cd_so_far):
        ei = self._expected_improvement(x_cand, best_cd_so_far)
        pof = self._probability_of_feasibility(x_cand)
        return ei * pof

    def select_infill_point(self):
        X_evaluated, Y_evaluated = self.state.get_evaluated_designs()
        
        feasible_designs = self.state.design_history[self.state.design_history['Feasible'] == True]
        if not feasible_designs.empty:
            best_cd_so_far = feasible_designs['Cd_sim'].min()
        else:
            best_cd_so_far = Y_evaluated[:, 0].max() * 1.5 if Y_evaluated.shape[0] > 0 else 1.0 

        candidate_points = np.random.uniform(X_BOUNDS[:, 0], X_BOUNDS[:, 1], 
                                             size=(NUM_CANDIDATE_POINTS_PER_ITER, N_DESIGN_VARS))
        
        best_infill_x = None
        max_acquisition = -np.inf

        for x_cand in candidate_points:
            acq_val = self._acquisition_function(x_cand, best_cd_so_far) 
            if acq_val > max_acquisition:
                max_acquisition = acq_val
                best_infill_x = x_cand
        
        if best_infill_x is not None:
            obj_func = lambda x: -self._acquisition_function(x, best_cd_so_far)
            res = minimize(obj_func, best_infill_x, bounds=X_BOUNDS, method='L-BFGS-B')
            if res.success:
                best_infill_x = res.x
            else:
                pass 

        return best_infill_x

# --- 5. Safety Agent ---
class SafetyAgent:
    def __init__(self, state: MDOState):
        self.state = state

    def check_feasibility(self, cd_sim, cl_sim, sigma_sim):
        cl_feasible = abs(cl_sim - CL_TARGET) <= CL_TOLERANCE
        sigma_feasible = sigma_sim <= SIGMA_LIMIT
        return cl_feasible and sigma_feasible

    def filter_proposals(self, x_proposal):
        return True 

# --- 6. Reviewer Agent ---
class ReviewerAgent:
    def __init__(self, state: MDOState):
        self.state = state
        self.llm_interface = LLMInterface()

    def rank_and_present_options(self):
        feasible_designs = self.state.design_history[self.state.design_history['Feasible'] == True]

        if feasible_designs.empty:
            return

        ranked_designs = feasible_designs.sort_values(by='Cd_sim', ascending=True)
        self.state.best_feasible_design = ranked_designs.iloc[0]

# --- LLM Interface (Simulated) ---
class LLMInterface:
    def generate_response(self, prompt):
        return f"LLM: {prompt}"

# --- Plotting Functions ---
def plot_ensemble_results(all_mdo_histories, all_ga_histories, all_omads_mads_histories, last_run_state: MDOState):
    plt.ioff() # Ensure non-interactive mode for final plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Agentic MDO Progress: Wing Design Optimization (Ensemble Results)', fontsize=16, fontweight='bold')

    # --- Plot 1: Optimization Progress (Mean, Median, Variance) ---
    ax1.set_title(f'Optimization Progress: Best Feasible Drag (N={N_ENSEMBLE_RUNS} Runs)', fontsize=14)
    ax1.set_xlabel('Total Simulations', fontsize=12)
    ax1.set_ylabel('Best Feasible Drag Coefficient (Cd)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    x_values = np.arange(1, NUM_TOTAL_MDO_SIMS + 1)

    def plot_stats(ax, histories, color, label):
        histories_array = np.array(histories)
        
        # Replace np.inf with NaN for statistical calculations, then filter NaNs
        histories_array[histories_array == np.inf] = np.nan
        
        # Calculate mean, median, std, ignoring NaNs
        mean_vals = np.nanmean(histories_array, axis=0)
        median_vals = np.nanmedian(histories_array, axis=0)
        std_vals = np.nanstd(histories_array, axis=0)
        
        # Handle cases where all values are NaN (e.g., no feasible designs yet)
        mean_vals[np.isnan(mean_vals)] = np.inf
        median_vals[np.isnan(median_vals)] = np.inf
        std_vals[np.isnan(std_vals)] = 0 # Std dev of all NaNs is 0
        
        # Plot mean
        ax.plot(x_values, mean_vals, color=color, linestyle='-', linewidth=2, label=f'{label} (Mean)')
        
        # Plot median
        ax.plot(x_values, median_vals, color=color, linestyle='--', linewidth=1.5, label=f'{label} (Median)')
        
        # Fill area for variance (mean ± std)
        lower_bound = mean_vals - std_vals
        upper_bound = mean_vals + std_vals
        
        # Ensure bounds are not negative and handle inf
        lower_bound[lower_bound < 0] = 0 # Drag cannot be negative
        lower_bound[mean_vals == np.inf] = np.inf # If mean is inf, bounds are inf
        upper_bound[mean_vals == np.inf] = np.inf
        
        ax.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.15, label=f'{label} (Mean ± Std Dev)')

    plot_stats(ax1, all_mdo_histories, 'blue', 'Agentic MDO')
    plot_stats(ax1, all_ga_histories, 'purple', 'GA Baseline')
    plot_stats(ax1, all_omads_mads_histories, 'darkgreen', 'OMADS.mads Baseline') 

    ax1.legend(fontsize=10, loc='upper right')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xlim(0, NUM_TOTAL_MDO_SIMS + 1)
    
    # Adjust y-limits based on all valid (non-inf) data
    all_finite_cds = []
    for hist in all_mdo_histories + all_ga_histories + all_omads_mads_histories:
        all_finite_cds.extend([cd for cd in hist if cd != np.inf])

    if all_finite_cds:
        min_cd = min(all_finite_cds) * 0.9
        max_cd = max(all_finite_cds) * 1.1
        ax1.set_ylim(min_cd, max_cd)
    else:
        ax1.set_ylim(0.0, 0.5)

    # --- Plot 2: Design Space Exploration (from last run) ---
    ax2.set_title(f'Design Space Exploration ({DESIGN_VAR_NAMES[0]} vs {DESIGN_VAR_NAMES[1]}) - Last Run', fontsize=14)
    ax2.set_xlabel(DESIGN_VAR_NAMES[0], fontsize=12)
    ax2.set_ylabel(DESIGN_VAR_NAMES[1], fontsize=12)
    ax2.set_xlim(X_BOUNDS[0,0], X_BOUNDS[0,1])
    ax2.set_ylim(X_BOUNDS[1,0], X_BOUNDS[1,1])
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Create empty scatter plots with specific markers for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Feasible Designs',
               markerfacecolor='green', markersize=10, markeredgecolor='black'), 
        Line2D([0], [0], marker='x', color='w', label='Infeasible Designs',
               markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', label='Current Best Feasible',
               markerfacecolor='red', markersize=15, markeredgecolor='black')
    ]
    ax2.legend(handles=legend_elements, fontsize=10, loc='upper left', frameon=True, shadow=True)

    X_data = np.array(last_run_state.all_evaluated_points_for_plot['X'])
    Cd_data = np.array(last_run_state.all_evaluated_points_for_plot['Cd'])
    Feasible_data = np.array(last_run_state.all_evaluated_points_for_plot['Feasible'])

    if X_data.shape[0] > 0:
        vmin = np.min(Cd_data)
        vmax = np.max(Cd_data)
        
        feasible_X = X_data[Feasible_data]
        feasible_Cd = Cd_data[Feasible_data]
        infeasible_X = X_data[~Feasible_data]
        infeasible_Cd = Cd_data[~Feasible_data]

        scatter_feasible = ax2.scatter(feasible_X[:, 0], feasible_X[:, 1], c=feasible_Cd, cmap='viridis_r', marker='o', s=80, alpha=0.8, vmin=vmin, vmax=vmax)
        scatter_infeasible = ax2.scatter(infeasible_X[:, 0], infeasible_X[:, 1], c=infeasible_Cd, cmap='gray', marker='x', s=80, alpha=0.6, vmin=vmin, vmax=vmax) 
        
        if last_run_state.best_feasible_design is not None:
            best_x = last_run_state.best_feasible_design[DESIGN_VAR_NAMES].values
            ax2.plot(best_x[0], best_x[1], marker='*', markersize=18, color='red', markeredgecolor='black', markeredgewidth=1.5, linestyle='None')
        
        fig.colorbar(scatter_feasible, ax=ax2, label='Drag Coefficient (Cd)', pad=0.02).ax.tick_params(labelsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

# --- MDO Orchestrator for a single run ---
def run_mdo_single_process(run_idx, base_seed=42):
    # Set a unique random seed for this run
    current_seed = base_seed + run_idx
    np.random.seed(current_seed)
    random.seed(current_seed) # For Python's built-in random module if used

    # --- Main Agentic MDO Setup ---
    state = MDOState()
    planner = PlannerAgent(state)
    simulator = SimulationAgent(state)
    surrogate = SurrogateAgent(state)
    optimizer = OptimizationAgent(state)
    safety = SafetyAgent(state)
    reviewer = ReviewerAgent(state)

    mdo_cd_history_run = []
    ga_cd_history_run = []
    omads_mads_cd_history_run = [] 

    # --- GA Baseline: Pre-evaluate samples for THIS RUN ---
    ga_samples_X = np.random.uniform(X_BOUNDS[:, 0], X_BOUNDS[:, 1], 
                                         size=(NUM_GA_BASELINE_SAMPLES, N_DESIGN_VARS))
    for i, x in enumerate(ga_samples_X):
        cd, cl, sigma = simulator.run_simulation(x) # Use the main simulator
        feasible = safety.check_feasibility(cd, cl, sigma) # Use the main safety agent
        new_row = pd.DataFrame([list(x) + [cd, cl, sigma, feasible]], 
                               columns=state.pre_evaluated_ga_baseline_designs.columns)
        state.pre_evaluated_ga_baseline_designs = pd.concat([state.pre_evaluated_ga_baseline_designs, new_row], ignore_index=True)
    
    # --- OMADS.mads Baseline: Setup its own MDO agents ---
    omads_mads_state = MDOState() # Separate state for OMADS.mads
    omads_mads_simulator = SimulationAgent(omads_mads_state) # OMADS.mads uses its own simulator (but same underlying functions)
    omads_mads_surrogate = SurrogateAgent(omads_mads_state)
    omads_mads_optimizer = OptimizationAgent(omads_mads_state)
    omads_mads_safety = SafetyAgent(omads_mads_state) # OMADS.mads uses its own safety agent

    # --- MDO and Baselines Loop ---
    for sim_step in range(1, NUM_TOTAL_MDO_SIMS + 1):
        # --- Main Agentic MDO Step ---
        if sim_step <= NUM_INITIAL_SAMPLES: # Initial Sampling Phase
            x_mdo = np.random.uniform(X_BOUNDS[:, 0], X_BOUNDS[:, 1], size=N_DESIGN_VARS)
        else: # Optimization Loop Phase
            if sim_step == NUM_INITIAL_SAMPLES + 1: # After initial samples, update surrogate for the first time
                surrogate.update_models()
            x_mdo = optimizer.select_infill_point()
            if x_mdo is None:
                print(f"Run {run_idx+1}: Agentic MDO failed to find infill point at step {sim_step}. Padding remaining history.")
                break # Exit loop for this run

        cd, cl, sigma = simulator.run_simulation(x_mdo)
        feasible = safety.check_feasibility(cd, cl, sigma)
        state.add_design_result(x_mdo, cd, cl, sigma, feasible) 
        mdo_cd_history_run.append(state._current_run_mdo_cd_history[-1])

        # --- GA Baseline Update ---
        state.update_ga_baseline_for_current_sim(sim_step) 
        ga_cd_history_run.append(state._current_run_ga_cd_history[-1])

        # --- OMADS.mads Baseline Step ---
        if sim_step <= NUM_INITIAL_SAMPLES: # Initial Sampling Phase for OMADS.mads
            x_omads_mads = np.random.uniform(X_BOUNDS[:, 0], X_BOUNDS[:, 1], size=N_DESIGN_VARS)
        else: # Optimization Loop Phase for OMADS.mads
            if sim_step == NUM_INITIAL_SAMPLES + 1: # After initial samples, update surrogate for the first time
                omads_mads_surrogate.update_models()
            x_omads_mads = omads_mads_optimizer.select_infill_point()
            if x_omads_mads is None:
                # If OMADS.mads fails, it should also pad its history
                print(f"Run {run_idx+1}: OMADS.mads failed to find infill point at step {sim_step}. Padding remaining history.")
                # We'll just break its internal loop and let its history be padded later
                # For now, append the last known best or inf
                omads_mads_cd_history_run.append(omads_mads_cd_history_run[-1] if omads_mads_cd_history_run else np.inf)
                continue # Continue main loop, but OMADS.mads won't progress further

        cd_omads, cl_omads, sigma_omads = omads_mads_simulator.run_simulation(x_omads_mads)
        feasible_omads = omads_mads_safety.check_feasibility(cd_omads, cl_omads, sigma_omads)
        omads_mads_state.add_design_result(x_omads_mads, cd_omads, cl_omads, sigma_omads, feasible_omads)
        omads_mads_state.update_omads_mads_baseline_for_current_sim() # Update its own internal history
        omads_mads_cd_history_run.append(omads_mads_state._current_run_omads_mads_cd_history[-1])
        
        # Update surrogate models for agentic MDO if in optimization loop
        if sim_step > NUM_INITIAL_SAMPLES:
            surrogate.update_models()
            omads_mads_surrogate.update_models() # OMADS.mads also updates its models

    # Pad histories to ensure they are all the same length (NUM_TOTAL_MDO_SIMS)
    while len(mdo_cd_history_run) < NUM_TOTAL_MDO_SIMS:
        mdo_cd_history_run.append(mdo_cd_history_run[-1] if mdo_cd_history_run else np.inf)
    while len(ga_cd_history_run) < NUM_TOTAL_MDO_SIMS:
        ga_cd_history_run.append(ga_cd_history_run[-1] if ga_cd_history_run else np.inf)
    while len(omads_mads_cd_history_run) < NUM_TOTAL_MDO_SIMS:
        omads_mads_cd_history_run.append(omads_mads_cd_history_run[-1] if omads_mads_cd_history_run else np.inf)

    reviewer.rank_and_present_options() # This will update best_feasible_design for the last run

    return mdo_cd_history_run, ga_cd_history_run, omads_mads_cd_history_run, state # Return state for last run's scatter plot

# --- Main MDO Orchestrator for Ensemble Runs ---
def run_mdo_ensemble_process():
    print("LLM: Initiating Wing Design Optimization ensemble process...")

    all_mdo_histories = []
    all_ga_histories = []
    all_omads_mads_histories = [] 
    last_run_state = None # To store the state of the last run for scatter plot

    # --- Run the ensemble ---
    for i in range(N_ENSEMBLE_RUNS):
        print(f"\n--- Starting Ensemble Run {i+1}/{N_ENSEMBLE_RUNS} ---")
        mdo_hist, ga_hist, omads_mads_hist, current_run_state = run_mdo_single_process(i) # Pass run index for seed
        all_mdo_histories.append(mdo_hist)
        all_ga_histories.append(ga_hist)
        all_omads_mads_histories.append(omads_mads_hist)
        last_run_state = current_run_state # Keep the state of the last run

    print("\nLLM: All ensemble runs complete. Plotting aggregated results...")
    plot_ensemble_results(all_mdo_histories, all_ga_histories, all_omads_mads_histories, last_run_state)

    print("\nLLM: MDO process complete. Thank you for using the agentic design system!")

if __name__ == "__main__":
    run_mdo_ensemble_process()