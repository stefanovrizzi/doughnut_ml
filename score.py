import os
import json
import numpy as np
from datetime import datetime
from emukit.core.initial_designs import RandomDesign
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import ticker
import plot_settings

class DoughnutScoring:
    
    def __init__(self, model, N=2):
        """
        Scores a given model over socio-environmental performance.

        Args:
            model (SocialEnvironmentalModel): The model object from model.py.
            N (int): Number of performance indicators (default 2).
            run_id (str): Run identifier for saving figures (default None).
        """
        self.N = N # Number of indicators
        self.w = np.ones(self.N) # weights over indicators
        
        self.model = model  # link to model passed in
        self.space = model.get_policy_param_space()  # Only sample the policy parameters
        
        self.fixed_params = model.get_fixed_model_params()  # Fixed parameters that do not change

        # Timestamp for unique run identification
        self.timestamp = self.generate_timestamp()

        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'score')
        os.makedirs(self.data_dir, exist_ok=True)

        self.fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'score')
        os.makedirs(self.fig_dir, exist_ok=True)

        self.log_params()

    # Helper function to generate timestamp
    def generate_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_params(self):
        log_path = os.path.join(self.data_dir, f'score_params_{self.timestamp}.json')
        score_params = {
            "N": self.N,  # Number of trees
            "w": list(self.w),
        }
        with open(log_path, 'w') as f:
            json.dump(score_params, f, indent=2)

    def indicators_performance(self, sol):
        """
        Evaluates the performance of the indicators over time.

        Args:
            sol (OdeResult): Solution from ODE solver.

        Returns:
            np.ndarray: Time-integrated values above critical thresholds.
        """
        v = np.zeros(self.N)
        v[0] = np.trapz(sol.y[0] - self.model.xecocrit, self.model.t_eval) / self.model.t_eval[-1]
        v[1] = np.trapz(sol.y[1] - self.model.xsoccrit, self.model.t_eval) / self.model.t_eval[-1]
        return v

    def doughnut_aggregate_score(self, v):
        """
        Computes aggregate score based on indicator performance.

        Args:
            v (np.ndarray): Indicator values.

        Returns:
            float: Aggregated score, non-negotiable targets.
        """
        D = np.dot(self.w, v) + (np.prod(v > 0) - 1) * np.dot(self.w, np.maximum(0, v))
        return D

    def score_model(self, params):
        """
        Scores the system's performance under a given policy.

        Args:
            params (list): A variable number of parameters (policy params).

        Returns:
            float: Aggregate performance score.
        """
        
        # Solve the model using the full parameter set
        sol = self.model.solve_model(params)
        v = self.indicators_performance(sol)
        score = self.doughnut_aggregate_score(v)
        return score

    def score_model_over_samples(self, s):
        """
        Scores the model over a sample of policy parameters.

        Args:
            s (int): Number of samples.

        Returns:
            tuple: Arrays of sampled parameters and corresponding scores.
        """
        # Create a random design only for the policy parameters
        design = RandomDesign(self.space)
        sampled_params = design.get_samples(s)
        
        # Score the model with the sampled policy parameters and fixed model parameters
        score = np.array([self.score_model(params) for params in sampled_params])
        return sampled_params, score

    def plot_policy_score_landscape(self, S):
        """
        Plot phase diagram (score landscape) over (c, eta) policy space.

        Parameters
        ----------
        S : int
            Number of samples to generate in parameter space.
        """
        params, score = self.score_model_over_samples(S)
        c, eta = np.meshgrid(np.linspace(-0.02, 1.02, 1000), np.linspace(-0.02, 1.02, 1000))
        Z = griddata(params, score.flatten(), (c, eta), method='linear')

        lim = (max(score.max(), -score.min()) * 1.1).round(2)
        vmin, vmax = -lim, lim
        tick_locator = ticker.MaxNLocator(nbins=5)

        plt.figure(figsize=(4, 3))
        contour = plt.contourf(c, eta, Z, cmap='PiYG', levels=np.arange(-lim, lim, 0.05),
                           vmin=vmin, vmax=vmax)
        clb = plt.colorbar(contour)
        clb.ax.set_title(r'$D$')
        clb.locator = tick_locator
        clb.update_ticks()

        plt.xlabel(r'$c$')
        plt.ylabel(r'$\eta$')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        title = os.path.join(self.fig_dir, f'policy_score_landscape_{self.timestamp}.pdf')
        plt.savefig(title, dpi=300, bbox_inches='tight')
        plt.close()
