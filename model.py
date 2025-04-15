import os
import json
import numpy as np
from scipy.integrate import solve_ivp
from emukit.core import ContinuousParameter, ParameterSpace
from datetime import datetime
import matplotlib.pyplot as plt
import plot_settings

class ToyModel:
    """
    A model of coupled social and environmental system dynamics.

    Attributes:
        r (float): Environmental regeneration rate.
        xecocrit (float): Critical threshold for environmental indicator.
        xsoccrit (float): Critical threshold for social indicator.
        T (int): Total simulation time.
        t_span (tuple): Time span for ODE integration.
        t_eval (np.ndarray): Time evaluation points for ODE solution.
        initial_conditions (list): Initial values for [environment, social].
        N (int): Number of performance indicators.
        space (ParameterSpace): Parameter space for random design.
    """
    def __init__(self, r=1.5, xecocrit=0.2, xsoccrit=0.8, T=100, xeco0=1, xsoc0=0.05):
        """
        Initializes the model with parameters and default values.
        """
        self.r = r
        self.xecocrit = xecocrit
        self.xsoccrit = xsoccrit
        self.T = T
        self.t_span = (0, T)
        self.t_eval = np.linspace(0, T, T)
        self.initial_conditions = [xeco0, xsoc0]

        self.space = ParameterSpace([
            ContinuousParameter(r'$c$', 0, 1),
            ContinuousParameter(r'$\eta$', 0, 1)
        ])

    # Helper function to generate timestamp
    def generate_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_params(self, scenarios):
        log_path = os.path.join(self.data_dir, f'model_params_{self.timestamp}.json')
        config = {
            "r": self.r,
            "xecocrit": self.xecocrit,
            "xsoccrit": self.xsoccrit,
            "T": self.T,
            "xeco0": self.initial_conditions[0],
            "xsoc0": self.initial_conditions[1],
            "scenarios": scenarios
        }

    def get_fixed_model_params(self):
        """
        Returns the fixed model parameters that do not change during sampling.
        
        Returns:
            list: Fixed model parameters [r, xecocrit, xsoccrit].
        """
        return [self.r, self.xecocrit, self.xsoccrit]

    def get_policy_param_space(self):
        """
        Returns the parameter space for the policy parameters, which will be sampled.
        
        Returns:
            ParameterSpace: Space for policy parameters to vary (e.g., c and eta).
        """
        space = ParameterSpace([
            ContinuousParameter(r'$c$', 0, 1),  # c is the policy parameter
            ContinuousParameter(r'$\eta$', 0, 1)  # eta is the policy parameter
        ])

        return space

    def model(self, t, X, c, eta):
        """
        Defines the system of differential equations.

        Args:
            t (float): Time point.
            X (list): Current state [environment, social].
            r (float): Environmental regeneration rate.
            c (float): Policy consumption rate.
            eta (float): Social responsiveness.

        Returns:
            list: Derivatives [dxeco/dt, dxsoc/dt].
        """
        xeco, xsoc = X
        c_tilde = min(c, xeco)
        dxecodt = self.r * xeco * (1 - xeco) * (xeco > self.xecocrit) - c_tilde
        dxsocdt = xsoc * (1 - xsoc) * eta * c_tilde - min(xsoc, c - c_tilde)
        return [dxecodt, dxsocdt]

    def solve_model(self, params):
        """
        Solves the model for given policy parameters (c, eta).

        Args:
            c (float): Policy consumption rate.
            eta (float): Social responsiveness.

        Returns:
            OdeResult: Solution object from `solve_ivp`.
        """
        # Assuming `params` includes all parameters like [r, xecocrit, xsoccrit, c, eta]
        (c, eta) = params  # policy parameters
        sol = solve_ivp(self.model, self.t_span, self.initial_conditions,
                         args=(c, eta), t_eval=self.t_eval)
        
        return sol

    def save_params(self, scenarios):
    
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'model')
        os.makedirs(self.data_dir, exist_ok=True)

        self.fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'model')
        os.makedirs(self.fig_dir, exist_ok=True)

        # Timestamp for unique run identification
        self.timestamp = self.generate_timestamp()

        self.log_params(scenarios)

    def plot_indicator_trajectories(self):
        """
        Plot time evolution of environmental and social indicators for example parameter sets.
        """
                
        indicator_colors = {'env': [0.13, 0.59, 0.2], 'soc': "#EECC66"}
        scenarios = [(0.1, 0.9), (0.35, 0.98)] # (c, eta)
        #scenarios = [(0.1, 0.9), (0.2, 0.5)] # (c, eta)
        
        self.save_params(scenarios)
        
        xenv_label = r'$x_{\text{env}}$'
        xsoc_label = r'$x_{\text{soc}}$'
        fig, ax = plt.subplots(len(scenarios), 1, figsize=(7, 5), sharex=True, sharey=True)

        for i, params in enumerate(scenarios):
            sol = self.solve_model(params)
            ax[i].plot(self.t_eval, sol.y[0], label=xenv_label, color=indicator_colors['env'])
            ax[i].plot(self.t_eval, sol.y[1], label=xsoc_label, color=indicator_colors['soc'])
            ax[i].axhline(y=self.xecocrit, label=xenv_label, color=indicator_colors['env'], ls='--', lw=0.85)
            ax[i].axhline(y=self.xsoccrit, label=xsoc_label, color=indicator_colors['soc'], ls='--', lw=0.85)
            ax[i].set_ylabel('Indicators (A.U.)')
            ax[i].legend()

        ax[1].set_xlabel('Time')
        ax[1].set_xlim([0, self.T])
        ax[1].set_ylim([0, 1])
        
        title = os.path.join(self.fig_dir, f'indicator_trajectories_{self.timestamp}.pdf')
        plt.savefig(title, dpi=300)
        plt.close()
