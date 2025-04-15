import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata
from rl import RLEnvironment, RLAgent

class RLPlotter:
    """
    A class to handle all plotting functionalities for Reinforcement Learning visualizations.
    """
    def __init__(self, env, agent, timestamp):
        """
        Parameters:
        - rl (ReinforcementLearning): An instance of the RL logic and data.
        - out_dir (str): Path to directory where plots are saved.
        """
        self.env = env
        self.agent = agent
        self.timestamp = timestamp
        self.rl_data_path = os.path.join(os.path.dirname(__file__), 'data', 'rl', f'run_{timestamp}')
                
        self.out_dir = os.path.join(os.path.dirname(__file__), 'figures', 'rl')
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.Q = np.load(self.rl_data_path+'/Q.npy')
        self.P = self.compute_P()

    def compute_P(self):
    
        P = np.zeros_like(self.Q)
        for i in range(self.env.num_c_states):
            for j in range(self.env.num_eta_states):
                possible_actions = self.env.find_possible_actions(i, j, pop=False)
                Q = self.Q[i, j, possible_actions]
                P[i, j, possible_actions] = self.agent.softmax(Q, inv_temperature=20) #self.inv_temperarure)

        return P

    def _calculate_limits(self, matrix, actions):
        """Calculate symmetric colorbar limits."""
        values = matrix[:, :, actions]
        max_val = np.max(values)
        min_val = np.min(values)
        return max(abs(min_val), abs(max_val))

    def _add_colorbar_label(self, fig, label=r'$\pi$'):
        """Add a label to a vertically-aligned colorbar."""
        label_ax = fig.add_axes([1.04, 0.885, 0.02, 0.02])
        label_ax.axis('off')
        label_ax.text(0.5, 1, label, ha='center', va='bottom')

    def _add_patch(self, ax, coord, color='white'):
        """Draw a rectangle around a grid cell."""
        ax.add_patch(
            plt.Rectangle((coord[0] - 0.5, coord[1] - 0.5), 1, 1,
                          edgecolor=color, linewidth=2, fill=False)
        )

    def plot_action_probabilities(self):
        """
        Heatmaps for each action's probability across state space.

        Parameters:
        - P (np.ndarray): A 3D array of shape [c, eta, action] with action probabilities.
        """
        actions_to_plot = np.array([0, 1, 2, 3, 5, 6, 7, 8])
        lim = self._calculate_limits(self.P, actions_to_plot)

        fig, axs = plt.subplots(3, 3, figsize=(6.7, 6.7), sharex=True, sharey=True, constrained_layout=True)
        axs = axs.flatten()

        for action in range(self.env.num_actions):
            ax = axs[action]
            q_values = self.P[:, :, action]
            im = ax.imshow(q_values.T, cmap='turbo', origin='lower', aspect='auto',
                           vmin=0, vmax=1)

            #self._add_patch(ax, self.env.sref, color='white')

            if action in [6, 7, 8]:
                ax.set_xlabel(r'$c$')
            if action in [0, 3, 6]:
                ax.set_ylabel(r'$\eta$')

            ax.set_xticks(np.arange(self.env.num_c_states))
            ax.set_yticks(np.arange(self.env.num_eta_states))
            ax.set_xticklabels(self.env.c_values.round(2))
            ax.set_yticklabels(self.env.eta_values.round(2))

        # Colorbar
        cbar_ax = fig.add_axes([1.02, 0.125, 0.02, 0.755])
        fig.colorbar(im, cax=cbar_ax)
        self._add_colorbar_label(fig, label=r'$\pi$')

        plt.savefig(f'{self.out_dir}/RL_prob_H_{self.timestamp}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined(self):
        """
        Generates a combined plot with:
        a) vector field of best actions
        b) heatmap for a selected action

        Parameters:
        - Q: Q-value table [c, eta, actions]
        - P: Policy probability table [c, eta, actions]
        """
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
        ax_vec, ax_heat = axs

        best_actions = np.argmax(self.Q, axis=2)
        S = 5000
        X, Y = self.env.get_Y(S)
        C, ETA = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))
        Z = griddata(X, Y.flatten(), (C, ETA), method='linear')

        lim = max(abs(Y.max()), abs(Y.min())) * 1.1
        contour = ax_vec.contourf(C, ETA, Z, cmap='PiYG', levels=np.arange(-lim, lim, 0.05), vmin=-lim, vmax=lim)
        clb = fig.colorbar(contour, ax=ax_vec)
        clb.ax.set_title(r'$D$')
        clb.locator = ticker.MaxNLocator(nbins=5)
        clb.update_ticks()

        self._plot_policy_arrows(ax_vec, self.Q)
        ax_vec.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'$c$', ylabel=r'$\eta$')

        # Heatmap for action 4
        lim = self._calculate_limits(self.P, np.array([0, 1, 2, 3, 5, 6, 7, 8]))
        im = ax_heat.imshow(self.P[:, :, 4].T, cmap='turbo', origin='lower', aspect='auto', vmin=-lim, vmax=1)

        self._set_heatmap_ticks(ax_heat)
        ax_heat.set(xlabel=r'$c$', ylabel=r'$\eta$')
        fig.colorbar(im, ax=ax_heat).set_label(r'$\pi$')

        # Subplot labels
        ax_vec.text(-0.25, 1.0, "a)", transform=ax_vec.transAxes, fontsize=12, fontweight='bold')
        ax_heat.text(-0.25, 1.0, "b)", transform=ax_heat.transAxes, fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/RL_combined_plot_{self.timestamp}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_policy_arrows(self, ax, Q):
        """Plot softmax-weighted arrows on policy vector field."""
        for i in range(self.env.num_c_states):
            for j in range(self.env.num_eta_states):
                possible = self.env.find_possible_actions(i, j, pop=False)
                values = Q[i, j, possible]
                probs = self.agent.softmax(values, inv_temperature=8)

                dx, dy = 0, 0
                for idx, action in enumerate(possible):
                    dx_, dy_ = self.env.action_meanings[action]
                    dx += dx_ * probs[idx]
                    dy += dy_ * probs[idx]

                scale = 0.07 * (1 - np.exp(-np.sqrt(dx ** 2 + dy ** 2)))
                ax.arrow(self.env.c_values[i], self.env.eta_values[j], dx * scale, dy * scale,
                         head_width=0.03, head_length=0.02, fc='white', ec='black', linewidth=10 * scale, width=0.01)

    def _set_heatmap_ticks(self, ax):
        """Sets symmetric tick labels for a square grid."""
        xticks = np.linspace(-0.5, self.env.num_c_states - 0.5, self.env.num_c_states // 2 + 1)
        yticks = np.linspace(-0.5, self.env.num_eta_states - 0.5, self.env.num_eta_states // 2 + 1)

        ax.set_xticks(xticks)
        ax.set_xticklabels(np.round(np.linspace(0, 1, len(xticks)), 2))
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.round(np.linspace(0, 1, len(yticks)), 2))

    def map_route(self):
        """
        Overlay route map on Q-table for a specific action, with policy arrows.

        Parameters:
        - Q_: Current Q-values
        - Q_video_: Video Q-values used to set color limits
        - figure_path: Path to save the figure
        """

        fig, ax = plt.subplots(figsize=(5, 3.5))

        im = ax.imshow(self.Q[:, :, 4].T, cmap='PiYG', origin='lower', aspect='auto',
                       extent=[0, 1, 0, 1], vmin=-np.max(self.Q), vmax=np.max(self.Q))

        self._plot_policy_arrows(ax, self.Q)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'$c$', ylabel=r'$\eta$')

        #ax.add_patch(plt.Rectangle((self.env.c_values[self.env.c_state_0] - 0.1, self.env.eta_values[self.env.eta_state_0] - .01),
        #                           0.1, 0.1, edgecolor='green', linewidth=3, fill=False))

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        fig.colorbar(im, ax=ax).set_label(r'$Q$')
        plt.tight_layout()
        
        title = f'RL_map_route_{self.timestamp}.pdf'
        figure_path = f'{self.out_dir}/{title}'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

# Running the script
def main():

    rl_env = RLEnvironment()
    rl_agent = RLAgent(rl_env, save=False)
    timestamp = "20250415_190628"
    plotter = RLPlotter(rl_env, rl_agent, timestamp)
    #plotter.plot_action_probabilities()
    #plotter.plot_combined()
    plotter.map_route()

if __name__ == "__main__":
    main()
