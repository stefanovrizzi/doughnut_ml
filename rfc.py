import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib
import seaborn as sns

from model import ToyModel
from score import DoughnutScoring

class RFC:
    def __init__(self, n_estimators=100, max_depth=3, random_state=42, save_params=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        #colors = ['darkred', 'white', 'forestgreen']
        #custom_cmap = LinearSegmentedColormap.from_list('RedWhiteGreen', colors, N=256)

        #self.cmap = custom_cmap  # Get colormap
        self.cmap = plt.get_cmap('PiYG')  # Get colormap
        #self.cmap = plt.get_cmap('seismic_r')  # Get colormap
        
        # Timestamp for unique run identification
        self.timestamp = self.generate_timestamp()

        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'rfc')
        os.makedirs(self.data_dir, exist_ok=True)

        self.fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'rfc')
        os.makedirs(self.fig_dir, exist_ok=True)
        
        self.out_dir = os.path.join(self.fig_dir, f'decision_trees_{self.timestamp}')
        os.makedirs(self.out_dir, exist_ok=True)

        self.log_params() if save_params else None

    # Helper function to generate timestamp
    def generate_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_params(self):
        log_path = os.path.join(self.data_dir, f'rfc_params_{self.timestamp}.json')
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state
        }
        with open(log_path, 'w') as f:
            json.dump(params, f, indent=2)

    def get_Y(self, S):
        model = ToyModel()
        scorer = DoughnutScoring(model)
        return scorer.score_model_over_samples(S)

    def run_accuracy_vs_samples(self, S_values=None, num_trials=10):
        
        if S_values is None:
            S_values = np.logspace(np.log10(150), np.log10(1000), 8, dtype=int)

        mean_acc = []
        std_acc = []

        for S in S_values:
            acc = []
            for _ in range(num_trials):
                X, Y = self.get_Y(S)
                y = np.squeeze(Y > 0).astype(int)
                y = 1 - y  # Flip labels

                X = pd.DataFrame(X, columns=[r"$c$", r"$\eta$"])
                y = pd.DataFrame(y, columns=["Doughnut"])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y.values.ravel(), test_size=0.3, stratify=y
                )

                clf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=None,
                    bootstrap=False,
                    n_jobs=-1
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc.append(accuracy_score(y_test, y_pred))

            mean_acc.append(np.mean(acc))
            std_acc.append(np.std(acc) / np.sqrt(num_trials))

        plt.figure(figsize=(7, 5))
        plt.errorbar(S_values, mean_acc, yerr=std_acc, fmt='o-', color='b', capsize=4)
        plt.xscale('log')
        plt.xlabel('Number of Samples (S)')
        plt.ylabel('Accuracy')
        plt.title('RFC Accuracy vs Sample Size')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'S_vs_accuracy.pdf'))
        plt.close()

    def _replace_text(self, obj):
        if isinstance(obj, plt.Text):
            txt = obj.get_text()
            txt = re.sub(r"\nsamples[^$]*class", "\nclass", txt)
            obj.set_text(txt)
        return obj

    def _fix_tree_colors(self, ax, clf, categorical_features):
    
        # Efficiently update background colors and transparency based on Gini index
        for obj in ax.findobj(matplotlib.text.Annotation):
            txt = obj.get_text()

            if '+' not in txt and '-' not in txt:
                continue  # Skip annotations that aren't class labels

            # Extract class and impurity
            txt_class = txt.split('=')[-1].strip()
            class_value = -1 if txt_class == '-' else 1
            impurity = float(txt.split('gini = ')[-1].split('\nclass')[0].strip())
            
            # Normalize and rescale color value to [0, 1]
            color_value = (class_value*(1 - 2*impurity) + 1) / 2
            color = self.cmap(color_value)

            # Apply styles
            obj.get_bbox_patch().set_facecolor(color)
            obj.set_color('white') if impurity < 0.25 else obj.set_color('black')

        # Refresh plot
        plt.draw()

    def _plot_decision_surface(self, clf, X_train, y_train, feature_names, ax):
        x_min, x_max = X_train.iloc[:, 0].min() - 0.1, X_train.iloc[:, 0].max() + 0.1
        y_min, y_max = X_train.iloc[:, 1].min() - 0.1, X_train.iloc[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        color_min = self.cmap(0.0)  # For class '-'
        color_max = self.cmap(1.0)  # For class '+'
        palette_extremes = [color_min, color_max]

        cmap = ListedColormap(palette_extremes)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

        labels = np.where(y_train == 0, '-', '+')
        sns.scatterplot(
            x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], hue=labels,
            palette=palette_extremes, edgecolor='k', ax=ax, s=40, legend=False
        )

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    def visualize_forest(self, S=500):
    
        X, Y = self.get_Y(S)
        y = np.squeeze(Y > 0).astype(int)
        feature_names = [r"$c$", r"$\eta$"]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.DataFrame(y, columns=["Doughnut"])

        X_train, _, y_train, _ = train_test_split(
            X, y.values.ravel(), test_size=0.3, stratify=y, random_state=self.random_state
        )

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            bootstrap=False,
            random_state=self.random_state
        )
        clf.fit(X_train, y_train)

        for i, tree in enumerate(clf.estimators_):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            plot_tree(
                tree,
                feature_names=feature_names,
                class_names=['-', '+'],
                filled=True, rounded=True, impurity=True,
                ax=axes[0], fontsize=8
            )

            for obj in axes[0].texts:
                self._replace_text(obj)

            self._fix_tree_colors(axes[0], tree, ['-', '+'])
            
            #axes[0].set_title(f"Tree {i + 1}")

            self._plot_decision_surface(tree, X_train, y_train, feature_names, axes[1])

            # Add subplot labels
            axes[0].text(0, 0.95, "a)", transform=axes[0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
            axes[1].text(-0.2, 0.95, "b)", transform=axes[1].transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"RFC_tree_{i}_decision.pdf"), dpi=300)
            plt.close()


if __name__ == "__main__":
    rfc = RFC(n_estimators=100, max_depth=3, random_state=42)
    #rfc.run_accuracy_vs_samples()
    rfc.visualize_forest(S=500)

