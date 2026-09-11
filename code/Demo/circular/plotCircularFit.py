import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


CONDITION_COLORS = {
    0: "#6f6f6f",
    1: "#4c78a8",
    2: "#f58518",
    3: "#54a24b",
    4: "#b279a2",
    5: "#e45756",
    6: "#72b7b2",
    7: "#9d755d",
    8: "#bab0ac",
    9: "#ff9da6",
}


def should_plot(iteration, plot_every):
    return plot_every > 0 and iteration % plot_every == 0


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _half_bias(value):
    return np.asarray(_to_numpy(value), dtype=float) / 2.0


class CircularFitPlotter:
    def __init__(self, output_path, grid, prior, x_set, stimulus_space_volume):
        self.output_path = output_path
        self.grid = _to_numpy(grid)
        self.grid_size = len(self.grid)
        self.x_set = list(x_set)
        self.stimulus_space_volume = stimulus_space_volume

        plt.rcParams["font.family"] = "DejaVu Sans"
        figure, axis = plt.subplots(1, 7, figsize=(12, 2.2))
        self.figure = figure
        self.axis = axis
        self._has_variability_legend = False

        axis[0].plot(self.grid, _to_numpy(prior), color="gray")
        axis[0].set_ylim(bottom=0)
        axis[0].set_title("Prior")
        axis[1].set_title("Resources")
        axis[2].set_title("Attraction")
        axis[3].set_title("Repulsion")
        axis[4].set_title("Bias")
        axis[5].set_title("Data")
        axis[6].set_title("Variability")

        for panel in axis:
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.set_xticks([0, 180, 360], ["0", "180", "360"])

        axis[1].set_ylim(0, 0.35)
        axis[1].set_yticks([0, 0.1, 0.2, 0.3], ["0", "", "0.2", ""])
        for panel in axis[2:6]:
            panel.set_ylim(-25, 25)
            panel.set_yticks([-20, 0, 20], ["-20", "0", "20"])
        axis[3].tick_params(labelleft=False)
        axis[4].tick_params(labelleft=False)
        axis[6].set_yticks([0, 20, 40], ["0", "20", "40"])

    def _resources(self, volume, sigma_logit):
        sigma2 = 4 * torch.sigmoid(sigma_logit)
        inverse_variance = 1.0 / float(sigma2.detach().cpu())
        return _to_numpy(2 * volume * math.sqrt(inverse_variance) * self.grid_size / self.stimulus_space_volume)

    def add_condition(self, condition, sigma_logit, volume, estimate, attraction, empirical_bias, estimate_sd=None, empirical_sd=None):
        color = CONDITION_COLORS.get(condition, "black")
        estimate = _to_numpy(estimate)
        attraction = _to_numpy(attraction)

        self.axis[1].plot(self.grid, self._resources(volume.detach(), sigma_logit), color=color)
        self.axis[2].plot(self.grid, _half_bias(attraction), color=color)
        self.axis[3].plot(self.grid, _half_bias(estimate - attraction - self.grid), color=color)
        self.axis[4].plot(self.grid, _half_bias(estimate - self.grid), color=color)

        if self.x_set:
            x_values = self.grid[self.x_set]
            self.axis[5].plot(x_values, _half_bias(empirical_bias), color=color, marker=".", linewidth=1)
            if empirical_sd is not None:
                label = "human" if not self._has_variability_legend else None
                self.axis[6].plot(x_values, _half_bias(empirical_sd), color=color, marker=".", linestyle="", label=label)
        if estimate_sd is not None:
            label = "model" if not self._has_variability_legend else None
            self.axis[6].plot(self.grid, _half_bias(estimate_sd), color=color, label=label)
        if estimate_sd is not None and empirical_sd is not None and not self._has_variability_legend:
            self.axis[6].legend(frameon=False, fontsize=6, loc="upper right")
            self._has_variability_legend = True

    def save(self, iteration):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.figure.suptitle(f"iteration {iteration}", fontsize=9)
        self.axis[6].relim()
        self.axis[6].autoscale_view()
        _, ymax = self.axis[6].get_ylim()
        self.axis[6].set_ylim(0, max(ymax, 1))
        self.figure.tight_layout()
        self.figure.subplots_adjust(top=0.8)
        self.figure.savefig(self.output_path, bbox_inches="tight", transparent=True)
        plt.close(self.figure)
