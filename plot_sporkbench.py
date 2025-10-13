# Thank you Rin Iwai for helping with this code.

import json
import sys
import os
from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot(j_plot, output_dir_name):
    x_key = j_plot["x_axis"]
    fig = plt.figure(constrained_layout=True)
    plt.title(j_plot["title"])
    ax = fig.gca()
    # ax2 = ax.twinx()

    j_raw_samples = j_plot["samples"]
    if j_raw_samples:
        assert x_key in j_raw_samples[0], f"Invalid x_axis {x_key!r}"
    j_sorted_samples = sorted(j_raw_samples, key=lambda j_sample: j_sample[x_key])

    # Plot each built-in kernel separately, and plot the max of all user kernels.
    max_tflops = 0
    x = [j_sample[x_key] for j_sample in j_sorted_samples]
    labels_y = {}
    for sample_index, j_sample in enumerate(j_sorted_samples):
        for j_kernel in j_sample["kernels"]:
            proc_name = j_kernel["proc_name"]
            flops_iqr = j_kernel["flops_iqr"]
            tflops = flops_iqr / 1e12
            max_tflops = max(max_tflops, tflops)
            is_builtin = j_kernel["is_builtin"]
            if is_builtin:
                label = proc_name
                assert proc_name != "exo"
            else:
                label = "exo"
            y_per_sample = labels_y.get(label)
            if y_per_sample is None:
                y_per_sample = [0.0] * len(j_sorted_samples)
                labels_y[label] = y_per_sample
            y_per_sample[sample_index] = max(y_per_sample[sample_index], tflops)
    
    for i, label in enumerate(sorted(labels_y.keys())):
        y = labels_y[label]
        color = list(mcolors.TABLEAU_COLORS.keys())[i]
        ax.plot(x, y, marker='o', color=color, label=label)
    
    ax.grid()
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel("TFLOPS")
    ax.set_ylim(0, max_tflops * 1.0625)

    fig.savefig(os.path.join(output_dir_name, f"{j_plot['name']}.png"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("arguments: input.json output_dir", file=sys.stderr)
        sys.exit(1)
    _, json_filename, output_dir_name = sys.argv
    os.makedirs(output_dir_name, exist_ok=True)

    with open(json_filename, "rb") as json_f:
        j_top = json.load(json_f)

    names_to_titles = {}
    for j_plot in j_top:
        name = j_plot["name"]
        title = j_plot["title"]
        if name in names_to_titles:
            raise ValueError(f"{name!r}: name collision\n{title}\n{names_to_titles[name]}")
        names_to_titles[name] = title
        plot(j_plot, output_dir_name)
