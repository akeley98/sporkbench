#! /usr/bin/env python3
# Thank you Rin Iwai for helping with this code.

import json
import math
import sys
import os
from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import ticker


plt.rcParams.update({"font.size": 15, "figure.figsize": [8.0, 4.8]})


def get_case_name(j_kernel):
    supports_split_k = j_kernel["supports_split_k"]
    name = j_kernel["proc"]
    if supports_split_k:
        name += ", K_split=" + str(j_kernel["K_split"])
    return name


def plot(j_plot, output_dir_name):
    title = j_plot["title"]
    x_key = j_plot["x_axis"]
    fig = plt.figure(constrained_layout=True)
    plt.title(title)
    ax = fig.gca()

    want_peak = False
    if "sm_90a" in title:
        want_peak = True
        if "GEMM" in title:
            h100_peak_flops = 494.5e+12
        elif "GEMV" in title:
            # XXX Is TB 1 trillion bytes or 1 << 40 bytes?
            h100_peak_flops = 3.35e+12 / 4 * 2
        else:
            assert 0, "implement peak"
        ax2 = ax.twinx()
    if "sm_100a" in title:
        assert 0, "TODO fill in peak performance info"

    j_raw_samples = j_plot["samples"]
    if j_raw_samples:
        assert x_key in j_raw_samples[0], f"Invalid x_axis {x_key!r}"
    j_sorted_samples = sorted(j_raw_samples, key=lambda j_sample: j_sample[x_key])

    x_label = x_key
    if x_key == "M":
        if all(j.get("N") == j["M"] for j in j_sorted_samples):
            x_label += ", N"
        if all(j.get("K") == j["M"] for j in j_sorted_samples):
            x_label += ", K"

    # Try to pick one "best" Exo kernel
    # This is the one with the highest product of TFLOPS.
    exo_global_metrics = {}
    for sample_index, j_sample in enumerate(j_sorted_samples):
        for j_kernel in j_sample["kernels"]:
            is_builtin = j_kernel["is_builtin"]
            if is_builtin:
                continue
            case_name = get_case_name(j_kernel)
            flops_iqr = j_kernel["flops_iqr"]
            if flops_iqr > 0:
                flops_lg2 = math.log2(flops_iqr)
            else:
                flops_lg2 = -1e300
            exo_global_metrics[case_name] = exo_global_metrics.get(case_name, 0) + flops_lg2
    best_exo_case_name = ""
    if exo_global_metrics:
        pairs = [(metric, name) for name, metric in exo_global_metrics.items()]
        pairs.sort()
        # print(pairs[-4:])
        best_exo_case_name = pairs[-1][1]

    # Plot each built-in kernel separately.
    # Plot the max of all user kernels (specialized).
    # Plot also best_exo_case_name.
    max_tflops = 0
    x = [j_sample[x_key] for j_sample in j_sorted_samples]
    labels_y = {}
    for sample_index, j_sample in enumerate(j_sorted_samples):
        for j_kernel in j_sample["kernels"]:
            is_builtin = j_kernel["is_builtin"]
            case_name = get_case_name(j_kernel)
            flops_iqr = j_kernel["flops_iqr"]
            tflops = flops_iqr / 1e12
            max_tflops = max(max_tflops, tflops)
            if is_builtin:
                labels = (case_name, )
                assert not case_name.startswith("Exo-GPU (")
            elif case_name == best_exo_case_name:
                labels = (f"Exo-GPU (one kernel)", "Exo-GPU (specialized)")
                # labels = (f"Exo-GPU ({best_exo_case_name})", "Exo-GPU (specialized)")
            else:
                labels = ("Exo-GPU (specialized)", )
            for label in labels:
                y_per_sample = labels_y.get(label)
                if y_per_sample is None:
                    y_per_sample = [0.0] * len(j_sorted_samples)
                    labels_y[label] = y_per_sample
                y_per_sample[sample_index] = max(y_per_sample[sample_index], tflops)

    if "cutlass_Sm80_gemm" in labels_y:
        print("HACK hiding cutlass_Sm80_gemm for now")
        del labels_y["cutlass_Sm80_gemm"]

    # We will always plot exo first
    def sort_key(nm):
        if nm == "Exo-GPU (specialized)":
            key = -1
        elif nm.startswith("Exo-GPU ("):
            key = -2
        else:
            key = 0
        return (key, nm)

    for i, label in enumerate(sorted(labels_y.keys(), key=sort_key)):
        y = labels_y[label]
        if i == 0:
            color = "gray"
            marker = 'o'
            # Put a continue here if you don't want best_exo_case_name plotted.
        elif i == 1:
            color = "black"
            marker = 'o'
        else:
            color = list(mcolors.TABLEAU_COLORS.keys())[i - 2]
            marker = 'x'
        ax.plot(x, y, marker=marker, color=color, label=label)

    def int_formatter(val, *args):
        return str(int(val))

    ax.grid()
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(int_formatter))
    ax.set_xlabel(x_label)
    ax.set_ylabel("TFLOPS")

    if want_peak:
        ax2.set_ylabel("%% of peak (%.3f TFLOPS)" % (h100_peak_flops / 1e12))
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.set_ylim(0, h100_peak_flops / 1e12)
        ax2.set_ylim(0, 1)
        ax2.get_xaxis().set_visible(False)
    else:
        ax.set_ylim(0, max_tflops * 1.0625)

    fig.savefig(os.path.join(output_dir_name, f"{j_plot['name']}.png"))
    print(title, best_exo_case_name)


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
