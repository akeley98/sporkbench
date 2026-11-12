#! /usr/bin/env python3
# Thank you Rin Iwai for helping with this code.

import json
import math
import sys
import os
import re
from dataclasses import dataclass
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import ticker


# HACK: leave out kernels whose proc name matches edited_* or *_edited_*.
skip_edited_kernels = True


plt.rcParams.update({"font.size": 15, "figure.figsize": [8.0, 4.8]})


def get_case_name(j_kernel):
    supports_split_k = j_kernel["supports_split_k"]
    name = j_kernel["proc"]
    if supports_split_k:
        name += ", K_split=" + str(j_kernel["K_split"])
    return name


@dataclass(slots=True, frozen=True)
class KnownGpu:
    description: str
    # Must match the run_sporkbench "gpu" info exactly.
    name: str
    sm_count: int
    # Dense (no sparsity) tensor core peaks.
    tf32_flops: float  # f32 inputs
    fp16_flops: float  # f16 and bf16 inputs

    def tensor_peak_flops(self, input_type: str) -> float:
        if input_type == "f32":
            return self.tf32_flops
        if input_type in ("f16", "bf16"):
            return self.fp16_flops
        raise ValueError(f"{self.description}: no peak recorded for {input_type} inputs")


# Source: NVIDIA H100 Tensor Core GPU datasheet,
# https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
# which listed (H100 SXM | H100 PCIe; second number of each pair is with sparsity):
#   Peak TF32 Tensor Core   494.7 TFLOPS | 989.4 TFLOPS     378 TFLOPS | 756 TFLOPS
#   Peak FP16 Tensor Core   989.4 TFLOPS | 1978.9 TFLOPS    756 TFLOPS | 1513 TFLOPS
# We use the dense numbers. (No FP8 numbers recorded yet.)
known_gpus = [
    KnownGpu("H100 SXM5", "NVIDIA H100 80GB HBM3", 132, tf32_flops=494.7e12, fp16_flops=989.4e12),
    KnownGpu("H100 PCIe", "NVIDIA H100 PCIe", 114, tf32_flops=378e12, fp16_flops=756e12),
]


def get_known_gpu(j_plot) -> KnownGpu:
    """Match the plot's "gpu" info (run_sporkbench) against known_gpus."""
    j_gpu = j_plot.get("gpu")
    if j_gpu is None:
        # JSON from before run_sporkbench recorded the GPU; these were all H100 SXM5.
        print(f"{j_plot['name']}: no GPU info, assuming H100 SXM5", file=sys.stderr)
        return known_gpus[0]
    for gpu in known_gpus:
        if j_gpu["name"] == gpu.name and j_gpu["sm_count"] == gpu.sm_count:
            return gpu
    raise ValueError(f"{j_plot['name']}: unknown GPU {j_gpu!r}; add it to known_gpus")


def exo_gemm_strategy_note(j_plot):
    """Legend note for the Exo-GPU Sm90 gemm strategy in this plot, if any.

    Sm90 Exo gemm proc names must say "_coop" or "_ping_pong" (tk_gemm_util)."""
    if not j_plot["title"].startswith("GEMM"):
        return None
    strategies = set()
    for j_sample in j_plot["samples"]:
        for j_kernel in j_sample["kernels"]:
            proc = j_kernel["proc"]
            if j_kernel["is_builtin"] or "Sm90" not in proc:
                continue
            is_coop = "_coop" in proc
            is_ping_pong = "_ping_pong" in proc
            assert is_coop != is_ping_pong, f"{proc}: expected _coop xor _ping_pong in name"
            strategies.add("coop" if is_coop else "ping_pong")
    if strategies == {"coop"}:
        return "Exo-GPU: cooperative only"
    if strategies == {"ping_pong"}:
        return "Exo-GPU: ping-pong only"
    return None


def get_peak_flops(j_plot) -> Optional[float]:
    """Peak for the "% of peak" axis (sm_90a only), else None."""
    title = j_plot["title"]
    if "sm_90a" not in title:
        return None
    if "GEMM" in title:
        # Title ends with " (sm_90a.{C}.{AB}.{AB})"; peak depends on the input (AB) type.
        m = re.search(r"\(sm_90a\.(\w+)\.(\w+)\.(\w+)\)$", title)
        assert m and m.group(2) == m.group(3), f"can't find AB type in {title!r}"
        return get_known_gpu(j_plot).tensor_peak_flops(m.group(2))
    elif "Attention" in title:
        # TODO not really peak for Attention.
        # Title ends with " (sm_90a.{T}, Hdim={Hdim}, causal)" or ", non-causal)"
        m = re.search(r"\(sm_90a\.(\w+), Hdim=\d+, (causal|non-causal)\)$", title)
        assert m, f"can't find T type in {title!r}"
        return get_known_gpu(j_plot).tensor_peak_flops(m.group(1))
    elif "GEMV" in title:
        return None  # GEMV peak (memory bandwidth) no longer maintained
    else:
        assert 0, f"implement peak for {title}"


def plot(j_plot, output_dir_name):
    title = j_plot["title"]
    x_key = j_plot["x_axis"]
    fig = plt.figure(constrained_layout=True)
    plt.title(title)
    ax = fig.gca()

    h100_peak_flops = get_peak_flops(j_plot)
    want_peak = h100_peak_flops is not None
    if want_peak:
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

    exo_specialized_label = "Exo-GPU (specialized)"
    exo_split_k_label = "Exo-GPU (K_split > 1)"

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
                # labels = (f"Exo-GPU ({best_exo_case_name})", exo_specialized_label)
            else:
                labels = (exo_specialized_label, )
            if j_kernel.get("K_split", 1) > 1:
                labels = labels + (exo_split_k_label, )
            for label in labels:
                y_per_sample = labels_y.get(label)
                if y_per_sample is None:
                    y_per_sample = [0.0] * len(j_sorted_samples)
                    labels_y[label] = y_per_sample
                y_per_sample[sample_index] = max(y_per_sample[sample_index], tflops)

    if "cutlass_Sm80_gemm" in labels_y and "sm_80" not in title:
        print("HACK hiding cutlass_Sm80_gemm for now")
        del labels_y["cutlass_Sm80_gemm"]

    # We will always plot exo first
    def sort_key(nm):
        if nm == exo_specialized_label:
            i = -3
        elif nm == exo_split_k_label:
            i = -2
        elif nm.startswith("Exo-GPU ("):
            i = -4
        else:
            i = 0
        return (i, nm)

    color_i = 0
    for label in sorted(labels_y.keys(), key=sort_key):
        y = labels_y[label]
        if label == exo_specialized_label:
            color = "black"
            marker = 'o'
        elif label == exo_split_k_label:
            color = "#60A000"
            marker = '|'
        elif label.startswith("Exo-GPU ("):
            color = "gray"
            marker = 'o'
            # Put a continue here if you don't want best_exo_case_name plotted.
        else:
            color = list(mcolors.TABLEAU_COLORS.keys())[color_i]
            color_i += 1
            marker = 'x'
        ax.plot(x, y, marker=marker, color=color, label=label)

    def int_formatter(val, *args):
        return str(int(val))

    if (note := exo_gemm_strategy_note(j_plot)) is not None:
        # Text-only legend entry (no line).
        ax.plot([], [], " ", label=note)

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
        if skip_edited_kernels:
            for j_sample in j_plot["samples"]:
                j_sample["kernels"] = [
                    j_kernel
                    for j_kernel in j_sample["kernels"]
                    if not (
                        j_kernel["proc"].startswith("edited_")
                        or "_edited_" in j_kernel["proc"]
                    )
                ]
        name = j_plot["name"]
        title = j_plot["title"]
        if name in names_to_titles:
            raise ValueError(f"{name!r}: name collision\n{title}\n{names_to_titles[name]}")
        names_to_titles[name] = title
        plot(j_plot, output_dir_name)
