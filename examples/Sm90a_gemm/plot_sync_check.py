import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rcParams.update({"font.size": 15, "figure.figsize": [8.0, 4.8]})

if __name__ == "__main__":
    this_dir = os.path.split(__file__)[0]
    in_name = os.path.join(this_dir, "benchmark_sync_check.json")
    out_name = os.path.join(this_dir, "benchmark_sync_check.png")
    with open(in_name, "r") as f:
        results = json.load(f)
    title = "Sync-check Runtime (log scale)"
    fig = plt.figure(constrained_layout=True)
    plt.title(title)
    ax = fig.gca()

    for label, sample_pairs in sorted(results.items()):
        if label == "gemv":
            marker = 'x'
            color = "darkred"
        elif label.startswith("non-split-k"):
            marker = 'o'
            color = "darkblue"
        elif label.startswith("split-k"):
            marker = 'o'
            color = "deepskyblue"
        else:
            assert 0, label
        x_samples = []
        y_samples = []
        for sz, dt in sample_pairs:
            x_samples.append(sz)
            y_samples.append(dt)
        ax.plot(x_samples, y_samples, marker=marker, color=color, label=label)

    def int_formatter(val, *args):
        return "%g" % val
    ax.grid()
    # ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    # ax.set_ylim(0, 150)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0, numticks=20))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(int_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(int_formatter))
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("Seconds")
    ax.legend()

    fig.savefig(out_name)
    print(out_name)
