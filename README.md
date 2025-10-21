# Sporkbench

Compiles your Exo-GPU kernels and compares them to cublas.
We require the `ninja` build tool (Ubuntu package `ninja-build`), `exocc`, and `nvcc` 12.3+.

TEMPORARY: we require `cutlass` cloned as a directory next to `sporkbench` (this directory); we should make this optional later.

**Compile Exo:** `./sporkbench.py [source directory]` (e.g. `examples/Sm90a_gemm`)

**Run CUDA:** `[source directory]/bin/sporkbench/run_sporkbench [data.json]`

**Plot Data:** `./plot_sporkbench.py [data.json] [outdir]`

For `Sm90a`, the middle step needs to be run remotely on `dogo` (or whatever machine that has H100s).
So currently I don't automate running the full pipeline.

You can specify the compilers with environment variables `EXO_NVCC`, `EXO_NVCC`, `EXO_CXX`, `EXO_NINJA`.

TODO: generate wrappers for nvbench, instead of our own test harness.


# Exo Compilation

`./sporkbench.py` scans `[source directory]`  recursively for Exo sources matching `exocc_*.py`; these are compiled and linked as CUDA kernels in the `[source directory]/bin/sporkbench/` directory.
These files must start with one of

* `exocc_Sm80_`: run on all supported Exo-GPU architectures

* `exocc_Sm90a_`: run on Hopper only

* `exocc_Sm100a_`: [TODO] Run on Blackwell only

Known issue: CPU-only Exo is not supported and will crash the build script.

Each `exocc_*.py` file must have an accompanying `exocc_*.py.json` file describing the kernels to test.
These JSON files must contain a list of objects (`dict`) of the format:

    {
        "algorithm": str  # "gemm" or "gemv"
        "proc": str  # Name of Exo proc
        "args": List[str]  # Names of Exo proc arguments (not including implicit ctxt)
        "A_major": str  # gemm only, "row" or "col"; gemv always has A row-major
        "B_major": str  # gemm only, "row" or "col" 
        "C_major": str  # gemm only, "row" or "col"
        f"{arg}_max": int  # (optional) maximum supported value for {arg} (e.g. K_max)
        f"{arg}_divisor": int  # (optional) this must divide {arg} (e.g. K_divisor)
    }

For gemv, the args must be some permutation of `["M", "K", "A", "x", "y"]`

For gemm, the args mut be some permutation of `["M", "N", "K", "A", "B", "C"]`, and optionally,

* `L` (batch size); if not included, `L_max=1` implicitly. `L` is always the outer (leftmost) dimension.

* `K_split` and `K_cluster` may replace `K`, in which case `K = K_split * K_cluster`.
  This indicates explicit support for split-K.
  `K_split` is the number of Exo tasks (Hopper clusters or pre-Hopper CTAs) cooperating on the K dimension.

You may write these `.py.json` files manually, or have the Python file generate them automatically.
For the latter choice, you may use the following code snippet:

    cases = [] : List[Dict]
    
    # ... generate Exo kernels and append dicts (JSON objects) to cases.
    
    import json
    json.dump(cases, open(__file__ + ".json", "w"))

Known issues: the build script doesn't understand the hidden dependency of the JSON file. Thus, we A. force a rebuild of the C++ file genereted from the JSON each time and B. if the JSON file is deleted, it will not be re-generated unless the original Python file is edited.


# Plot Data


The `run_sporkbench` output JSON file contains a list of plot objects.
Each plot object corresponds to a C++ `GemmPlotInput` or `GemvPlotInput` object (TODO: the list of these is hard-wired in C++), and contains

    {
        "name": str  # Used for file name
        "title": str
        "x_axis":  str  # "M", "N", or "K"
        "samples": List[SampleData]
    }

where each `SampleData` object contains

    {
        "L": int  # gemm only
        "M": int
        "N": int  # gemm only
        "K": int
        "kernels": List[KernelData]
    }

and each `KernelData` object contains

    {
        "proc": str
        "K_split": int  # Passed as K_split arg; 1 if K_split not supported
        "is_builtin": bool  # False iff user (build input directory) provided this kernel
        "flops_samples": List[float]
        "flops_iqr": float  # As defined by benchmark methodology (below)
    }

Multiple `KernelData` objects per `SampleData` may be defined for kernels that support split-k.


# Benchmark Methodology

After the warmup rounds (in which we additionally test output correctness), we do 100 rounds of timed testing.
In each round, we run all kernels (cublas and Exo) in a shuffled order.
This is to prevent systematic error from kernels running earlier having an advantage (lower temperature GPU).
Each sample is taken by

* Memsetting a separate section of memory in an attempt to dirty the L2 cache.

* Recording a CUDA event

* Running the tested kernel

* Recording a second CUDA event, and recording the device-side elapsed between the two events; translate this to FLOPS.

If we assume that the time taken by the device for the memset is considerably longer than the CPU time needed to record the command, then latency caused by the time taken by the CPU to record GPU commands should be minimal.
For each kernel, we sort the samples and plot the performance (in FLOPS) as the mean of the middle 50 samples (IQR).
For Exo, we plot the performance of the best-performing kernel for the problem size (issue: could this give an unfair advantage, with random noise giving Exo more ``chances'' to get lucky?).
The samples from the full 100 runs are still reported in the output JSON.
