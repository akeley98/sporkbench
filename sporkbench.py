#!/usr/bin/env python3
import sys
import os

from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional, Type
from warnings import warn

import ninja_syntax
import shlex

sporkbench_dir = os.path.split(__file__)[0]
python3 = sys.executable
nvcc = os.environ.get("EXO_NVCC", "nvcc")
cxx = os.environ.get("EXO_CXX", "g++-12")
ninja = os.environ.get("EXO_NINJA", "ninja")

# ninja has inconsistent quoting rules I don't fully understand.
# It seems we want to use Qarg whenever the value is parsed
# by a shell (e.g. specifying the nvcc bin) and we use
# Qpath when ninja itself needs to parse the file name,
# e.g. listing the depfile/input/output files for a build
# (i.e. what would become the $in/$out variable).

def Qarg(s):
    # Quote argument
    return ninja_syntax.escape(shlex.quote(s))

# Quote path
Qpath = ninja_syntax.escape_path

if len(sys.argv) != 2:
    raise ValueError("Expect 1 argument: directory to scan (e.g. examples/)")

exocc_src_dir: str = sys.argv[1]
bin_dir: str = os.path.join(exocc_src_dir, "bin/sporkbench/")

if not os.path.isdir(exocc_src_dir):
    raise ValueError(f"Could not open directory: {exocc_src_dir}")

# Create bin directory (in user's src directory/bin/sporkbench)
# and add a gitignore file if it doesn't exist.
os.makedirs(bin_dir, exist_ok=True)
try:
    open(os.path.join(bin_dir, ".gitignore"), "x").write("*\n")
except FileExistsError:
    pass


@dataclass(slots=True)
class RunnerSource:
    cu: str
    o: str


# Scan for all .cpp and .cu files in the runner/ directory
# These will be compiled in the user's bin_dir/sporkbench_runner
runner_sources: List[RunnerSource] = []
runner_src_dir = os.path.join(sporkbench_dir, "runner")
runner_bin_dir = os.path.join(bin_dir, "sporkbench_runner")
for dname, _, fnames in os.walk(runner_src_dir):
    for fname in fnames:
        if fname.endswith(".cpp") or fname.endswith(".cu"):
            full_path = os.path.join(dname, fname)
            rel_dir = os.path.relpath(dname, runner_src_dir)
            o_path = os.path.join(os.path.join(runner_bin_dir, rel_dir), fname + ".o")
            runner_sources.append(RunnerSource(full_path, o_path))
            print(f"\x1b[1m\x1b[34mrunner:\x1b[0m {full_path}")


@dataclass(slots=True)
class ExoccSource:
    rel_dir: str
    stem: str  # File name, without directory and without .py
    arch: str


# Scan for all exocc_*.py files.
exocc_sources: List[ExoccSource] = []
for dname, _, fnames in os.walk(exocc_src_dir, followlinks=True):
    for fname in fnames:
        if fname.endswith(".py") and fname.startswith("exocc_"):
            full_path = os.path.join(dname, fname)
            rel_dir = os.path.relpath(dname, exocc_src_dir)
            if fname.startswith("exocc_Sm80_"):
                arch = "Sm80"
            elif fname.startswith("exocc_Sm90a_"):
                arch = "Sm90a"
            else:
                raise ValueError(f"Must start with exocc_Sm80_, or exocc_Sm90a_: {full_path}")
            for c in full_path:
                if ord(c) <= ord(' ') or c == '$':
                    warn(f"{c!r} in {full_path!r}")
            exocc_sources.append(ExoccSource(rel_dir, fname[:-3], arch))
            print(f"\x1b[1m\x1b[32mexocc:\x1b[0m {full_path}")


# We will write a build.ninja file to the bin_dir.
build_path = os.path.join(bin_dir, "build.ninja")
build = open(build_path, "w")

# Write nvcc build rules
build.write(f"""\
archcode90a = -arch compute_90a -code sm_90a,compute_90a
archcode80 = -arch compute_80 -code sm_80,sm_90a,compute_80
nvcc_bin = {Qarg(nvcc)}
cxx = {Qarg(cxx)}
python3 = {Qarg(python3)}
nvcc_args = -DNDEBUG=1 -Xcompiler -Wno-abi -I . -I {Qarg(sporkbench_dir)}/runner/ $
    -ccbin $cxx -O2 -Xcompiler -Wall -Xcompiler -fPIC -g -std=c++20 $
    --expt-extended-lambda --expt-relaxed-constexpr

rule nvcc_Sm80
  command = $nvcc_bin -c --ptxas-options=-O3 -lineinfo $nvcc_args $archcode80 $in -o $out -MD -MF $out.d

rule nvcc_Sm90a
  command = $nvcc_bin -c --ptxas-options=-O3 -lineinfo $nvcc_args $archcode90a $in -o $out -MD -MF $out.d

rule link
  command = $nvcc_bin $nvcc_args $in -o $out -lcuda -lcublas
""")


# Write Exo -> CUDA -> .o builds for user's Exo files.
# We also remember the names of generated .h/.c/.cu/.cuh files without extensions.
# Finally, we also remember the list of associated JSON files.
exocc_bin_stems = []
exocc_json_list = []
for i, src_info in enumerate(exocc_sources):
    py_src_name = os.path.join(exocc_src_dir, os.path.join(src_info.rel_dir, src_info.stem + ".py"))
    out_dir = os.path.join(bin_dir, os.path.join(src_info.rel_dir, src_info.stem))
    bin_stem = os.path.join(out_dir, src_info.stem)
    exocc_bin_stems.append(bin_stem)
    exocc_json_list.append(py_src_name + ".json")
    h_name = bin_stem + ".h"
    c_name = bin_stem + ".c"
    cu_name = bin_stem + ".cu"
    cuh_name = bin_stem + ".cuh"
    d_name = bin_stem + ".d"
    arch = src_info.arch
    build.write(f"\nrule exocc_{i}\n")
    build.write(f"  command = exocc $in -o {Qarg(out_dir)}\n")  # Don't use Qpath here.
    build.write(f"  depfile = {Qpath(d_name)}\n")
    build.write(f"build {Qpath(c_name)} $\n")
    build.write(f"      {Qpath(cu_name)} $\n")
    build.write(f"      {Qpath(cuh_name)} $\n")
    build.write(f"      {Qpath(h_name)}:$\n")
    build.write(f"  exocc_{i} {Qpath(py_src_name)}\n")
    build.write(f"build {Qpath(c_name + '.o')}:$\n  nvcc_{arch} {Qpath(c_name)}\n")
    build.write(f"build {Qpath(cu_name + '.o')}:$\n  nvcc_{arch} {Qpath(cu_name)}\n")


# Write CUDA -> .o builds for the runner itself.
build.write("\n")
for src_info in runner_sources:
    build.write(f"build {Qpath(src_info.o)}: nvcc_Sm80 {Qpath(src_info.cu)}\n")


# Write JSON-to-C build
json_to_cases_py = os.path.join(sporkbench_dir, "runner/json_to_cases.py")
quoted_json_list = " $\n    ".join(Qarg(j) for j in exocc_json_list)
quoted_json_h_list = " $\n    ".join(Qpath(bin_stem + ".h") for bin_stem in exocc_bin_stems)
user_cases_cpp = os.path.join(runner_bin_dir, "sporkbench_user_cases.cpp")
user_cases_o = os.path.join(runner_bin_dir, "sporkbench_user_cases.cpp.o")
build.write(f"""
# .cpp output and .h inputs are expected to be explicit outputs/inputs for ninja.
# The json files are hidden from ninja, because we want to be agnostic as to
# whether the json file is a side-effect of exocc or hand-written by the user.
# Therefore json_to_cases.py needs to be a phony target in ninja, as we can't
# explicitly encode the dependency on .json
# NB the header dependency guarantees happens-after exocc.
build phony_json_to_cases: phony
rule json_to_cases
  command = {python3} {Qarg(json_to_cases_py)} $out $in {quoted_json_list}
build {Qpath(user_cases_cpp)}: json_to_cases {quoted_json_h_list} | phony_json_to_cases

build {Qpath(user_cases_o)}: nvcc_Sm80 {Qpath(user_cases_cpp)}
""")


# Write linker build
exe_name = os.path.join(bin_dir, "run_sporkbench")
build.write(f"\nbuild {Qpath(exe_name)}: link")
build.write(f" $\n    {Qpath(user_cases_o)}")
for src in runner_sources:
    build.write(f" $\n    {Qpath(src.o)}")
for stem in exocc_bin_stems:
    build.write(f" $\n    {Qpath(stem + '.c.o')} {Qpath(stem + '.cu.o')}")
build.write("\n")


# Run ninja
# Always direct compile_commands.json to this repo, not user's dir.
build.close()
Qninja = shlex.quote(ninja)
Qninja_f = shlex.quote(build_path)
Qcompile_commands = shlex.quote(os.path.join(sporkbench_dir, "compile_commands.json"))
os.system(f"{Qninja} -f {Qninja_f} -t compdb nvcc_Sm80 nvcc_Sm90a > {Qcompile_commands}")
os.system(f"{Qninja} -f {Qninja_f}")
print(f"\x1b[1m\x1b[33mOutput executable:\x1b[0m {exe_name}")
