import os
import math
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description='Generate shell scripts to extract patches')
parser.add_argument('--njobs', type=int, default=5, help='Number of jobs to run in parallel')
parser.add_argument('--nproc', type=int, default=16, help='Number of processes to run in parallel')
parser.add_argument('--total_cases', type=int, required=True, help='Total number of cases to process')
parser.add_argument('--sh_path', type=str, default='/GPUFS/sysu_jhluo_1/sh', help='Path to store the shell scripts')
parser.add_argument('--sh_name', type=str, default='extract', help='Name of the shell script')
parser.add_argument('--py_path', type=str, default=None, help='Path to the Python script')
args = parser.parse_args()

# Compute values based on command-line arguments
total_cases = args.total_cases
case_per_proc = math.ceil(total_cases/(args.njobs*args.nproc))
nproc = math.ceil(total_cases/(args.njobs*case_per_proc))
assert total_cases <= args.njobs*nproc*case_per_proc

# Generate shell scripts
for job in range(args.njobs):
    with open(f"{args.sh_path}/{args.sh_name}_{job+1}.sh","w") as f:
        txt = f"""for i in {{{nproc*job+1}..{nproc*(job+1)}}}; do
        nohup python {args.py_path} -n {case_per_proc} -i $i >& ${{i}}_{args.sh_name}.out &
done"""
        f.write(txt)
