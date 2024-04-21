#!/bin/bash
#SBATCH -p general            # Partition
#SBATCH -q public             # QOS
#SBATCH -c ${SLURM_CPUS}      # Number of Cores, specified by the user
#SBATCH --time=${SLURM_TIME}  # Compute time, specified by the user
#SBATCH --job-name=${SLURM_JOB_NAME}  # Job name, specified by the user
#SBATCH --output=slurm.%j.out  # job stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   # job stderr record
#SBATCH --export=NONE          # keep environment clean
#SBATCH --mail-type=ALL        # Email notifications for job state change

# Function to load necessary modules
load_modules() {
    module load mamba/latest
    source activate adam
}

# Function to change directory and run Python script
run_python_script() {
    python main.py || return 1
}

# Main function to control the flow of script execution
main() {
    load_modules || { printf "Failed to load modules\n" >&2; return 1; }
    run_python_script || { printf "Failed to run Python script\n" >&2; return 1; }
}

# Setting global variables from script arguments
SLURM_CPUS=$1
SLURM_TIME=$2
SLURM_JOB_NAME=$3

# Calling the main function with error handling
main || exit 1

