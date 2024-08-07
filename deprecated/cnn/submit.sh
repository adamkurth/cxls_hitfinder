#!/bin/bash

# This script is used to submit a job to a SLURM cluster for running a CNN model script.
# It specifies the job name, partition, number of nodes, number of tasks per node, memory requirement,
# time limit, and output log file. Then, it runs a Python script for a CNN model and captures any errors.

submit_job() {
  local run="$1"
  local tasks="$2"
  local partition="$3"
  local qos="$4"
  local hours="$5"
  local path="$6" # Path to the Python script
  local slurmfile="${run}.slurm"

  # Create SLURM script
  printf "#!/bin/sh\n\n" > "${slurmfile}"
  printf "#SBATCH --time=0-${hours}:00\n" >> "${slurmfile}"
  printf "#SBATCH --ntasks=%s\n\n" "$tasks" >> "${slurmfile}"
  printf "#SBATCH --job-name=%s\n" "$run" >> "${slurmfile}"
  printf "#SBATCH --output=%s.out\n" "$run" >> "${slurmfile}"
  printf "#SBATCH --error=%s.err\n\n" "$run" >> "${slurmfile}"

  # Command lines to load environment and run the Python script
  echo "module load mamba/latest" >> "${slurmfile}" # Load the mamba module
  echo "source activate adam" >> "${slurmfile}" # adam is the name of the conda environment
  echo "python ${path}" >> "${slurmfile}" # Run the Python script
  # Submit the job
  sbatch -p "$partition" -q "$qos" -t "${hours}:00:00" "${slurmfile}"
}

main() {
  local run="$1"
  local tasks="$2"
  local partition="$3"
  local qos="$4"
  local hours="$5"
  local path="$6"
#   local tag="$7"

  # Check for mandatory arguments
  if [[ -z "$run" || -z "$tasks" || -z "$partition" || -z "$qos" || -z "$hours" || -z "$path" ]]; then
    printf "Usage: %s RUN TASKS PARTITION QOS HOURS PATH\n" "$0" >&2
    return 1
  fi

  submit_job "$run" "$tasks" "$partition" "$qos" "$hours" "$path"
}

main "$@"
