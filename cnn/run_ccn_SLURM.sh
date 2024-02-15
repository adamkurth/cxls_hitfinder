#!/bin/bash


# This script is used to submit a job to a SLURM cluster for running the ccn_test program.
# It specifies the job name, partition, number of nodes, number of tasks per node, memory requirement,
# time limit, and output log file.

RUN=$1
TASKS=$2
PARTITION=$3
QOS=$4
HOURS=$5
PATH=$6
TAG=$7

RUN="$RUN$TAG"
SLURMFILE="$RUN.slurm"

echo "#!/bin/sh" > $SLURMFILE
echo >> $SLURMFILE

echo "#SBATCH --time=0-60:00" >> $SLURMFILE
echo "#SBATCH --ntasks=$TASKS" >> $SLURMFILE
echo >> $SLURMFILE

#echo "#SBATCH --chdir   $PWD" >> $SLURMFILE 
echo "#SBATCH --job-name  $RUN" >> $SLURMFILE
echo "#SBATCH --output    $RUN.out" >> $SLURMFILE
echo "#SBATCH --error    $RUN.err" >> $SLURMFILE
echo >> $SLURMFILE

command_line1="module load python/3.7.1"
command_line2="python ccn_test.py $PATH"
echo $command_line1 >> $SLURMFILE
echo $command_line2 >> $SLURMFILE

sbatch -p $PARTITION -q $QOS -t $HOURS:00:00 $SLURMFILE
