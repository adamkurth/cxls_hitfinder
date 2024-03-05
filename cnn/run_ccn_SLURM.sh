#!/bin/bash


# This script is used to submit a job to a SLURM cluster for running the cnn/src/cnn.py.
# It specifies the job name (1) number of tasks (2), partition (3), quality of service (4), hours (5), path (6) and tag (7).
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
