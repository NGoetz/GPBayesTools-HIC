#!/bin/bash
#SBATCH --mem-per-cpu=15000M
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --time=0-09:00:00
#SBATCH --mail-user=<goetz@itp.uni-frankfurt.de>
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --job-name=RunEmuValidation

set -e
echo "Running on ${SLURM_NNODES} nodes."
echo "Number of tasks: ${SLURM_NTASKS}"
echo "Number of CPUs per node: ${SLURM_CPUS_ON_NODE}"
echo "Executed from: ${SLURM_SUBMIT_DIR}"
echo "List of nodes: ${SLURM_JOB_NODELIST}"
echo "Job id: ${SLURM_JOB_ID}"

singularity exec $CON bash -c "set -e && source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && python3 emulator_validation_script.py"
