#!/bin/bash
#SBATCH --mem-per-cpu=2000M
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=long
#SBATCH --time=3-08:11:00
#SBATCH --mail-user=<goetz@itp.uni-frankfurt.de>
#SBATCH --mail-type=ALL
#SBATCH --job-name=RunFullAnalysis

set -e
echo "Running on ${SLURM_NNODES} nodes."
echo "Number of tasks: ${SLURM_NTASKS}"
echo "Number of CPUs per node: ${SLURM_CPUS_ON_NODE}"
echo "Executed from: ${SLURM_SUBMIT_DIR}"
echo "List of nodes: ${SLURM_JOB_NODELIST}"
echo "Job id: ${SLURM_JOB_ID}"
#always cuts as floats!!!
#training_set="data:7.7_200:05:base:noetacut:nostarptcut.pkl data:7.7_200:05:all:noetacut:nostarptcut.pkl data:7.7_200:05:base:3.0:nostarptcut.pkl data:7.7_200:05:all:3.0:nostarptcut.pkl data:7.7_200:05:all:noetacut:2.0.pkl  data:7.7_200:05:all:3.0:2.0.pkl data:7.7_200:05:base-wo-phobos:noetacut:nostarptcut.pkl data:7.7:05:base:noetacut:nostarptcut.pkl data:7.7:05:all:noetacut:nostarptcut.pkl"
training_set="data:allenergies:05:base:noetacut:nostarptcut.pkl"
singularity exec $CON bash -c "set -e && \
source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && \
python3 emulator_generation_script.py --training_set  ${training_set} && \
python3 emulator_validation_script.py --training_set ${training_set} && \
python3 run_chain.py --training_set ${training_set}"
