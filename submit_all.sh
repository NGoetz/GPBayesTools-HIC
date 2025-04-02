#!/bin/bash
#SBATCH --mem-per-cpu=8000M
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=long
#SBATCH --time=7-00:00:00
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
#training_set="data:7.7_200:05:base-star-eta:noetacut:nostarptcut.pkl data:7.7_200:05:all:noetacut:nostarptcut.pkl data:7.7_200:05:base-star-eta:3.0:nostarptcut.pkl data:7.7_200:05:all:3.0:nostarptcut.pkl data:7.7_200:05:all:noetacut:2.0.pkl  data:7.7_200:05:all:3.0:2.0.pkl data:7.7_200:05:base-star-eta-wo-phobos:noetacut:nostarptcut.pkl data:7.7:05:base-star-eta:noetacut:nostarptcut.pkl data:7.7:05:all:noetacut:nostarptcut.pkl"
#training_set="data:allenergies:allsystems:base-star-eta:noetacut:nostarptcut.pkl data:allenergies:allsystems:base-star-eta:3.0:nostarptcut.pkl"
#training_set="data:allenergies:05_2030:base-star-eta-wo-phobos:3.0:nostarptcut.pkl"
# Split the training_set variable into an array
#training_sets=( "data:allenergies:allsystems:base-star-eta:2.0:nostarptcut.pkl" "data:allenergies:allsystems:base-star-eta:2.5:nostarptcut.pkl" "data:allenergies:allsystems:base-star-eta:3.0:nostarptcut.pkl"  "data:allenergies:allsystems:base-star-eta:3.5:nostarptcut.pkl" "data:allenergies:allsystems:base:0.5:nostarptcut.pkl" "data:allenergies:allsystems:base:1.0:nostarptcut.pkl" "data:allenergies:allsystems:base:1.5:nostarptcut.pkl" "data:allenergies:allsystems:base:2.0:nostarptcut.pkl" "data:allenergies:allsystems:base:2.5:nostarptcut.pkl" "data:allenergies:allsystems:base:3.0:nostarptcut.pkl" "data:allenergies:allsystems:base:3.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base:3.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base-star-eta:3.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base:2.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base-star-eta:2.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base:1.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base-star-eta:1.5:nostarptcut.pkl" )
training_sets=("data:7.7_19.6:allsystems:base-star-eta:2.0:nostarptcut.pkl" "data:7.7_19.6:allsystems:base-star-eta:3.0:nostarptcut.pkl"  "data:7.7_19.6:allsystems:base-star-eta:3.5:nostarptcut.pkl" "data:7.7_19.6:allsystems:base:2.0:nostarptcut.pkl" "data:7.7_19.6:allsystems:base:3.0:nostarptcut.pkl"  "data:7.7_19.6:allsystems:base:3.5:nostarptcut.pkl" "data:allenergies:allsystems:base-star-eta:2.0:nostarptcut.pkl" "data:allenergies:allsystems:base-star-eta:3.0:nostarptcut.pkl"  "data:allenergies:allsystems:base-star-eta:3.5:nostarptcut.pkl" "data:allenergies:allsystems:base:2.0:nostarptcut.pkl" "data:allenergies:allsystems:base:3.0:nostarptcut.pkl"  "data:allenergies:allsystems:base:3.5:nostarptcut.pkl"  )
#training_sets=("data:allenergies:allsystems:base-star-eta:2.0:nostarptcut.pkl" "data:allenergies:allsystems:base-star-eta:3.0:nostarptcut.pkl"   "data:allenergies:allsystems:base:3.0:nostarptcut.pkl"   )

#training_sets=("data:allenergies:allsystems:base-star-eta-wo-eta-19.6:3.0:nostarptcut.pkl"  "data:allenergies:allsystems:base-star-eta-wo-eta-19.6:1.0:nostarptcut.pkl")
# Loop through each training set and run the commands in parallel
#training_sets=( "data:allenergies:allsystems:base-star-eta:2.0:nostarptcut.pkl" )
singularity exec $CON bash -c "set -e && \
source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && \
cd /lustre/hyihp/ngoetz/good_bayes"
#&& \ /lustre/hyihp/ngoetz/good_bayes/merge_folders.sh"
#echo "NO REMOVE"
for training_set in "${training_sets[@]}"; do
  singularity exec $CON bash -c "set -e && \
  source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && \
  cd /lustre/hyihp/ngoetz/good_bayes/inference && \
  /lustre/hyihp/ngoetz/good_bayes/inference/remove.sh ${training_set} && \
  cd /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC"
done
for training_set in "${training_sets[@]}"; do
  singularity exec $CON bash -c "set -e && \
  cd /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC && \
  source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && \
  python3 emulator_generation_script.py --training_set ${training_set} && \
  python3 emulator_validation_script.py --training_set ${training_set} && \
  python3 run_chain.py --training_set ${training_set}" &
done
# wait
# echo "repeat"
# for training_set in "${training_sets[@]}"; do
#   singularity exec $CON bash -c "set -e && \
#   source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && \
#   cd /lustre/hyihp/ngoetz/good_bayes/inference && \
#   /lustre/hyihp/ngoetz/good_bayes/inference/remove.sh ${training_set} && \
#   cd /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC"
# done
# for training_set in "${training_sets[@]}"; do
#   singularity exec $CON bash -c "set -e && \
#   source /lustre/hyihp/ngoetz/good_bayes/inference/GPBayesTools-HIC/bayes_venv/bin/activate && \
#   python3 emulator_generation_script.py --training_set ${training_set} && \
#   python3 emulator_validation_script.py --training_set ${training_set} && \
#   python3 run_chain.py --training_set ${training_set}" &
# done
#/lustre/hyihp/ngoetz/good_bayes/inference/remove.sh ${training_set} && \ #-> change to something which just copies
# Wait for all background processes to complete
wait
