#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=ec-en+zh-cap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=164g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=36:00:00
#SBATCH --array=0

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}

# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

#DATE=`date +%Y%m%d`
PROJ_DIR=/home1/zliu9986/communication-translation
SAVE_ROOT=${PROJ_DIR}/Output
mkdir -p $SAVE_ROOT
pwd

# SAVE=${SAVE_ROOT}/mbart_BTen+zh_lr1.0e-5_7distractors_maxstep8192_again
dir_name=accGrad4_mbart_BTen+zh-captions_lr6.0e-6_7distractors_maxstep8192_seed43
SAVE=${SAVE_ROOT}/${dir_name}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh
cp Configs/en-zh_ec_captions.yml ${SAVE}/
./RunScripts/run_ec.sh en-zh_ec_captions.yml
