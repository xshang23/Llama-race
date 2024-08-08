#!/bin/bash
#SBATCH --job-name=0.0_l2          # Job name
#SBATCH --output=./wave_log/l2/0.0res.txt       # Output file name
#SBATCH --error=./wave_log/l2/0.0error.txt      # Error file name
#SBATCH --time=48:00:00            # Time limit hrs:min:sec
#SBATCH --partition=gpu            # Partition to submit to
#SBATCH --gres=gpu:1               # Number of GPUs to user
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=16G                  # Memory per node
#SBATCH --mail-type=ALL            # Type of email notification-BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xshang@scu.edu # Email to which notifications will be sent
##SBATCH --nodelist=gpu03          # Request specific node

cd /WAVE/projects/newsq_scu/xiaoxiao_git/Llama-race/
module load Python
source venv/bin/activate

# pip install -r requirements.txt

python fairness_train.py --lambda_val 0.0 --model_name Meta-Llama-3-8B-Instruct --frac 0.001 --exp_name l2_norm_scale --batch_size 8 --loss_scale True --reg_type l2

#Mistral-7B-v0.1
#Llama-2-7b-chat-hf
#Meta-Llama-3-8B-Instruct


