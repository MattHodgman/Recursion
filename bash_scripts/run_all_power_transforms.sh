# This script power transforms the embeddings of all experiments to remove batch effects.

sbatch run_power_transform.sh HRCE-1
sbatch run_power_transform.sh HRCE-2
sbatch run_power_transform.sh VERO-1
sbatch run_power_transform.sh VERO-2