


sbatch --nodes=1 --account=cstdl --partition=dc-gpu-devel --time=0:30:00 slurm_launch_juwels.sh "experiment.session_name=design_model
experiment.experiment_name=test_setup_1
optim.lr=0.002"