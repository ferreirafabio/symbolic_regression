


sbatch --nodes=1 --account=cstdl --partition=dc-gpu-devel --time=0:30:00 slurm_launch_juwels.sh "experiment.session_name=design_model
experiment.experiment_name=test_setup_1
model.model_dim=256
model.num_head=4
model.n_layers=6
data_simple.num_realisations=500
data_simple.val_samples=500
data_simple.batch_size=32
optim.lr=0.002"