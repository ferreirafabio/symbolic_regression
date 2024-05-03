


sbatch --nodes=1 --account=cstdl --partition=dc-gpu-devel --time=2:00:00 slurm_launch_juwels.sh "experiment.session_name=design_model
experiment.experiment_name=test_setup_512-6_bs8_rel200
model.model_dim=512
model.num_head=4
model.n_layers=8
data_simple.num_realisations=100
data_simple.val_samples=500
data_simple.batch_size=8
optim.lr=0.0003"