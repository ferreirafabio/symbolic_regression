


sbatch --nodes=4 --account=projectnucleus --partition=develbooster --time=1:00:00 slurm_launch_juwels.sh "experiment.session_name=design_model
experiment.experiment_name=test_setup_1024-12_bs4x4x8_rel200
model.model_dim=1024
model.num_head=8
model.n_layers=12
data_simple.num_realisations=100
data_simple.val_samples=500
data_simple.batch_size=16
optim.lr=0.0003"