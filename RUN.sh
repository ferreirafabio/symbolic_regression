


sbatch --nodes=4 --account=projectnucleus --partition=develbooster --time=1:00:00 slurm_launch_juwels.sh "experiment.session_name=design_model
experiment.experiment_name=test_setup_1024-12_bs4x4x8_rel200
model.model_dim=1024
model.num_head=8
model.n_layers=12
dataloader.generator.num_realizations=100
dataloader.val_samples=500
dataloader.batch_size=16
optim.lr=0.0003"


sbatch --nodes=4 --account=projectnucleus --partition=booster --time=10:00:00 slurm_launch_juwels.sh "experiment.session_name=first_hpo_1
experiment.experiment_name=setup_1024-12_bs4x4x16_rel200_lr0003
model.model_dim=1024
model.num_head=8
model.n_layers=12
dataloader.num_realisations=200
dataloader.val_samples=500
dataloader.batch_size=16
optim.lr=0.0003"

sbatch --nodes=4 --account=projectnucleus --partition=booster --time=10:00:00 slurm_launch_juwels.sh "experiment.session_name=first_hpo_1
experiment.experiment_name=setup_1024-12_bs4x4x16_rel200_lr001
model.model_dim=1024
model.num_head=8
model.n_layers=12
dataloader.num_realisations=200
dataloader.val_samples=500
dataloader.batch_size=16
optim.lr=0.001"