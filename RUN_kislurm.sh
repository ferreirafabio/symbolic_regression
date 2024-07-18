


sbatch --nodes=1 --partition=alldlc_gpu-rtx2080 --time=6:00:00 slurm_launch_kislurm.sh "experiment.session_name=first_hpo_1
experiment.experiment_name=setup_1024-12_bs4x4x16_rel200_lr001
model.model_dim=1024
model.num_head=8
model.n_layers=12
dataloader.num_realisations=200
dataloader.val_samples=500
dataloader.batch_size=16
optim.lr=0.001"

sbatch --nodes=1 --partition=testdlc_gpu-rtx2080 --time=1:00:00 slurm_launch_kislurm.sh "experiment.session_name=first_hpo_1
experiment.experiment_name=setup_1024-12_bs4x4x16_rel200_lr001
model.model_dim=1024
model.num_head=8
model.n_layers=12
dataloader.num_realisations=200
dataloader.val_samples=500
dataloader.batch_size=16
optim.lr=0.001"