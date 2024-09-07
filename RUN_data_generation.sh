




pip install antlr4-python3-runtime==4.11




sbatch -p bosch_cpu-cascadelake -nodes=1 -time=4:00:00
srun -p bosch_cpu-cascadelake --nodes=1 --time=1:00:00 --cpus-per-task=32 --bosch --pty bash

/home/frankej/workspace/ScalingSymbolicRegression/venv/bin/python gpr/data/data_creator.py -f -c feynman_arc_config data_dir="/home/frankej/workspace/data"

/home/frankej/workspace/ScalingSymbolicRegression/venv/bin/python gpr/data/data_creator.py -f -c feynman_arc_config data_dir="/home/frankej/workspace/data"



sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh -c feynman_arc_config   dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=1 dataloader.train_samples=10000000 dataloader.valid_samples=10000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh -c feynman_float_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=1 dataloader.train_samples=10000000 dataloader.valid_samples=10000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh -c feynman_noarc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=1 dataloader.train_samples=10000000 dataloader.valid_samples=10000

sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_arc_config   dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=2 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_float_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=2 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_noarc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=2 dataloader.train_samples=10000000

sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_arc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=3 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_float_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=3 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_noarc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=3 dataloader.train_samples=10000000

sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_arc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=4 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_float_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=4 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_noarc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=4 dataloader.train_samples=10000000

sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_arc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=5 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_float_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=5 dataloader.train_samples=10000000
sbatch --bosch --time=10:00:00 slurm_data_gen_kislurm.sh --no_valid -c feynman_noarc_config dataloader.data_dir="/home/frankej/workspace/gpr_data"  dataloader.generator.seed=5 dataloader.train_samples=10000000
