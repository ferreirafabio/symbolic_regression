

# GPR - Generative Pretrained Regression
____

other names:
GPSR - Generative Pretrained Symbolic Regression
PSR - Pretrained Symbolic Regression



## Setup Environments

### Install local CPU env
```bash
bash setup_cpu.sh 
source venv/bin/activate
```


### Install GPU env
```bash
bash setup_gpu.sh 
source venv/bin/activate
```


## Run Training local

### Install local with default_config.yaml
```bash
python train_gpr.py 
```

### Install local with default_config.yaml with overwritten parameters
```bash
python train_gpr.py model.model_dim=32 model.num_head=4 model.n_layers=2 dataloader.generator.num_realizations=100
```


### Install local with custom config
```bash
python train_gpr.py -c my_custom_config.yaml

```



## Run Training on cluster

### Install on cluster with kislurm_config.yaml
```bash
sbatch --nodes=1 --partition=testdlc_gpu-rtx2080 --time=1:00:00 slurm_launch_kislurm.sh
```

or see `RUN_kislurm.sh` for more options


## Generating data and visualizing variable dependencies
```bash
python generator.py 

```

Plots are saved by default in `plots/`

