

# GPR - Generative Pretrained Regression
Collaborators: Jörg Franke, Arber Zela, Frank Hutter
____



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
python -i generator.py 

```
Plots are saved by default in `plots/`. You can access the generated expressions using `generator.expression_str` or `generator.expression_latex`. Or if you want constants being in {mantissa, exponent} format use `generator.expression_str_man_exp` or `generator.expression_latex_man_exp`.


## Test datasets
For testing we use the curated datasets from the SRBench, that contains 130 datasets from two sources: Feynman Symbolic Regression Database, and the ODE-Strogatz repository. When pulling or cloning the repository use `--recurse-submodules`, e.g.:
```bash
git clone https://github.com/EpistasisLab/pmlb.git
git pull --recurse-submodules
cd pmlb
git lfs pull
```

The following command will load the datasets in a pandas dataframe, which includes all the necessary metadata, realizations and equation strings:
```bash
PYTHONPATH=. python gpr/test/load_datasets.py
```
