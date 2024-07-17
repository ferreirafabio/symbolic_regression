

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



## Run Training

### Install local with decault_config.yaml
```bash
python train_gpr.py 
```

### Install local with decault_config.yaml with overwritten parameters
```bash
python train_gpr.py model.model_dim=32 model.num_head=4 model.n_layers=2 dataloader.generator.num_realizations=100
```


### Install local with custom config
```bash
python train_gpr.py -c my_custom_config.yaml

```