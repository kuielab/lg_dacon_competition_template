# LG_Dacon_Competition

This repository shows the example of the [pytorch-lightning](https://www.pytorchlightning.ai/) and [hydra](https://hydra.cc/) on [LG-Dacon competition](https://dacon.io/competitions/official/235746/overview/description). 

### Setup Environment

```
conda env create -f environment.yaml -n lg
conda activate lg
```

### Train script example

```
python main.py run_name=lg_example 
```

If you want to change the option of this code, you need to modify the YAML files on the config folder or put the other options in the training script. 
For example, you want to change the batch size, put options as follows:

```
python main.py run_name=lg_example dataloader.datasets.batch_size=64
```

### Training result example

https://wandb.ai/ielab/LG_hydra_example/reports/LG_Dacon_Competition--Vmlldzo4NjU2NzI