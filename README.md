# lg_dacon_competition_template


This repository shows the example of the [pytorch-lightning](https://www.pytorchlightning.ai/) and [hydra](https://hydra.cc/) on [LG-Dacon competition](https://dacon.io/competitions/official/235746/overview/description). 

### Setup Environment

#### Conda

```
conda env create -f environment.yaml -n lg
conda activate lg
```

#### .env file

you ***must*** create .env file by copying .env.sample to set environmental variables.

```
wandb_api_key=[Your Key] # "xxxxxxxxxxxxxxxxxxxxxxxx"
data_dir=[Your Path] # "/home/kuielab/lg_dacon_data_dir"
```

- about ```wandb_api_key```
   - we currently only support [wandb](https://wandb.ai/site) for logging.
   - for ```wandb_api_key```, visit [wandb](https://wandb.ai/site), go to ```setting``` copy your api key
- about ```data_dir```
   - the ***absolute*** path where datasets are stored

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
