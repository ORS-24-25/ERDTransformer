### Overview: 
Code to accompany our paper: _Latent Space Planning for Multi-Object Manipulation with Environment-Aware Relational Classifiers_. [[PDF]](https://arxiv.org/pdf/2305.10857.pdf) [[Website]](https://sites.google.com/view/erelationaldynamics)


## Installation
virtualenv
```
pip install -r requirements.txt
```

conda
```
conda env create -f conda.yaml
```

### Simulation
We use the [[isaacgym]](https://bitbucket.org/robot-learning/ll4ma_isaac/src/environments/) for simulation datasets generation and simulation experiments. 

## Quick Start with Pretrained Models
- Set the PYTHONPATH: `export PYTHONPATH=/path/to/erdtransformer/relational_precond`
- Download pretrained models from [this link](https://drive.google.com/drive/folders/14UUH2VXuC73P5Sb9sh_91TDXQu6yUYsF?usp=sharing)
- Download test data for [bookshelf environment](https://drive.google.com/drive/folders/1ihnxTKk5mKMg6xBSy6vxBpc5zxZ4Q13x?usp=sharing)

```bash
python ./relational_precond/planner/RD_planner.py \
  --result_dir $PretrainedModelDir  \
  --test_dir $TestDataDir  \
  --test_max_size 1 \
  --checkpoint_path $PretrainedModelCheckpointDir/pretrained.pth
```

## Training

- Download [training data](https://drive.google.com/drive/folders/185gyrdmf2v6uqwe_qT-l0hG7HtpruV0B?usp=sharing).

```bash
python ./relational_precond/trainer/multi_object_precond/train_multi_object_precond_e2e.py  \
   --result_dir $YourResultDir  \
   --train_dir $TrainingDataDir \
   --num_epochs 50 \
   --max_size 100
```

## Citation
If you find our work useful in your research, please cite:
```
@InProceedings{huang2023erdtransformer,
author = {Yixuan Huang, Nichols Crawford Taylor, Adam Conkey, Weiyu Liu,and Tucker Hermans},
title = {{Latent Space Planning for Multi-Object Manipulation with Environment-Aware Relational Classifiers}},
url = {https://arxiv.org/pdf/2305.10857.pdf},
year = 2023
}
```
