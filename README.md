# Affine Steerable Equivariant Layer for Canonicalization of Neural Networks

Official PyTorch implementation of the ICLR 2025 paper [Affine Steerable Equivariant Layer for Canonicalization of Neural Networks](https://openreview.net/pdf?id=5i6ZZUjCA9) by Yikang Li, Yeqing Qiu, Yuxuan Chen, and Zhouchen Lin.

## Environment
```
python==3.10.13
numpy==1.25.2
kornia==0.7.3
torch==2.4.0
torchvision==0.19.0
rbf==2022.6.12+17.gad934a8
```
(For installation of the `rbf` package, see [RBF](https://github.com/treverhines/RBF))


## Experiments
We evaluate the steerable EquivarLayer in the role of a canonicalization function on MNIST and its transformed variants under three non-compact continuous groups: $\mathrm{GL}^+(2)$, the rotation-scale group, and the scale group.
Instructions to reproduce the experiments are provided below.

### $\mathrm{GL}^+(2)$ Group
 * Baselines
```bash
# Vanilla
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug vanilla --test_aug GL2
# Mild Augmentation
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug mild --test_aug GL2
# Full Augmentation
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug GL2 --test_aug GL2
```
 * EquivarLayer Canonicalizer
 ```bash
# Train and evaluate the EquivarLayer as a canonicalization function.
# Please replace <mild_aug_checkpoint> with the checkpoint file name of the mild augmentation baseline.
CUDA_VISIBLE_DEVICES=0 python main_canonicalization.py --model EquivarLayer_affine --ss_transform GL2 --test_aug GL2 --mode train --dataset mnist --log 1 --predict_checkpoint <mild_aug_checkpoint>
```

### Rotation-Scale Group
 * Baselines
```bash
# Vanilla
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug vanilla --test_aug RS
# Mild Augmentation
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug mild --test_aug RS
# Full Augmentation
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug RS --test_aug RS

# If you have trained models with vanilla and mild augmentation, you can evaluate them by loading the corresponding checkpoints:
# Vanilla (Test)
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug vanilla --test_aug RS --mode test --predict_checkpoint <vanilla_checkpoint>
# Mild Augmentation (Test)
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug mild --test_aug RS --mode test --predict_checkpoint <mild_aug_checkpoint>
```

 * EquivarLayer Canonicalizer
 ```bash
# Train and evaluate the EquivarLayer as a canonicalization function.
# Please replace <mild_aug_checkpoint> with the checkpoint file name of the mild augmentation baseline.
CUDA_VISIBLE_DEVICES=0 python main_canonicalization.py --model EquivarLayer_RS --ss_transform RS --test_aug RS --mode train --dataset mnist --log 1 --predict_checkpoint <mild_aug_checkpoint>
```


### Scale Group
 * Baselines
```bash
# Vanilla
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug vanilla --test_aug scale
# Mild Augmentation
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug mild --test_aug scale
# Full Augmentation
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug scale --test_aug scale

# If you have trained models with vanilla and mild augmentation, you can evaluate them by loading the corresponding checkpoints:
# Vanilla (Test)
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug vanilla --test_aug scale --mode test --predict_checkpoint <vanilla_checkpoint>
# Mild Augmentation (Test)
CUDA_VISIBLE_DEVICES=0 python main_prediction.py --model resnet50 --log 1 --dataset mnist --train_aug mild --test_aug scale --mode test --predict_checkpoint <mild_aug_checkpoint>
```

 * EquivarLayer Canonicalizer
```bash
# Train and evaluate the EquivarLayer as a canonicalization function.
# Please replace <mild_aug_checkpoint> with the checkpoint file name of the mild augmentation baseline.
CUDA_VISIBLE_DEVICES=0 python main_canonicalization.py --model EquivarLayer_scale --ss_transform scale --test_aug scale --mode train --dataset mnist --log 1 --predict_checkpoint <mild_aug_checkpoint>
```



## Citation
If you find this work helpful in your research, please consider citing our paper:
```bibtex
@inproceedings{li2025affine,
title={Affine Steerable Equivariant Layer for Canonicalization of Neural Networks},
author={Yikang Li and Yeqing Qiu and Yuxuan Chen and Zhouchen Lin},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}