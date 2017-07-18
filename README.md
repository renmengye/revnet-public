# revnet-public
Code for paper
*The Reversible Residual Network: Backpropagation without Storing Activations.*
[[arxiv](https://arxiv.org/abs/1707.04585)]

## Installation
Customize paths first in `setup.sh` (data folder, model save folder, etc.).
```bash
git clone git://github.com/renmengye/revnet-public.git
cd revnet-public
# Change paths in setup.sh
# It also provides options to download CIFAR and ImageNet data. (ImageNet
# experiments require dataset in tfrecord format).
./setup.sh
```

## CIFAR-10/100
```bash
./run_cifar_train.py --dataset [DATASET] --model [MODEL]
```
Available values for `DATASET` are `cifar-10` and `cifar-100`.
Available values for `MODEL` are `resnet-32/110/164` and `revnet-38/110/164`.

## ImageNet
```
# Run synchronous SGD training on 4 GPUs.
./run_imagenet_train.py --model [MODEL]

# Evaluate a trained model. Launch this on a separate GPU. 
./run_imagenet_eval.py --id [EXPERIMENT ID]
```
Available values for `MODEL` are `resnet-50/101` and `revnet-54/104`.

## Provided Model Configs
See `resnet/configs/cifar_configs.py` and `resnet/configs/imagenet_configs.py`

## Pretrained RevNet Weights
You can use our pretrained model weights for the use of other applications.

RevNet-104: 23.10% error rate on ImageNet validation set (top-1 single crop).
```
wget http://www.cs.toronto.edu/~mren/revnet/pretrained/revnet-104.tar.gz
```

## Future Releases
* `tf.while_loop` implementation of RevNets, which achieves further memory
  savings.

## Citation
If you use our code, please consider cite the following:
Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse.
The Reversible Residual Network: Backpropagation without Storing Actications.
*CoRR*, abs/1707.04585, 2017.

```
@article{gomez17revnet,
  author   = {Aidan N. Gomez and Mengye Ren and Raquel Urtasun and Roger B. Grosse},
  title    = {The Reversible Residual Network: Backpropagation without Storing Activations}
  journal  = {CoRR},
  volume   = {abs/1707.04585},
  year     = {2017},
  url      = {https://arxiv.org/abs/1707.04585},
}
```
