# Non-linear Erasure-coded Computation for CNN in Distributed Edge Computing

## Quick Start

### 1. Install requirements

```
pip install -r requirements.txt
```
PyTorch is needed. You can install it by following the instructions on the [pytorch website](https://pytorch.org/get-started).

### 2. Choose a base model and train it

You need a trained base model before training your coders. In `./base_model` folder, there are some base models, and trained base model files are in `./base_model/<MODEL>/<TASK>[/<DATE>]` folder. You can choose one from them or train a new base model.

If you are to train a new base model, you can use the `./test/train_base_model.ipynb` script to train and save your base model.

### 3. Choose a dataset and split it

You need a spilted dataset before training your coders. In `./data/<DATASET>/split/<SPLIT NUM>/split_<test/train>_datasets.pt` folder, there are splited datasets. You can choose from them or split a dataset by yourself.

If you are to split a dataset, you can run `./data/split_data.py` to split data and save the splited dataset.

### 4. Choose a config file

The final thing before training your coders is to choose or write a config file. In `./config/minist/LeNet5/k4r1/mlp` folder, there is a example `eg.config.json` file which you can refer to. And the details of config files are described in `./config/README.md`.

If you are to write a new config file, you need to calculate the split input shape by hand and write it into your config file (`general -> split_data_shape`). You can use the `./cal_split_data_shape.py` script to calculate the split input shape. You need to overwrite something in this script marked by `TODO`.

### 5. Run main.py

Now you can run `main.py` to train and test your coders.

```
python ./main.py <config path>

e.g.
python ./main.py config/minist/LeNet5/k4r1/mlp/eg.config.json
python ./main.py config/minist/LeNet5/k2r1/mlp/config.json
```

## Current Problems

1. 需要重新研究数据集的分割方式。不能按空间（长/宽）分割，因为部分模型（VGG16(512, 7, 7), ResNet18(512, 1, 1)）的输出的空间大小无法被整除。

    (1) 可能的解决方式：更早地对卷积层进行分割。例如ResNet18，不让它过池化层就开始分割。目前无法实现。因为在用配置文件训练coder时会直接把分割之前的卷积区的输出张量拉直，这又过不了池化了。
    (2) 把池化层删掉。
    
2. 训练coder时会获取模型的卷积区算分割坐标，此时需要卷积区被解包成一个若干层组成的Sequence。目前的代码直接把Block当成了一个层，需要修改。