# 自定义ImageFolder数据集使用指南

## 📋 概述

本指南说明如何使用 `total/001` 文件夹作为 ImageFolder 格式的数据集，与 FLUX 联邦学习框架集成。

## 🗂️ 数据集结构

你的数据集结构应该如下：

```
total/001/
├── 001_ds1/
│   ├── 001__M_Left_index_finger.BMP
│   ├── 001__M_Left_little_finger.BMP
│   └── ...
├── 002_ds1/
│   ├── 002__F_Left_index_finger.BMP
│   └── ...
├── 003_ds1/
│   └── ...
...
└── 600_ds1/
    └── ...
```

- 每个子文件夹（如 `001_ds1`）代表一个类别
- 子文件夹内的图像文件属于该类别
- 总共有 600 个类别

## ⚙️ 配置步骤

### 1. 修改 `public/config.py`

将数据集名称改为 `CUSTOM_IMAGEFOLDER`，并配置相关参数：

```python
# Dataset settings
dataset_name = "CUSTOM_IMAGEFOLDER"  # 使用自定义ImageFolder数据集
drifting_type = 'static'
non_iid_type = 'label_skew_strict'  # 根据需要选择non-IID类型

# Custom ImageFolder dataset settings
custom_data_path = "./total/001"  # 数据集路径
train_test_split_ratio = 0.8  # 训练/测试集划分比例
custom_n_classes = 600  # 类别数量
custom_input_size = (90, 90)  # 图像尺寸（会自动调整）
```

### 2. 配置其他参数

根据你的需求调整以下参数：

```python
# Overall settings
n_clients = 10  # 客户端数量
n_rounds = 10  # 训练轮数
local_epochs = 2  # 本地训练轮数
batch_size = 64
lr = 0.005

# Model settings
model_name = "ResNet9"  # 对于600类的大型数据集，建议使用ResNet9
```

### 3. 运行数据集生成脚本

生成联邦学习客户端的数据分片：

```bash
python public/generate_datasets.py --fold 0 --scaling 1 --non_iid_type label_skew_strict
```

参数说明：
- `--fold`: 交叉验证折数（0-4）
- `--scaling`: 非IID程度的缩放因子（1-8）
- `--non_iid_type`: 非IID类型，支持的选项：
  - `label_skew_strict`: 标签偏斜（每个客户端只有部分类别）
  - `feature_skew_strict`: 特征偏斜（旋转/颜色变换）
  - `label_condition_skew`: 标签条件偏斜
  - `feature_condition_skew`: 特征条件偏斜

## 📊 支持的Non-IID类型

### 1. Label Skew Strict (标签偏斜)
```bash
python public/generate_datasets.py --fold 0 --scaling 1 --non_iid_type label_skew_strict
```
- `scaling=1`: 每个客户端10个类别
- `scaling=2`: 每个客户端9个类别
- ... 以此类推

### 2. Feature Skew Strict (特征偏斜)
```bash
python public/generate_datasets.py --fold 0 --scaling 1 --non_iid_type feature_skew_strict
```
- 通过旋转和颜色变换模拟特征分布差异

### 3. Label Condition Skew (标签条件偏斜)
```bash
python public/generate_datasets.py --fold 0 --scaling 1 --non_iid_type label_condition_skew
```
- 模拟 P(Y|X) 的分布差异

### 4. Feature Condition Skew (特征条件偏斜)
```bash
python public/generate_datasets.py --fold 0 --scaling 1 --non_iid_type feature_condition_skew
```
- 模拟 P(X|Y) 的分布差异

## 🔍 关键修改说明

### 修改的文件：

1. **`ANDA/utils.py`**
   - 添加了 `CUSTOM_IMAGEFOLDER` 数据集支持
   - 使用 `torchvision.datasets.ImageFolder` 加载数据
   - 自动进行训练/测试集划分（stratified split保持类别分布）

2. **`ANDA/anda.py`**
   - 在 `load_split_datasets` 函数中添加 `custom_data_path` 和 `train_test_split_ratio` 参数

3. **`public/generate_datasets.py`**
   - 支持 `CUSTOM_IMAGEFOLDER` 数据集
   - 传递自定义数据路径和划分比例

4. **`public/config.py`**
   - 添加自定义数据集配置项
   - 更新 `n_classes_dict` 和 `input_size_dict`

## 🚀 完整使用流程

### 步骤1：配置数据集
编辑 `public/config.py`：
```python
dataset_name = "CUSTOM_IMAGEFOLDER"
custom_data_path = "./total/001"
custom_n_classes = 600
custom_input_size = (90, 90)
```

### 步骤2：生成数据分片
```bash
python public/generate_datasets.py --fold 0 --scaling 1 --non_iid_type label_skew_strict
```

### 步骤3：训练模型
生成的数据会保存在 `./data/cur_datasets/` 目录下，包括：
- `client_0.npy`, `client_1.npy`, ... `client_9.npy`: 每个客户端的数据
- `n_clusters.npy`: 簇的数量

## 📝 数据加载细节

### ImageFolder格式要求
- 根目录包含多个子目录
- 每个子目录名即为类别标签
- 子目录内包含该类别的所有图像

### 自动处理
- 图像会自动转换为Tensor格式
- 进行stratified split以保持训练/测试集的类别分布
- 标签会自动从0开始重新编号

### 数据格式
加载后的数据格式：
```python
{
    'train_features': torch.Tensor,  # shape: [N, C, H, W]
    'train_labels': torch.Tensor,    # shape: [N]
    'test_features': torch.Tensor,   # shape: [M, C, H, W]
    'test_labels': torch.Tensor,     # shape: [M]
    'cluster': int                    # 簇编号
}
```

## ⚠️ 注意事项

1. **数据路径**：确保 `custom_data_path` 指向正确的目录
2. **类别数量**：`custom_n_classes` 应该与实际子目录数量一致
3. **图像尺寸**：建议设置合适的 `custom_input_size`，过大会占用更多内存
4. **模型选择**：对于600类的大型数据集，建议使用 `ResNet9` 而不是 `LeNet5`
5. **内存使用**：600个类别的数据集可能需要较大内存，注意调整 `batch_size`

## 🔧 故障排除

### 问题1：找不到数据集路径
```
FileNotFoundError: [Errno 2] No such file or directory: './total/001'
```
**解决方案**：检查 `custom_data_path` 路径是否正确，使用绝对路径或相对路径。

### 问题2：类别数量不匹配
```
AssertionError: Number of classes mismatch
```
**解决方案**：确保 `custom_n_classes` 与实际子目录数量一致。

### 问题3：内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：
- 减小 `batch_size`
- 减小 `custom_input_size`
- 使用CPU训练（设置 `gpu = -1`）

## 📊 示例输出

运行成功后，你会看到类似的输出：

```
Loading custom ImageFolder dataset from: ./total/001
Total samples in dataset: 6000
Number of classes: 600
Class to index mapping: {'001_ds1': 0, '002_ds1': 1, ...}
Train samples: 4800, Test samples: 1200
Data for client 0 saved
Data for client 1 saved
...
Number of correct clusters: 10
Datasets saved successfully!
```

## 🎯 下一步

生成数据后，你可以：
1. 运行联邦学习训练脚本
2. 使用不同的non-IID类型进行实验
3. 调整客户端数量和数据分布
4. 可视化数据分布（设置 `plot_clients = True`）
