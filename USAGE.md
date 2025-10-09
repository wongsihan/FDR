# 使用指南

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保有CUDA环境（可选）
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 数据准备

```bash
# 下载PU数据集到 ./data 目录
# 然后运行数据预处理
python scripts/prepare_data.py -w 5 --data_root ./data --output_dir ./tempdata
```

### 3. 快速测试

```bash
# 使用合成数据进行快速测试
python scripts/quick_train.py
```

### 4. 完整训练

```bash
# 使用真实数据进行完整训练
python scripts/train_meta_learning.py -s 1 -b 15 -e 100 -g 0
```

### 5. 模型评估

```bash
# 评估训练好的模型
python scripts/evaluate.py \
    --feature_encoder_path ./results/feature_encoder_final.pkl \
    --relation_network_path ./results/relation_network_final.pkl \
    --num_runs 5
```

## 详细说明

### 数据格式

项目支持以下数据格式：
- **1D时间序列数据**：原始振动信号
- **2D图像数据**：时频变换后的图像

### 模型架构

- **特征编码器**：1D/2D CNN编码器，提取特征表示
- **关系网络**：计算查询样本与支持样本之间的关系
- **注意力机制**：增强特征表示能力

### 训练策略

- **元学习**：few-shot学习框架
- **迁移学习**：跨域故障诊断
- **注意力机制**：提升特征质量

### 参数调优

主要参数：
- `feature_dim`: 特征维度 (默认: 64)
- `relation_dim`: 关系网络隐藏层维度 (默认: 8)
- `class_num`: 类别数量 (默认: 5)
- `sample_num_per_class`: 每类样本数 (默认: 1)
- `learning_rate`: 学习率 (默认: 0.001)

### 结果分析

训练完成后会生成：
- 模型权重文件
- 训练日志
- 准确率曲线图
- 评估结果

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少 `batch_num_per_class` 参数
   - 使用CPU训练：设置 `-g -1`

2. **数据加载错误**
   - 检查数据路径是否正确
   - 确保数据格式正确

3. **模型收敛慢**
   - 调整学习率
   - 增加训练轮数

### 性能优化

1. **使用GPU加速**
   ```bash
   python scripts/train_meta_learning.py -g 0
   ```

2. **并行训练**
   - 使用多GPU训练（需要修改代码）

3. **内存优化**
   - 减少批次大小
   - 使用梯度累积

## 扩展功能

### 自定义数据集

1. 修改 `src/data/data_generator.py`
2. 实现数据加载函数
3. 更新配置文件

### 新模型架构

1. 在 `src/models/` 中添加新模型
2. 更新训练脚本
3. 测试模型性能

### 实验记录

建议使用以下工具记录实验：
- TensorBoard
- Weights & Biases
- MLflow
