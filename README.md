# Few-shot Multiscene Fault Diagnosis of Rolling Bearing

Official PyTorch implementation for the paper: **"Few-shot multiscene fault diagnosis of rolling bearing under compound variable working conditions"**

The project has been restructured using a large model. If there are any issues with the code, please contact me.

## 📖 Paper

[Few-shot multiscene fault diagnosis of rolling bearing under compound variable working conditions](https://ietresearch.onlinelibrary.wiley.com/share/GY5UQBH9GAJKI3P2UAEG?target=10.1049/cth2.12315)

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/few-shot-fault-diagnosis.git
cd few-shot-fault-diagnosis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

For PU dataset experiments, please download the [PU dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) and place it in the `./data` directory.

### Data Preparation

First, prepare the data:
```bash
python scripts/prepare_data.py -w 5 --data_root ./data --output_dir ./tempdata
```

### Training

#### Quick Training (for testing)
```bash
python scripts/quick_train.py
```

#### Full Training
```bash
python scripts/train_meta_learning.py -s 1 -b 15 -e 100 -g 0
```

#### Original Training Script
```bash
python train.py -s 1 -b 15 -e 100 -g 0
```

### Testing

Test a trained model:
```bash
python test.py --feature_encoder_path ./results/feature_encoder_final.pkl --relation_network_path ./results/relation_network_final.pkl
```

### Evaluation

Comprehensive evaluation with multiple runs:
```bash
python scripts/evaluate.py --feature_encoder_path ./results/feature_encoder_final.pkl --relation_network_path ./results/relation_network_final.pkl --num_runs 5
```

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── models/             # Model definitions
│   │   ├── encoders.py     # CNN encoders (1D/2D)
│   │   ├── relation_networks.py  # Relation networks
│   │   └── classifiers.py  # Additional classifiers
│   ├── data/               # Data processing
│   │   ├── dataset.py      # Dataset classes
│   │   └── data_generator.py  # Data generation utilities
│   ├── utils/              # Utilities
│   │   ├── logger.py       # Logging utilities
│   │   └── losses.py       # Loss functions
│   └── config.py           # Configuration management
├── scripts/                # Training and evaluation scripts
│   ├── train_meta_learning.py  # Full meta-learning training
│   ├── quick_train.py      # Quick training for testing
│   ├── prepare_data.py     # Data preparation script
│   └── evaluate.py         # Model evaluation script
├── train.py                # Main training script
├── test.py                 # Testing script
├── example.py              # Usage example
├── USAGE.md               # Detailed usage guide
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## ⚙️ Parameters

### Model Parameters
- `-f, --feature_dim`: Feature dimension (default: 64)
- `-r, --relation_dim`: Relation dimension (default: 8)
- `-w, --class_num`: Number of classes (default: 5)
- `-s, --sample_num_per_class`: Samples per class (default: 1)
- `-b, --batch_num_per_class`: Batch size per class (default: 15)

### Training Parameters
- `-e, --episode`: Number of episodes (default: 100)
- `-t, --test_episode`: Test episodes (default: 1000)
- `-l, --learning_rate`: Learning rate (default: 0.001)

### Data Parameters
- `-d, --datatype`: Data type ('fft' or 'raw', default: 'fft')
- `-m, --modeltype`: Model type ('1d' or '2d', default: '1d')
- `-n, --snr`: Signal-to-noise ratio (default: -100)

## 📊 Results

The training process will generate:
- Model checkpoints in `./results/`
- Training logs in `./results/train.log`
- Accuracy plots in `./results/accuracy.png`
- Results data in `./results/accuracy.pkl`

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2022few,
  title={Few-shot multiscene fault diagnosis of rolling bearing under compound variable working conditions},
  author={Wang, Sihan and Wang, Dazhi and Kong, Deshan and Li, Wenhui and Wang, Jiaxing and Wang, Huanjie},
  journal={IET Control Theory \& Applications},
  volume={16},
  number={14},
  pages={1405--1416},
  year={2022},
  publisher={Wiley Online Library}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PU Bearing Dataset from Paderborn University
- PyTorch framework
- The original authors of the paper


