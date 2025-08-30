# Few-shot multiscene fault diagnosis of rolling bearing under compound variable working conditions
PyTorch code for paper: [Few-shot multiscene fault diagnosis of rolling bearing under compound variable working conditions](https://ietresearch.onlinelibrary.wiley.com/share/GY5UQBH9GAJKI3P2UAEG?target=10.1049/cth2.12315)

# Data

For PU dataset experiments, please download [PU dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) 

# Run

meta transfer learning 1 shot:

```
python main_meta_transfer.py -s 1
```

you can change -b parameter based on your GPU memory.

## Citing

If you use this code in your research, please use the following entry for citation.


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


