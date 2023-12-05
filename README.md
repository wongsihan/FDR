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

If you use this code in your research, please use the following entry.

```
Wang, Sihan, Dazhi Wang, Deshan Kong, Wenhui Li, Jiaxing Wang, and Huanjie Wang. "Few‐shot multiscene fault diagnosis of rolling bearing under compound variable working conditions." IET Control Theory & Applications 16, no. 14 (2022): 1405-1416.
```
