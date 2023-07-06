<p align="center">

  <h1 align="center">Relighting Neural Radiance Fields with Shadow and Highlight Hints</h1>
  <p align="center">
    <a href="https://www.chong-zeng.com/"><strong>Chong Zeng</strong></a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/guoch/"><strong>Guojun Chen</strong></a>
    ·
    <a href="https://yuedong.shading.me/"><strong>Yue Dong</strong></a>
    ·
    <a href="https://www.cs.wm.edu/~ppeers/"><strong>Pieter Peers</strong></a>
    ·
    <a href="https://svbrdf.github.io/"><strong>Hongzhi Wu</strong></a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/xtong/"><strong>Xin Tong</strong></a>
  </p>
  <h2 align="center">SIGGRAPH 2023 Conference Proceedings</h2>
  <div align="center">
    <img src="nrhints-teaser.png">
  </div>

  <p align="center">
  <br>
    <a href="https://nrhints.github.io/"><strong>Project Page</strong></a>
    |
    <a href="https://nrhints.github.io/pdfs/nrhints-sig23.pdf"><strong>Paper</strong></a>
    |
    <a href="TODO"><strong>Data</strong></a>
  </p>
</p>

---

# Setup

## Environment

We support Python 3.8+ and PyTorch 1.11+. After getting a required Python environment, you can setup the rest of requirements by running:

```bash
git clone https://github.com/iamNCJ/NRHints.git
cd NRHints
pip install -r requirements.txt
```

## Data

Our data is compatible with `NeRF Blender Dataset`, except that we have extra fields in each frame for point light position.

You can download our data through [link](TODO).

# Usage

## Configuration System

We use [tyro](https://github.com/brentyi/tyro) for configuration management. Description to all configurations can be found by running `python main.py -h`.

## Training

```bash
python3 main.py config:nr-hints --config.data.path /path/to/data/ --config.scene-name XXX
```

Refer to [train_synthetic.sh](scripts/train_synthetic.sh) and [train_real.sh](scripts/train_real.sh) for training on synthetic and real data, respectively.

## Testing

```bash
python3 main.py config:nr-hints --config.data.path /path/to/data/ --config.scene-name XXX --config.evaluation-only True
```

Refer to [eval_synthetic.sh](scripts/eval_synthetic.sh) and [eval_real.sh](scripts/eval_real.sh) for testing on synthetic and real data, respectively.

Our pretrained models can be downloaded through [link](TODO).

# Citation

Cite as below if you find this repository is helpful to your project:

```
@inproceedings {zeng2023nrhints,
    title      = {Relighting Neural Radiance Fields with Shadow and Highlight Hints},
    author     = {Chong Zeng and Guojun Chen and Yue Dong and Pieter Peers and Hongzhi Wu and Xin Tong},
    booktitle  = {ACM SIGGRAPH 2023 Conference Proceedings},
    year       = {2023}
}
```

# Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS) and [Nerfstudio](https://nerf.studio/). Thanks for these great projects.
