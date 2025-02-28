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
    <a href="https://arxiv.org/abs/2308.13404"><strong>arXiv</strong></a>
    |
    <a href="#data-and-models"><strong>Data</strong></a>
  </p>
</p>

---

# Setup

## Environment

The code is developed and tested on Linux servers with NVIDIA GPU(s). We support Python 3.8+ and PyTorch 1.11+. After getting a required Python environment, you can setup the rest of requirements by running:

```bash
git clone https://github.com/iamNCJ/NRHints.git
cd NRHints
pip install -r requirements.txt
```

## Data

Our data is compatible with `NeRF Blender Dataset`, except that we have extra fields in each frame for point light position.

You can download our data [here](#data-and-models).

# Usage

## Configuration System

We use [tyro](https://github.com/brentyi/tyro) for configuration management. Description to all configurations can be found by running `python main.py -h`.

## Training

```bash
python3 main.py config:nr-hints --config.data.path /path/to/data/ --config.scene-name XXX
```

Refer to [train_synthetic.sh](scripts/train_synthetic.sh) and [train_real.sh](scripts/train_real.sh) for training on synthetic and real data, respectively.

> **Note**: 
> 1. Our code automatically detects the number of GPUs and uses all of them for training. If you want to use a subset of GPUs, you can set the `CUDA_VISIBLE_DEVICES` environment variable.
> 2. For training on real captured scenes, we recommend turning on camera optimization by using `config:nr-hints-cam-opt`, which can significantly reduce the blurry effects. Since this is an improvement after the paper submission, details are described in [the author's version](https://arxiv.org/abs/2308.13404).

## Testing

```bash
python3 main.py config:nr-hints --config.data.path /path/to/data/ --config.scene-name XXX --config.evaluation-only True
```

Refer to [eval_synthetic.sh](scripts/eval_synthetic.sh) and [eval_real.sh](scripts/eval_real.sh) for testing on synthetic and real data, respectively.

Our pretrained models can be downloaded [here](#data-and-models).


# Data and Models

## Real Captured Scenes

| Object      | Data        | Pre-trained model        |
| ----------- | :----------: | :-----------: |
| Cat | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Real/Cat.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/Cat_step_1000000.ckpt) |
| Cluttered Scene  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Real/FurScene.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/FurScene_step_1000000.ckpt) |
| Pixiu Statuette   | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Real/Pixiu.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/Pixiu_step_1000000.ckpt) |
| Ornamental Fish | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Real/Fish.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/Fish_step_1000000.ckpt) |
| Cat on Decor   | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Real/CatSmall.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/CatSmall_step_1000000.ckpt) |
| Cup and Fabric   | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Real/CupFabric.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/CupFabric_step_1000000.ckpt) |
| Pikachu Statuette   | [Link Part1](https://igwebhost.azurewebsites.net/NRHints/Data/Real/Pikachu01.zip) [Part2](https://igwebhost.azurewebsites.net/NRHints/Data/Real/Pikachu02.zip)       | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Real/Pikachu_step_1000000.ckpt) |

## Synthetic Rendered Scenes

> **Note**:
> Our synthetic data rendering scripts are released at [here](https://github.com/iamNCJ/bpy-helper/tree/main/examples/nrhints-data).

| Object      | Data        | Pre-trained model        |
| ----------- | :----------: | :-----------: |
|  Diffuse  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_Diffuse_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_Diffuse_PL_500_step_1000000.ckpt) |
|  Metallic  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_Metal_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_Metal_PL_500_step_1000000.ckpt) |
|  Glossy-Metal  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_Metal_Rough_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_Metal_Rough_PL_500_step_1000000.ckpt) |
|  Rough-Metal  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_Metal_VeryRough_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_Metal_VeryRough_PL_500_step_1000000.ckpt) |
|  Anisotropic-Metal  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_Metal_Aniso_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_Metal_Aniso_PL_500_step_1000000.ckpt) |
|  Plastic  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_NonMetal_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_NonMetal_PL_500_step_1000000.ckpt) |
|  Glossy-Plastic  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_NonMetal_Rough_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_NonMetal_Rough_PL_500_step_1000000.ckpt) |
|  Rough-Plastic  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_NonMetal_VeryRough_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_NonMetal_VeryRough_PL_500_step_1000000.ckpt) |
|  Short-Fur  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_ShortFur_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_ShortFur_PL_500_step_1000000.ckpt) |
|  Long-Fur  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_LongFur_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_LongFur_PL_500_step_1000000.ckpt) |
|  Translucent  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Cup_Plane_SSS_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Cup_Plane_SSS_PL_500_step_1000000.ckpt) |
|  Fur-Ball  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/FurBall_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/FurBall_PL_500_step_1000000.ckpt) |
|  Basket  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Basket_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Basket_PL_500_step_1000000.ckpt) |
|  Layered Woven Ball  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Complex_Ball_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Complex_Ball_PL_500_step_1000000.ckpt) |
|  Drums  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Drums_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Drums_PL_500_step_1000000.ckpt) |
|  Hotdog | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Hotdog_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Hotdog_PL_500_step_1000000.ckpt) |
|  Lego  | [Link](https://igwebhost.azurewebsites.net/NRHints/Data/Synthetic/Lego_PL_500.zip)        | [Link](https://igwebhost.azurewebsites.net/NRHints/Model/Synthetic/Lego_PL_500_step_1000000.ckpt) |

You can use the script [download_data.sh](scripts/download_data.sh) to download all data.

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
