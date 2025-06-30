# PViC: Predicate Visual Context

> Frederic Z. Zhang, Yuhui Yuan, Dylan Campbell, Zhuoyao Zhong, Stephen Gould; _Exploring Predicate Visual Context for Detecting Human-Object Interactions_; In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 10411-10421.


## Prerequisites

Use the package management tool of your choice and run the following commands after creating your environment. 
    ```bash
    # Say you are using Conda
    conda create --name pvic python=3.8
    conda activate pvic
    # Required dependencies
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install matplotlib==3.6.3 scipy==1.10.0 tqdm==4.64.1
    pip install numpy==1.24.1 timm==0.6.12
    pip install wandb==0.13.9 seaborn==0.13.0
    # Clone the repo and submodules
    git clone https://github.com/fredzzhang/pvic.git
    cd pvic
    git submodule init
    git submodule update
    pip install -e pocket
    # Build CUDA operator for MultiScaleDeformableAttention
    cd h_detr/models/ops
    python setup.py build install
    ```


## Inference

Visualisation utilities are implemented to run inference on a single image and visualise the cross-attention weights. A [reference model](https://drive.google.com/file/d/12ow476JpjrRNGMRd1f2DN_YJTOqtaOly/view?usp=sharing) is provided for demonstration purpose if you don't want to train a model yourself. Download the model and save it to `./checkpoints/`. Use the argument `--index` to select images and `--action` to specify the action index. Refer to the [lookup table](https://github.com/fredzzhang/upt/blob/main/assets/actions.txt) for action indices.

```bash
DETR=base python inference.py --resume checkpoints/pvic-detr-r50-hicodet.pth --index 4050 --action 111
```

The detected human-object pairs with scores overlayed are saved to `fig.png`, while the attention weights are saved to `pair_xx_attn_head_x.png`. 
In addition, the argument `--image-path` enables inference on custom images.

