# PViC: Predicate Visual Context

> Frederic Z. Zhang, Yuhui Yuan, Dylan Campbell, Zhuoyao Zhong, Stephen Gould; _Exploring Predicate Visual Context for Detecting Human-Object Interactions_; In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 10411-10421.


## Inference

Visualisation utilities are implemented to run inference on a single image and visualise the cross-attention weights. A [reference model](https://drive.google.com/file/d/12ow476JpjrRNGMRd1f2DN_YJTOqtaOly/view?usp=sharing) is provided for demonstration purpose if you don't want to train a model yourself. Download the model and save it to `./checkpoints/`. Use the argument `--index` to select images and `--action` to specify the action index. Refer to the [lookup table](https://github.com/fredzzhang/upt/blob/main/assets/actions.txt) for action indices.

```bash
DETR=base python inference.py --resume checkpoints/pvic-detr-r50-hicodet.pth --index 4050 --action 111
```

The detected human-object pairs with scores overlayed are saved to `fig.png`, while the attention weights are saved to `pair_xx_attn_head_x.png`. 
In addition, the argument `--image-path` enables inference on custom images.

