## Exploring Long-Sequence Masked Autoencoders

This is the code release of the paper [Exploring Long-Sequence Masked Autoencoders](https://arxiv.org/abs/2210.07224):
```
@Article{hu2022exploring,
  author  = {Ronghang Hu and Shoubhik Debnath and Saining Xie and Xinlei Chen},
  journal = {arXiv:2210.07224},
  title   = {Exploring Long-Sequence Masked Autoencoders},
  year    = {2022},
}
```

* This repo is a modification on the [MAE repo](https://github.com/facebookresearch/mae), and supports long-sequence pretraining on both GPUs and TPUs using PyTorch.

* This repo is based on [`timm==0.4.12`](https://github.com/rwightman/pytorch-image-models), which can be installed via `pip3 install timm==0.4.12`.

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model (pretrained w/ L=784, image size 448, patch size 16)</th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr>
<td align="left">COCO (train2017 + unlabeled2017) 4000-epoch</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/coco%2Bunlabeled_dup5/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/coco%2Bunlabeled_dup5/vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth">download</a></td>
</tr>
<tr>
<td align="left">ImageNet-1k 800-epoch</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth">download</a></td>
</tr>
<tr>
<td align="left">ImageNet-1k 1600-epoch</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitb_dec384d12h8b_1600ep_img448_crop0.2-1.0_maskds2.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitl_dec512d16h8b_1600ep_img448_crop0.2-1.0_maskds2.pth">download</a></td>
</tr>
</tbody></table>

### Using the codebase

* Follow [`PRETRAIN_LONG_SEQ_TPU.md`](PRETRAIN_LONG_SEQ_TPU.md) for long-sequence pretraining on Google Cloud TPUs (which we used for our experiments).
* Follow [`PRETRAIN_LONG_SEQ_GPU.md`](PRETRAIN_LONG_SEQ_GPU.md) for long-sequence pretraining on Nvidia GPUs.
* Follow [`FINETUNE_DETECTION.md`](FINETUNE_DETECTION.md) to fine-tune on the object detection task using the ViTDet codebase from Detectron2.

In addition, this codebase is also compatible with the features in the original MAE repo. Follow [`README_MAE.md`](README_MAE.md) to use the features of the original MAE repo (such as fine-tuning on image classification).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
