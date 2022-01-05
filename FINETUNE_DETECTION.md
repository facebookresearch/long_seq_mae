## Fine-tuning for Object Detection using ViTDet

The ViT models pretrained using this repository is compatible with the [ViTDet](https://arxiv.org/abs/2203.16527) format in Detectron2, so one can use the ViTDet codebase for object detection fine-tuning.

Please refer to the details in [ViTDet instructions](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) for object detection fine-tuning. Specifically, append `train.init_checkpoint=<pretrained_ckpt_path>` to the Detectron2 command to fine-tuning from a pretrained model checkpoint, and append `model.backbone.net.pretrain_img_size=448` to match the 448x448 pretraining image size.

For example, to fine-tune a ViT-Base model trained under long-sequence (L=784, pretraining image size 448x448):
```bash
# download from https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/coco%2Bunlabeled_dup5/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth
MODEL_PATH=./vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth

python3 ../../tools/lazyconfig_train_net.py \
    --config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py \
    train.init_checkpoint=\"$MODEL_PATH\" model.backbone.net.pretrain_img_size=448
```

To fine-tune a ViT-Large model trained under long-sequence (L=784, pretraining image size 448x448):
```bash
# download from https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/coco%2Bunlabeled_dup5/vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth
MODEL_PATH=./vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth

python3 ../../tools/lazyconfig_train_net.py \
    --config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_l_100ep.py \
    train.init_checkpoint=\"$MODEL_PATH\" model.backbone.net.pretrain_img_size=448
```
