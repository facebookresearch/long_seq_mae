## Pre-training long-sequence MAE on Nvidia GPUs

First, set up the pretraining datasets following [`DATA.md`](DATA.md). Then, to pre-train ViT-Base with a long sequence (L=784) on the COCO dataset (using train2017 + unlabeled2017 splits), run the following on 16 nodes with 8 GPUs each:
```bash
# change to your save directory
SAVE_DIR=/checkpoint/ronghanghu/long_seq_mae/pretrain_gpu/coco+unlabeled_dup5/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2

EPOCH=800
MODEL=mae_vit_base_patch16_dec384d12h8b
BATCH_SIZE=4096
DATA_DIR=./data/coco/mae_pretrain_with_unlabeled_dup5

mkdir -p $SAVE_DIR
python3 ./submitit_pretrain.py \
    --job_dir ${SAVE_DIR} \
    --nodes 16 --use_volta32 --partition learnlab \
    --effective_batch_size ${BATCH_SIZE} --batch_size -1 \
    --model ${MODEL} \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs ${EPOCH} \
    --blr 1.5e-4 --weight_decay 0.05 \
    --num_workers 8 \
    --resume automatic \
    --warmup_epochs 40 \
    --data_path $(realpath $DATA_DIR) \
    --input_size 448 --mask_downsampling 2
```
- Here the effective batch size is directly specified as 4096, and the per GPU batch size is unspecified (via `--batch_size -1`) and will be automatically inferred from the effective batch size. Also, `--resume automatic` automatically searches and loads the last checkpoint. See [`PRETRAIN.md`](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md) for the details of all other parameters.
- The COCO dataset in `./data/coco/mae_pretrain_with_unlabeled_dup5` contains the train2017 + unlabeled2017 splits duplicated 5 times (so that their total size is roughly comparable to ImageNet-1k), we set `EPOCH=800` to get an equivalent of 4000 epochs on COCO train2017 + unlabeled2017 splits.
- Here `--input_size 448` means that we will use an input image size of 448x448 for pretraining, which gives (L=28*28=784 sequence length under patch size 16). And `--mask_downsampling 2` means that we will jointly mask 2x2 blocks of image patches for MAE reconstruction.
- To train ViT-Large with a long sequence (L=784) on the COCO dataset, set `MODEL=mae_vit_large_patch16_dec512d16h8b`.

- To train on the ImageNet-1k dataset, set `DATA_DIR=./data/imagenet-1k/` after setting up the ImageNet-1k dataset.

After pretraining, follow [`FINETUNE_DETECTION.md`](FINETUNE_DETECTION.md) to fine-tune on the object detection task using the ViTDet codebase from Detectron2.
