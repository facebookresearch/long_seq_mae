## Pre-training long-sequence MAE on Google Cloud TPUs

### Step 1: Creating your TPU VM

[TPU VMs](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) are now the recommended way to use cloud TPUs. They are directly attached to TPU chips and are faster than using standalone compute VM with TPU nodes. In the example below, we will first create a startup script and then allocate the TPU VM with it.

#### Step 1.1: Installing Google Cloud SDK and account

To use Google Cloud TPUs, first install the Google Cloud SDK and log into your Google Cloud account, and project following the instructions in https://cloud.google.com/sdk/docs/quickstart.

#### Step 1.2: Making a TPU VM startup script

First, let's create a TPU VM *startup script* by saving the content below to a file `tpu_start_mae.sh`. The startup script contains the command to run when setting up a new TPU VM node and should contain all the dependencies we want to install.

Note: this script mounts Ronghang's Filestore NFS directory (`xxx.xxx.xxx.xxx:/checkpoint`, where ImageNet is stored) to `/checkpoint`. **You should create your own Filestore NFS directory [here](https://console.cloud.google.com/filestore/instances?authuser=1) and modify the startup script accordingly.** (One should create the NFS directory in `europe-west4-a` location where we will create our TPUs below).

```bash
# (save this content to a file "tpu_start_mae.sh")

# install all the dependencies needed for training
sudo pip3 install timm==0.4.12  # use timm 0.4.12 in MAE pretraining for compatibility with PyTorch 1.10

# !!! this script mounts Ronghang's NFS directory (`xxx.xxx.xxx.xxx:/checkpoint`) to `/checkpoint`.
# !!! You should create your own NFS directory in https://console.cloud.google.com/filestore/instances?authuser=1
# !!! and modify the startup script accordingly
SHARED_FS=xxx.xxx.xxx.xxx:/checkpoint
MOUNT_POINT=/checkpoint
# try mounting 10 times to avoid transient NFS mounting failures
for i in $(seq 10); do
  ALREADY_MOUNTED=$(($(df -h | grep $SHARED_FS | wc -l) >= 1))
  if [[ $ALREADY_MOUNTED -ne 1 ]]; then
    sudo apt-get -y update
    sudo apt-get -y install nfs-common
    sudo mkdir -p $MOUNT_POINT
    sudo mount $SHARED_FS $MOUNT_POINT
    sudo chmod go+rw $MOUNT_POINT
  else
    break
  fi
done
```

#### Step 1.3: Allocating the TPU VM

Now, you can create your TPU VM with the above startup script (see [Cloud TPU PyTorch quickstart](https://cloud.google.com/tpu/docs/pytorch-quickstart-tpu-vm) for more details).

In the example below, we will create a [v3-256 TPU pod](https://cloud.google.com/tpu/docs/types-zones#europe) as used in the MAE paper. Here we create our TPUs in `europe-west4-a` location based on our TPU [quota](https://console.cloud.google.com/iam-admin/quotas?authuser=1).

```bash
# (run on your local laptop)

TPU_NAME=mae-tpu-256  # !!! change to another name you like
ZONE=europe-west4-a  # a location where we have available TPU quota
ACCELERATOR_TYPE=v3-256
STARTUP_SCRIPT=/Users/ronghanghu/workspace/gcp_scripts/tpu_start_mae.sh  # !!! change to your startup script path

RUNTIME_VERSION=tpu-vm-pt-1.10  # this is the runtime we use for PyTorch/XLA (it contains PyTorch 1.10)

# create a TPU VM (adding `--reserved` to create reserved TPUs)
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} --reserved \
  --version ${RUNTIME_VERSION} \
  --metadata-from-file=startup-script=${STARTUP_SCRIPT}
```

#### Step 1.4: Logging into the TPU VM

Now we can log into the TPU VM we just created.
```bash
# (run on your local laptop)

TPU_NAME=mae-tpu-256  # !!! change to the TPU name you created
ZONE=europe-west4-a
# it takes a while for the SSH to work after creating TPU VM
# if this command fails, just retry it
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 0
```
Here `--worker 0` means that we are going to log into the first VM node (a v3-256 TPU pod has 16 VM nodes, and each node is connected to 8 TPU cores). Note: it may take a few minutes for the setup script to finish (and it could still be running in the background after you log in). *So if you don't see your NFS directory (mounted to `/checkpoint` above), it should appear in one or two minutes.* (In case it doesn't appear after a while, see "Troubleshooting".)

After logging into TPU VM, now we can set up the codebase and datasets for our experiments.

### Step 2: Setting up codebase and datasets

On the TPU VM after logging in, we should store all the codebase in **a shared NFS directory** so that the same repo can be accessed from all VM nodes.

In the example below, we clone the repo under `/checkpoint/ronghanghu/workspace` as follows.
```bash
# (run on your TPU VM)

# !!! this should be under a shared NFS directory (you can change to a different path)
WORKSPACE_DIR=/checkpoint/ronghanghu/workspace
mkdir -p $WORKSPACE_DIR && cd $WORKSPACE_DIR
git clone https://github.com/facebookresearch/long_seq_mae.git ./long_seq_mae
```

Then, set up the pretraining datasets following [`DATA.md`](DATA.md).

Now your TPU VM is set up and we can run experiments on it.

### Step 3: Running experiments with PyTorch/XLA

#### Running MAE pretraining

Before running any experiments, first set up the gcloud ssh configuration on your TPM VM as follows (*only need to do it once*):
```bash
# (run on your TPU VM)

bash -c "cd ${HOME} && gcloud compute config-ssh --quiet"
```

Now we can run our experiments. To pretrain ViT-Large with MAE, run the following **in a tmux sesssion** (note that the first few iterations are very slow due to compilation):
```bash
# (run on your TPU VM, preferably in a tmux session)

# change to your save directory
SAVE_DIR=/checkpoint/ronghanghu/long_seq_mae/pretrain_tpu/coco+unlabeled_dup5/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2

EPOCH=800
MODEL=mae_vit_base_patch16_dec384d12h8b
BATCH_SIZE=4096
DATA_DIR=./data/coco/mae_pretrain_with_unlabeled_dup5
TPU_NAME=rh-256-0  # !!! change to the TPU name you created

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # workaround for permission issue
CODE_DIR=$(pwd)
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod-server \
  --env XRT_MESH_CONNECT_WAIT=3600 --env PYTHONUNBUFFERED=1 -- \
python3 $(realpath ${CODE_DIR}/main_pretrain.py) \
    --output_dir ${SAVE_DIR} --log_dir ${SAVE_DIR} \
    --use_xla \
    --effective_batch_size ${BATCH_SIZE} --batch_size -1 \
    --model ${MODEL} \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs ${EPOCH} \
    --blr 1.5e-4 --weight_decay 0.05 \
    --num_workers 8 \
    --resume automatic \
    --warmup_epochs 40 \
    --data_path $(realpath ${CODE_DIR}/${DATA_DIR}) \
    --input_size 448 --mask_downsampling 2 \
    2>&1 | tee ${SAVE_DIR}/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```
- Here the effective batch size is directly specified as 4096, and the per TPU batch size is unspecified (via `--batch_size -1`) and will be automatically inferred from the effective batch size. Also, `--resume automatic` automatically searches and loads the last checkpoint. See [`PRETRAIN.md`](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md) for the details of all other parameters.
- The COCO dataset in `./data/coco/mae_pretrain_with_unlabeled_dup5` contains the train2017 + unlabeled2017 splits duplicated 5 times (so that their total size is roughly comparable to ImageNet-1k), we set `EPOCH=800` to get an equivalent of 4000 epochs on COCO train2017 + unlabeled2017 splits.
- Here `--input_size 448` means that we will use an input image size of 448x448 for pretraining, which gives (L=28*28=784 sequence length under patch size 16). And `--mask_downsampling 2` means that we will jointly mask 2x2 blocks of image patches for MAE reconstruction.
- Here `--use_xla` runs the script in XLA mode. The stdout and stderr outputs are saved under `$SAVE_DIR/stdout_stderr_*.log`. Note that the training processes need to be launched on all VM nodes in a TPU pod (e.g. a v3-256 TPU pod has 16 nodes with 8 TPU cores attached to each node) and `torch_xla.distributed.xla_dist` is used to spawn the training process on all the VM nodes.
- To train ViT-Large with a long sequence (L=784) on the COCO dataset, set `MODEL=mae_vit_large_patch16_dec512d16h8b`.
- To train on the ImageNet-1k dataset, set `DATA_DIR=./data/imagenet-1k` after setting up the ImageNet-1k dataset.

After pretraining, follow [`FINETUNE_DETECTION.md`](FINETUNE_DETECTION.md) to fine-tune on the object detection task using the ViTDet codebase from Detectron2.

#### Troubleshooting

1. Note that in a few rare cases, the TPU VM startup script can fail to set up the NFS directory on all TPU VM nodes. So if your training hangs around "effective batch size: 4096" and you have the following error in your log (as saved to `$SAVE_DIR/stdout_stderr_*.log` above)
```
2022-08-09 03:04:10 10.164.0.64 [14] python3: can't open file '/checkpoint/ronghanghu/workspace/long_seq_mae/main_pretrain.py': [Errno 2] No such file or directory
```
it shows that the code cannot be accessed from node 14 (shown as `[14]`). To fix it, log into the problematic node via `gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 14` and manually mount NFS disk as in the startup script. A simple way to re-run the NFS mounting on all worker nodes is to connect to all nodes in gcloud SSH via `--worker all` as follows
```bash
TPU_NAME=mae-tpu-256  # !!! change to the TPU name you created
ZONE=europe-west4-a
SHARED_FS=xxx.xxx.xxx.xxx:/checkpoint  # !!! change to your NFS directory
MOUNT_POINT=/checkpoint  # !!! change to your mounting point

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --worker all \
  --command "(((\$(df -h | grep $SHARED_FS | wc -l) >= 1)) && echo NFS already mounted on \$(hostname)) || (sudo mkdir -p $MOUNT_POINT && sudo mount $SHARED_FS $MOUNT_POINT && sudo chmod go+rw $MOUNT_POINT && echo mounted NFS on \$(hostname))"
```

2. Sometimes if a training crashes, there can be some remaining Python processes on the VM nodes that prevent new training from being launched. One can kill them as follows.
```bash
# (run on your local laptop)

TPU_NAME=mae-tpu-256  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --worker all --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep /dev/accel0 | awk '{print \"sudo kill -9 \" \$2}' | sort | uniq | sh
sudo rm -f /tmp/libtpu_lockfile
mkdir -p /tmp/tpu_logs && sudo chmod a+w -R /tmp/tpu_logs
"
```

3. See more debugging and troubleshooting tips here: https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md.
