## Setting up datasets for pretraining

Please follow the instructions below to set up the datasets for long-sequence MAE pretraining. When using Cloud TPUs, download datasets to a shared directory on the TPU VMs (such as Filestore NFS or Persistent Disk) so that the data can be accessed from all VM nodes.

### COCO (train2017 + unlabeled 2017)

In the example below, the [COCO](https://cocodataset.org/) data is downloaded to `/checkpoint/coco/`, which should have the following structure.
```
/checkpoint/coco
|_ images
|  |_ train2017
|  |  |_<im-1-name>.jpg
|  |  |_...
|  |  |_<im-N-name>.jpg
|  |  |_...
|  |_ unlabeled2017
|  |  |_<im-1-name>.jpg
|  |  |_...
|  |  |_<im-M-name>.jpg
|  |  |_...
```

Then, symlink the COCO image folders to `./data/coco/`.
```bash
cd ./data/coco/
rm train2017 unlabeled2017
ln -sf /checkpoint/coco/images/train2017 .
ln -sf /checkpoint/coco/images/unlabeled2017 .
cd ../..
```

### ImageNet-1k

In the example below, the [ImageNet-1k](https://image-net.org/) data is downloaded to `/checkpoint/imagenet-1k/`, which should have the following structure (the validation images to labeled subfolders, following the [PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet#requirements)).
```
/checkpoint/imagenet-1k
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```

Then, symlink the ImageNet-1k image folders to `./data/coco/`.
```bash
cd ./data/imagenet-1k/
rm train val
ln -sf /checkpoint/imagenet-1k/train .
ln -sf /checkpoint/imagenet-1k/val .
cd ../..
```