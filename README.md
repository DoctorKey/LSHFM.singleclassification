# LSHFM.classification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-imagenet)](https://paperswithcode.com/sota/knowledge-distillation-on-imagenet?p=in-defense-of-feature-mimicking-for-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-coco)](https://paperswithcode.com/sota/knowledge-distillation-on-coco?p=in-defense-of-feature-mimicking-for-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-pascal-voc)](https://paperswithcode.com/sota/knowledge-distillation-on-pascal-voc?p=in-defense-of-feature-mimicking-for-knowledge)

This is the PyTorch source code for [Distilling Knowledge by Mimicking Features](https://arxiv.org/abs/2011.01424). We provide all codes for three tasks.

* single-label image classification: [LSHFM.singleclassification](https://github.com/DoctorKey/LSHFM.singleclassification)
* multi-label image classification: [LSHFM.multiclassification](https://github.com/DoctorKey/LSHFM.multiclassification)
* object detection: [LSHFM.detection](https://github.com/DoctorKey/LSHFM.detection)

## Dependence

* python3
* pytorch 1.7.1
* torchvision 0.8.2

## Prepare the dataset

Please prepare the COCO and VOC datasets by youself. Then you need to check and edit the `get_data_path` function in `src/dataset/coco_utils.py` and `src/dataset/voc_utils.py`. 


## CIFAR-100

Teacher:

* wrn_40_2
* resnet56
* resnet110
* resnet32x4
* vgg13
* ResNet50

Student:

* wrn_16_2
* wrn_40_1
* resnet20
* resnet32
* resnet8x4
* vgg8
* MobileNetV2
* ShffleNetV1
* ShffleNetV2

### Train vanilla teacher and student

Train the teacher:
```
python train_vanilla.py --model [teacher network] --gpus 0
e.g.
python train_vanilla.py --model resnet56 --gpus 0
```

Train the student:
```
python train_vanilla.py --model [student network] --gpus 0
e.g.
python train_vanilla.py --model wrn_16_2 --gpus 0
```

### Feature mimicking & knowledge distillation

Please use the below command to run experiments:
```
python train_student.py --model_s [student network] --path_t [path to the teacher] --gpus 0
e.g.
python train_student.py --model_s wrn_16_2 --path_t save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --gpus 0
```

## Imagenet

Please use the below command to run experiments:
```
python imagenet_lsh.py /mnt/ramdisk/ImageNet --gpus 0,1 -a ResNet18 --teacher-arch ResNet34 
```


## Citing this repository

If you find this code useful in your research, please consider citing us:

```
@article{LSHFM,
  title={Distilling knowledge by mimicking features},
  author={Wang, Guo-Hua and Ge, Yifan and Wu, Jianxin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
}
```

## Acknowledgement

* [RepDistiller](https://github.com/HobbitLong/RepDistiller)
