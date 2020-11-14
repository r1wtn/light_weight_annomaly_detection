# Light Weight Annomaly Detection

This repository is aimded to implement a smartphone applicable annomaly detection model based on **Pytorch**.

## Referrence
- pytorch-metric-learning
  - https://github.com/KevinMusgrave/pytorch-metric-learning
  
## Training

We use coco dataset as ABNORMAL image dataset.

```bash
cd src/
python train.py --<options>
```

|  | type | description | default |
| ------ | ------ | ------ | ------ |
| --train_data_path | str | path to NORMAL image directory for training | - |
| --train_coco_path | str | path to COCO image directory for training | - |
| --valid_data_path | str | path to NORMAL image directory for validation | - |
| --vaild_coco_path | str | path to COCO image directory for validation | - |
| --batch_size | int | batch size | 16 |
| --epoch_size | int | epoch size | 100 |
| --experiment_name | str | experiment name | - |
| --save_interval | int | epoch interval to save model | 10 |

You can find `logs/`, `model_files/`, `saved_features` in this root direcotry.  

## Detection Demo

After training you can evaluate your image as NORMAL or ABNORMAL by using saved model and features.

```bash
cd src/
python detection.py --<options>
```

|  | type | description | default |
| ------ | ------ | ------ | ------ |
| --ckpt_file | str | checkpoint file of derived from previous training | - |
| --feature_file | str | feature file of derived from previous training  | - |
| --image_path_list | str | `,` splited image pathes you want to evaluate | - |
