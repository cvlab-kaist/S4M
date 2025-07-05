# Data Preprocessing

Our code works by reading image and annotation paths from json files corresponding to different data fractions. The scripts in this folder can be used to generate random splits and writing them to the json files.

For every dataset, there is a labeled and an unlabled dataset. 
## Labeled datasets
### COCO 
Before running, make sure you have downloaded the [COCO 2017 dataset](https://cocodataset.org/#download) (train2017, val2017, annotations) and placed it under $DATA_PATH. 

We generate json files for different annotation percentages. 

```bash
DATA_PATH="/path_to/coco/"
S4M_PATH="/path_to/S4M/"

cd $S4M_PATH/tools/datasets;
mkdir s4m_dataset;
cd s4m_dataset/; mkdir coco; cd coco
ln -s $DATA_PATH/train2017 train2017
mkdir annotations;

cp $DATA_PATH/annotations/instances_train2017.json annotations/
cp $DATA_PATH/annotations/instances_val2017.json annotations/

cd ../..;
python generate_coco_splits.py --json_dir s4m_dataset/coco/annotations/instances_train2017.json -percentages 1 2 5 10
```
Note: For percentages < 1%, the output filename is 100/percentage to be an integer, e.g. 200 for 0.5%.

The labeled dataset has the following structure:
```
coco/
    annotations/
    train2017/
    val2017/
```

### Cityscapes
For Cityscapes, the default dataloader does not read a global json file but individual annotations from the gtFine folder, therefore we make different copies of the annotations directory each with a different percentage of labeled data.


```bash
cd $S4M_PATH/tools/datasets/s4m_dataset/; mkdir cityscapes
cd ../..;
python generate_cityscapes_splits.py --dataset_dir s4m_dataset/cityscapes/ --percentage 5 10 20 30 --dest_dir s4m_dataset/cityscapes/
```

The cityscapes dataset directory should then have the following struture :
```
cityscapes/
    gtFine/
        train/
        train_5/
        ...
    leftimg8bit/
        train/
        train_5/
        ... 
```


## Unlabeled datasets
The unlabeled dataset has the following structure :

```
{dataset}_unlabel/
    images/
    {dataset}_unlabel_train.json
```

To generate the json file, you should have your unlabeled images with ```{dataset}_unlabel/images``` then launch `generate_{dataset}_unl.py` to generate the json file. 

### COCO

``` bash
DATA_PATH="/path_to/coco/"
S4M_PATH="/path_to/S4M/"

cd $S4M_PATH/tools/datasets/s4m_dataset; 
mkdir coco_unlabel; cd coco_unlabel
ln -s $DATA_PATH/val2017 val2017
ln -s $DATA_PATH/train2017 images

cd ../..;
python generate_coco_unl.py --img_folder s4m_dataset/coco_unlabel
```


## Cityscapes

Create the unlabeled json:
```bash
DATA_PATH="/path_to/cityscapes"
S4M_PATH="/path_to/S4M/"

cd $S4M_PATH/tools/datasets/s4m_dataset; mkdir cityscapes_unlabel; cd cityscapes_unlabel
cd ../..;
python generate_cityscapes_unl.py --dest_img_folder $S4M_PATH/tools/datasets/s4m_dataset/cityscapes_unlabel --source_img_folder $DATA_PATH/leftImg8bit/train
```


Finally:
```bash
export DETECTRON2_DATASETS=$S4M_PATH/tools/datasets/s4m_dataset
```
