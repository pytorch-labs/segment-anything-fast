To run the experiments you need to update the script paths and install fire, pandas and tqdm

## Model Checkpoints

Need checkpoints from https://github.com/facebookresearch/segment-anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## COCO2017 dataset

Need to download

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

## Folder structure of experimental data

experiments_data/tmp
experiments_data/tmp/sam_coco_mask_center_cache
experiments_data/tmp/sam_eval_masks_out
experiments_data/datasets
experiments_data/datasets/coco2017
experiments_data/datasets/coco2017/val2017
experiments_data/datasets/coco2017/annotations
experiments_data/checkpoints
