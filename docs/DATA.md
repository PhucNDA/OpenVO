# Data Preparation

We plan to release the precomputed set for community reproducibility

## Preprocess nuScenes
Download [nuScenes](https://www.nuscenes.org/nuscenes) dataset.

We plan to release the precomputed nuScenes.

1. Run the following source code to extract the dataset by modifying your directory paths
```bash
python data/preprocess_nuscene.py
```
* ``` L153 <datapath>```: your absolute path of OpenVO from root.
* ``` L154 <root>```: your absolute path to the downloaded dataset from root.

2. Run the preprocess source code ```processor/depth_processor.py``` for estimating the intrinsics (WildCamera) and the depth images (Metric3Dv2)
```bash
conda activate processor
python processor/depth_processor.py
```

* ``` L299-301 <input_size>```: please try different depth input image sizes to see which one fits the best for your dataset (high-quality).
* ``` L358 <datapath>```: your data path you wanted to extract
* ```L361```: Use start_id, end_id if you have multiple threads to save some time

The nuScenes directory tree
```
OpenVO
├── data
│    ├── nuScenes/NUSCv1.0-trainval_12hz/CAM_FRONT
│    │    ├── sequences                                      <- RGB images
│    │    |    ├── 000
│    │    |    │    ├──image_2                                
│    │    │    │    │    000000.png
│    │    │    │    │    ...
│    │    |    │    ├──image_3                                
│    │    │    │    │    000000.png
│    │    │    │    │    ...                         
│    │    |    ├── 001
│    │    |    ....
│    │    ├── depth_est_intrs/sequences                      <- Estimated depth images
│    │    |    ├── 01
│    │    |    │    ├──image_2                               <- We use front view only
│    │    │    │    │    000000.png
│    │    │    │    │    ...                            
│    │    |    ├── 02
│    │    |    ....
│    │    ├── poses 
│    │    |    ├── 01.txt
│    │    |    ├── 02.txt
│    │    |    ....
│    │    ├── nuscene_est_intrs.json                         <- Estimated intrinsics 
####################################################################################
```

## Preprocess KITTI
We provide the downloadable link [KITTI](https://huggingface.co/datasets/PhucDucAnhNguyen/OpenVO_KITTI/) set here!

```
OpenVO
├── data
│    ├── KITTI
│    │    ├── sequences                                      <- RGB images
│    │    |    ├── 01
│    │    |    │    ├──image_2                                
│    │    │    │    │    000000.png
│    │    │    │    │    ...
│    │    |    │    ├──image_3                                
│    │    │    │    │    000000.png
│    │    │    │    │    ...
│    │    |    │    ├──calib.txt                                  
│    │    |    │    ├──times.txt                             
│    │    |    ├── 02
│    │    |    ....
│    │    ├── depth_est_intrs/sequences                      <- Estimated depth images
│    │    |    ├── 01
│    │    |    │    ├──image_2                               <- We use front view only
│    │    │    │    │    000000.png
│    │    │    │    │    ...                            
│    │    |    ├── 02
│    │    |    ....
│    │    ├── poses 
│    │    |    ├── 01.txt
│    │    |    ├── 02.txt
│    │    |    ....
│    │    ├── kitt_gt_intrs.json                            <- GT intrinsics (run get_true_intrinsic.py) [Optional]
│    │    ├── kitt_est_intrs.json                           <- Estimated intrinsics 
│    │    ├── get_true_intrinsic.py                  
####################################################################################
```

## Preprocess Argoverse2

Would be the same as nuScenes, we provide a draft script. Please edit correspondingly!

```bash
python data/preprocess_av2.py
```
Run preprocessing source code  ```processor/depth_processor.py``` as nuScenes

## Process YTB data / Your Video
Decompose a video into multiple consecutive frames using this script:

```bash
python data/process_ytb.py
```
Run preprocessing source code  ```processor/depth_processor.py``` as nuScenes

## Custom dataset pose training

The training pose labels for the three benchmark datasets are available in the ```poses``` directory. To regenerate the labels or adapt to custom datasets, please refer to the code and execute the following command:
```bash
python prepare_trainpose.py
```