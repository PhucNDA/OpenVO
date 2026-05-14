
 # Installation
 To ensure everything running stable, we have 2 environments: One for data preprocessing and One for training + inference.
 
 Cloning OpenVO
```bash
git clone https://github.com/PhucNDA/OpenVO
```

 ## 1. Training + Inference OpenVO

```bash
conda create -n openvo python=3.9
conda activate openvo
# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c iopath iopath
# install pytorch3d
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt201.tar.bz2
conda install pytorch3d-0.7.5-py39_cu117_pyt201.tar.bz2
rm pytorch3d-0.7.5-py39_cu117_pyt201.tar.bz2
```

Download CMake at : https://cmake.org/download/

Adjust the prefix to local install
```bash
chmod +x cmake-4.1.2-linux-x86_64.sh
./cmake-4.1.2-linux-x86_64.sh --prefix=cmake
```

NVCC Cuda if you do not have apt-get
```bash
conda install -c nvidia/label/cuda-11.7.1 cuda
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Correlation CUDA (sensitive)
```bash
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 ninja
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
cd ./OpenVO/src/model/correlation_package
pip install -e . --no-build-isolation
```

The rest dependencies
```bash
pip install fvcore==0.1.5.post20221221 PyYAML==6.0.2 timm==1.0.16 matplotlib==3.5.3 pandas==2.3.0 opencv-python==4.11.0.86 a-unet==0.0.16
pip install -U pip setuptools wheel
pip install -U openmim
mim install "mmcv-full==1.7.2"
# mim install mmcv-full==1.7.0 (This would promnpt error)
pip install numpy==1.26.4 pillow==11.0.0 av2==0.2.1 nuscenes-devkit==1.1.11
python -m pip install pyviz3d
```
Manually copy/move the file wrt to **your** workspace dir {openvo} <-- your env name
```bash
# from root
cd ../../../
cp -r ./vision_transformer_cross.py  miniconda3/envs/{openvo}/lib/python3.9/site-packages/timm/models/

```

## 2. Preprocessing Metric3Dv2 + WildCamera

```bash
# from root
cd ./OpenVO/processor

conda create -n processor python=3.9
conda activate processor
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Refer to the above CMake & NVCC if needed!

Dependencies
```bash
pip install opencv-python numpy==1.23.1 xformers==0.0.21 Pillow DateTime matplotlib plyfile timm tensorboardX imgaug iopath imagecorruptions loguru einops natsort h5py tabulate
pip install -U pip setuptools wheel
pip install -U openmim
mim install "mmcv-full==1.7.2"
```

Build Setup Source
```bash
# For Tame a Wild Camera
cd WildCamera
pip install -e .

cd ../
# For Metric3D
cd Metric3D
pip install -e .
```

Your environments are full set!

## 3. Checkpoint downloading

Download the pre-processor checkpoints and our OpenVO checkpoint at [GGDrive](https://drive.google.com/drive/folders/1emJE1prn0rwZwbmRpFcx_J5r2kYQSQGG?usp=sharing):

Store them in the **weights** folder, inside OpenVO directory.
```
OpenVO
│   weights (GGDrive)
│    │    ├── init_weights          <--(flow weight)
│    │    ├── Metric3D
│    │    │    ├── metric_depth_vit_large_800k.pth
│    │    │    ├── metric_depth_vit_small_800k.pth
│    │    ├── WildCamera
│    │    │    ├── wild_camera_all.pth
│    │    │    ├── wild_camera_gsv.pth
│    │    ├── OpenVO/openvo_nusc_gt_combine123_gradclip_nodiff
│    │    │    ├── logs
│    │    │    ├── model_ep-022.pt
```

