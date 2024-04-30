# CapstoneLPR

### Copy the Repository


```
git clone https://github.com/SotaroKaneda/CapstoneLPR.git  
```
### Install Dependencies  
*Tested with Python>=3.8.0, and PyTorch>=1.8*  
```
cd CapstoneLPR
pip install -r requirements.txt
```
##### Install the appropriate torch and torchvision libraries for your system.  
Install Guide: https://pytorch.org/get-started/locally/  
Commands Below:  
**Linux**  
*CUDA 11.8*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*CUDA 12.1*  
```
pip3 install torch torchvision
```
*CPU*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Windows**  
*CUDA 11.8*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*CUDA 12.1*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*CPU*  
```
pip3 install torch torchvision torchaudio
```
**Mac**  
```
pip3 install torch torchvision
```
### Run the Model

1. Place the weights folder containing the model weights in the CaptsoneLPR folder.

2. Run main.py with a path to an image or folder of images

```
python main.py "image/or/folder/path"
```

### Train the model on BigRed

1. Load libraries
   
```
module load python/gpu/3.11.5
pip3 install transformers
pip3 install datasets
pip3 install jiwer
```

2. Run on 1 GPU node (4 GPUS)
```
srun -p gpu --gpus-per-node 1 --nodes 1 -A c00533 python test.py
```
