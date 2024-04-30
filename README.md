# CapstoneLPR

### Copy the Repository


```
git clone https://github.com/SotaroKaneda/CapstoneLPR.git  
```
### Install Dependencies  
Tested with Python>=3.8.0, and PyTorch>=1.8
```
cd CapstoneLPR
pip install -r requirements.txt
```

### Run the Model

1. Place weights folder containing the model weights in the CaptsoneLPR folder.

2. Run main.py with a Path to an image or folder of images

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


**Make sure that the `sicecse` user has been added as a collaborator to your
 repository so we can access and grade your assignment.**

**Late submission penalties are 5% per day after the due date upto a maximum of 
one week. After that week your submission would NOT be accepted.**
