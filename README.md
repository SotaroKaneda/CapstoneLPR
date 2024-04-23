# CapstoneLPR

### Copy the Repository

```
git clone https://github.com/SotaroKaneda/CapstoneLPR.git
```

### Run the Model

1. Create a file named `twos_comp_part1.c` and implement the following in the file.

2. Perform the following operations
   * Read the input argument integer to be converted. You should pass this value as a command-line argument, in decimal format, upon execution. (This integer can be negative, positive or 0)
   
     order: **< number in decimal >**
     
   * Convert the given integer and print the 8-bit binary representation of the string
     - **NOTE:** Generally integers are 32-bit values; but for simplicity we are converting the given integer to an 8-bit binary number.
   * You have to check if the integer passed in as an argument is within the range of values that can be accurately shown in an 8-bit **signed** number.

3. Compile

```
$ gcc -Werror twos_comp_part1.c -o tc1 -std=c99 -pedantic -Wall -Wextra
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
