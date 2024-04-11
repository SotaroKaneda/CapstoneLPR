# CapstoneLPR

### Copy the Repository

$ git clone https://github.com/SotaroKaneda/CapstoneLPR.git
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

4. Example Executions
   
Converting "50" to the 8-bit binary value

```
$ ./tc1 "50"
> 00110010
```
   
Converting "-67" to the 8-bit binary value

```
$ ./tc1 "-67"
> 10111101
```

### Train the model

1. Create a file named `twos_comp_part2.c` and implement the following in the file.

2. Perform the following operations
   * Read in the _8-bit binary string representation_ and _type of integer_ to be converted
   * Based on the type of integer, convert the given binary string representation and print the decimal value
     - **NOTE:** Generally integers are 32-bit values; but for simplicity, we are converting an 8-bit binary number to the given integer
   * You have to check if the type passed in as an argument is a valid type of integer (i.e. _signed_ or _unsigned_)

3. You should pass the command-line as follows in the given order
   
   order: **< 8-bit binary string representation >** **< integer type >**
   
4. Important Tips

   * In an 8-bit _unsigned_ binary string, the range of representable integers is [0, 255].
   * In an 8-bit _signed_ binary string, the range of representable integers is [-128, 127]. Here the Most Significant Bit (MSB), which is also the sign bit, is used to represent the magnitude of number. So we can represent -128 using 8 bits.
   * The Autograder will only test and grade a limited set of numbers. Make sure to test your code for numbers in the ranges specified above. 

5. Compile

```
$ gcc -Werror twos_comp_part2.c -o tc2 -std=c99 -pedantic -Wall -Wextra
```
6. Example Executions

Converting "11001100" when the integer type is "signed"

```
$ ./tc2 "11001100" "signed"
> -52
```

Converting "11001100" when the integer type is "unsigned"

```
$ ./tc2 "11001100" "unsigned"
> 204
```


**Make sure that the `sicecse` user has been added as a collaborator to your
 repository so we can access and grade your assignment.**

**Late submission penalties are 5% per day after the due date upto a maximum of 
one week. After that week your submission would NOT be accepted.**
