# AI Image Recognition Challenge - Chihuaha vs Muffin
created by Rashmi Pandit

## Basic Overview

I used Python for the whole project usng tenserflow with it. All the images used to create the datasets are downloaded from the internet.
The basic requirements in which the project was made were:
- HP Pavillion Gaming Laptop
- OS : Windows 11
- Processor: 8th Generation Intel(R) Core(TM) i5 plus 8300H
- System type: 64- bit operating system
- 8 GB SDRAM

### Install Python and the Prereuisites

To use python , one can download it from the below link :
https://www.python.org/downloads/release/python-3105/ 

After installing Python, we need to download some pre-requisites for the code to run.
1. install tenserflow
2. install numpy
3. install matplotlib or matplotlib.pyplot

For all the above, we can download them via pip. 
The tutorial for downloading via pip is below:
https://www.youngwonks.com/blog/How-to-install-Tensorflow-on-Windows
or
https://www.tensorflow.org/install/pip

Note : To run keras with GPU on Windows, you will need to configure a few settings 
(running this software on the CPU would be quite sluggish), and I recommend following this tutorial:
https://lifewithdata.com/2022/01/16/how-to-install-tensorflow-and-keras-with-gpu-support-on-windows/

## How does it works 
Download the files from Github and then follow the below steps:

Steps for Windows OS:
Start -> Python IDLE -> open -> search for the file which was downloaded from github -> click enter -> 
Run the program from the ribbon toolbar or from the Run menu from the Menubar.

At first , we need to run the file Categorizing_folders.py on the python IDLE.
In the Categorizing_folders.py , we are creating a new folder called chihuahua_vs_muffin.
It contains test, train and validation datasets. All the images from the datasets have also been renamed into 
their type and number of the image.For example :- muffin_401.png, chihuahua_34.png etc.

Detector.py : trains the CNN, saves the model in vgg19_chihuahua_vs_muffin.h5, and queries the model
Then, we need to run the Detector.py file for training and validating the program.

Note : The first time we run the Detector.py file it is going to take around 1 or 2 minutes.


### Querying the model

At the end, we run the Final.py file the same way fro testing our program.
After the program has run successfully, the following options will be displayed.

Choose Selector:
1. chihuahua 
2. muffin 
3. Exit 

Enter a valid instance id
1. Valid id's for chihuahua 1-900 (if 1 is entered)
2. Valid id's for muffin 1-500 (if 2 is entered)


Once, we have selected teh inputs we want, the program will output the result. It will display the probabality and predict the outcome of the selected picture, whether it is 
a chihuahua or a muffin. For example like below :
![image](https://user-images.githubusercontent.com/92164111/182050371-066dcabb-a2a7-4bf6-a7ae-cfd11d77036c.png)

For confirming whether the prediction was correct or not, it wil also display the image choosen.
![image](https://user-images.githubusercontent.com/92164111/182050432-207e5d6a-f572-430b-a0f4-c31e72386fcd.png)



