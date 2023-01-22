# Nail Metrics
## About
Nail_Metrics is an OpenCV based program created for experimental research in Material Science for the Mechanical Engineering Department at LSU.

## Installation
To install Nail_Metrics, simply download the files to the desired directory by clicking on "<> Code" in green near the top right, then "Download ZIP", unzip the files into the folder you want, and you will have the program along with the environment file. On windows, when you click "Extract All" and it asks you where to extract the files to, I recommend deleting the "MechE_Nail_Statistics-main" as it will save the folder with the files into another folder named that, so you will have a redundant folder.

### Dependencies
Nail_Metrics uses Anaconda to handle Python and it's dependencies, which are found in environment.yml . To setup the nm-py310 Python environment, install and configure Anaconda or Miniconda. I recommend Miniconda as it does not include many unnecessary packages. Read below for instructions on how to do this.
#### Conda Installation For Windows
Follow this tutorial to install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html , you can skip step 2. To find the .exe file in step 3, go to your "Downloads" folder. After step 4, to find/open the anaconda/miniconda prompt simply search up either anaconda or miniconda (depending on which you download) in your computer's search bar and click "anaconda Powershell Prompt (Anaconda)" and replace "anaconda" with "miniconda" if you downloaded miniconda. If you made it to step 5 you are done setting up conda. If you had installation issues check this website for troubleshooting: https://docs.anaconda.com/anaconda/user-guide/troubleshooting/ 

#### Conda Installation For Mac/Linux
Follow this tutorial to install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html , you can skip step 2. To find the .pkg file for anaconda, go to your "Downloads" folder. After step 5, to find/open conda, you can either search for anaconda or miniconda in your computer's search bar, or you can search terminal in your computer's search bar. If you had installation issues check this website for troubleshooting: https://docs.anaconda.com/anaconda/user-guide/troubleshooting/ 




After conda installation, first find the folder you downloaded this code and the environment.yaml file to, unless you changed the name it will be something similar to: "MechE_Nail_Statistics-main, once you have found that directory, double click into the folder where you will see main.py and environment.yaml. You will need to copy the path to that folder, to do this:

For windows follow this tutorial: https://www.howtogeek.com/670447/how-to-copy-the-full-path-of-a-file-on-windows-10/#:~:text=Here%27s%20how.,select%20%E2%80%9CCopy%20As%20Path.%E2%80%9D

For mac/linux follow this tutorial: https://themacbeginner.com/copy-full-path-file-folder-finder-mac-osx/

NOTE: The end of your path, should be the name of the folder. By default this should be "MechE_Nail_Statistics-main", if it is not, the next steps will not work. However, if you changed the folder name, it will need to end with whatever you changed it to. 

Now that you have the path, we can create the conda environment. To do this open up conda with the methods I provided in the conda installation section. Once conda is opened type in the following line (replacing ENTER_PATH_HERE with the path you have copied):

For windows:
  ```
  conda env create -f ENTER_PATH_HERE\environment.yaml
  ```
For mac/linux:
  ```
  conda env create -f ENTER_PATH_HERE/environment.yaml
  ```
  
  ## Usage
  Before using the program, you must activate the ```nm-py310``` environment. To do this enter:
  ```
  conda activate nm-py310
  ``` 
  You should see (nm-py310) as the first part of the line in the conda terminal. If you still see (base) then the environment was either created wrong, or you typed in the wrong command. Retry the steps above if that happens. 
  ### Running the Program
  When you're ready to run the program, first you will need to find the path to the image folder you wish to preform calculations on. To do this the steps are the same as above for finding a path name, except you are looking for the image folder. Once you have that found, you will enter the line below into your terminal, replacing ENTER_PATH_TO_PROGRAM_HERE with the path to the program, and ENTER_PATH_TO_IMAGES_HERE with the path to the image folder. Note that there are spaces after 'python' 'path to program' 'path to images' 'options'.
  ### Options
  The options entry allows for the user to pick what kind of output they want. Entering "csv" will output only a csv file of the nail metrics, "video" will output only a video of the found metrics, "all" will output both a video and csv file. Note you do not use ```""``` around the option you end up choosing, for example if we wanted the csv file only we would simply put: ```csv``` in the options entry NOT ```"csv"```.
  NOTE: If either of your paths have spaces in them, you will need to put '' around the path that has spaces, so a path on windows like ```C:\Users\Owner``` is fine but ```C:\Users\Owner\Nail Program```, will need to be entered as ```'C:\Users\Owner\Nail Program'```, same applies for mac/linux. 
  For windows:
  ```
  python ENTER_PATH_TO_PROGRAM_HERE\main.py ENTER_PATH_TO_IMAGES_HERE options
  ```
 EXAMPLE: ```
 python C:\Users\Owner\Downloads\MechE_Nail_Statistics-main\main.py C:\Users\Owner\Images all
 ```
 For mac/linux:
 ```
 python ENTER_PATH_TO_PROGRAM_HERE/main.py ENTER_PATH_TO_IMAGES_HERE options
 ```
 Example: ```
 python '/Users/Jerry's Computer/MechE_Nail_Statistics-main/main.py' /Users/Jerry/Images csv
 ```
 NOTE: Note in the last example, the usage of ```' '``` because of the space between ```Jerry's``` and ```Computer```
 ## Output
 



