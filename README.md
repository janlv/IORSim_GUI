## IORSim GUI and python script
IORSim is a reservoir simulator add-on that brings advanced geochemical IOR methods to existing 
Eclipse simulations.

## Running the compiled version
A compiled Windows and Linux version of the python script is provided with every new release of IORSim_GUI. The compiled version contains all the required packages and there is no need to install additional code. At startup, the python packages are extracted to a temporary folder before the script is executed. This leads to longer startup time for the compiled version. For Windows 10, the temporary folder is `C:\Users\<username>\AppData\Local\Temp\_MEI<number>` where `<number>` is a random number.

## Running the python script
The python interpreter must be version 3.8 or newer. Install the required libraries by executing 

`pip install -r requirements.txt`

If python is properly installed in your path, IORSim_GUI.py can simply be invoked by

`./IORSim_GUI.py`

In case this fails, you need to specify the python interpreter

`python IORSim_GUI.py` 

## Files
At startup, a folder named `.iorsim` is created in the home-folder. The `.iorsim` folder contains settings-, session- and log-files.  



