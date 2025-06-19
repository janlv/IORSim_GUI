## IORSim GUI and python script
IORSim is a reservoir simulator add-on that brings advanced geochemical IOR methods to existing Eclipse simulations.

## Running the compiled version
Compiled Windows and Linux versions of the python script are provided with every new release of IORSim_GUI. The compiled versions contain all the required packages, and there is no need to install additional code. Upon execution, the python packages are first extracted to a temporary folder. This leads to longer startup time when running the compiled versions compared to running the python script directly.     

For Linux, the downloaded file must first be made executable by entering the command `chmod u+x IORSim_GUI_ubuntu` in a Terminal window. 

## Files
At startup, a folder named `.iorsim` is created in the home-folder. The `.iorsim` folder contains settings-, session- and log-files.  

The compiled versions create a temporary folder named `_MEIxxxxxx`, where `xxxxxx` is a randum number. For Linux and Windows the temporary folder is located under `\tmp` and `C:\Users\<username>\AppData\Local\Temp`, respectively.

## Project Installation Guide

This project includes installation scripts that create a virtual environment and install all necessary dependencies.


### Installation (Linux/macOS)

1. Make the script executable:
```bash
chmod +x install.sh
```

2. Run the installer:
```bash
./install.sh
```


### Installation (Windows)

1. Open a terminal (e.g., Command Prompt or PowerShell)

2. Run:
```bat
install.bat
```


## Activating and deactivating the environment

### Linux/macOS:

To **activate** the environment:
```bash
source .venv_resplot/bin/activate
```

To **deactivate**, run:
```bash
deactivate
```


### Windows:

To **activate** the environment:
```bat
call .venv_resplot\Scripts\activate.bat
```

To **deactivate**, run:
```bat
deactivate
```
