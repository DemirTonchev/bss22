# Bachkovski Manastir 22
## Setup

For this workshop you will need [Anaconda or miniconda](https://docs.conda.io/en/latest/miniconda.html) setup and installed on your system. If you do not, please install Anaconda on your system before proceding with the setup of the needed environment.

The next step is to clone this repository. Use git to run this command:

    git clone https://github.com/DemirTonchev/bss22.git
    

***
The repository for this workshop contains a file called `environment.yml` that includes a list of all the packages used for the tutorial. If you run:

    conda env create -f environment.yml
    
from the main workshop directory, it will create the environment for you and install all of the needed packages. This environment can be enabled using:

    conda activate bss22
    
Then, you can start **JupyterLab** to access the materials:

    jupyter lab
