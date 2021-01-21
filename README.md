Thank you for downloading the Transformers AI package to predict house pricing and time to sale.

To run this package please follow the instructions below:

1. Create a conda environment with python version 3.8.5 with the following command
    - conda create --name myenvironment python==3.8.5
2. Activate the environment with the following command
    - conda activate myenvironment
3. Clone this repository to your machine
4. Inside the repository create a folder called "data" and in that folder create a folder called "raw"
5. Move all raw data to data/raw
    - download link for raw data: XXXXXXX
6. Change the path of the raw data to your path
    - in conf.json adjust the base_folder path to your path
    - make sure it ends with "data/"
        - OPTIONAL
        - you can change the parameters of all models in the conf.json as well. You can adjust/add/remove parameter values per model.
    - make sure to save the changes
7. Open terminal and move to the repository with the following commands
    - ls (list all directories in your current location)
    - cd (change to the directory that you specify)
8. From inside the product3team1 directory run the following command and wait for it to install all packages
    - pip install -e .
9. Move to the run folder with the following command
    - cd run
10. Run the pipeline with the following command and enjoy the output
    - python run.py

We sincerly hope you enjoy the package and if you have any questions please feel free to reach out to us at: niels.jonker@hva.nl



