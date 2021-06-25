# CNN Artist Classifier

This repo contains the code for a convolutional neural network to classify the musical artist of a provided song. Note that because of the amount of RAM required during training, this model cannot be run in Colab. Before running, ensure the machine you're using has the following dependencies:
- about 13Gb of RAM
- the matplotlib module
- the pillow module (PIL)
- the numpy module
- the tensorflow module
- the sklearn module

Depending on your machine, these should be easily obtainable with `pip install <module_name>` commands in an open shell. Some of them may take some time to download.

## Important Notes

You shouldn't need to run any of the data processing scripts, as we've already run them and uploaded the processed data to this repo. Note that this data is ~300Mb, so it will take a little while to download when you clone the repo. The training data can be found in `bw_train` and the test data is in `bw_test`.

## Training the model
To train the model, it will take approximately 30 minutes for the 25 epochs, depending on your machine specs. In your shell, run `python3 model.py`. When prompted, input `bw_train` for the training data directory and `bw_test` for the testing data directory. You will be prompted for both of these. The terminal will show the progress of the trainings and output the accuracy of the test data at the end. Please contact Nicholas Molica (molica.n@northeastern.edu) and Joshua Kwok (kwok.jo@northeastern.edu) if you run into any problems while trying to run the code.
