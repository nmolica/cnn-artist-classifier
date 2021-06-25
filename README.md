# CNN Artist Classifier

This repo contains the code for a convolutional neural network to classify the musical artist of a provided song. Note that because of the amount of RAM required during training, this model cannot be run in Colab. Before running, ensure the machine you're using has the following dependencies:
- about 13Gb of RAM
- the matplotlib module
- the pillow module (PIL)
- the numpy module
- the tensorflow module
- the sklearn module

Depending on your machine, these should be easily obtainable with `pip install <module_name>` commands in an open shell.

## Important Notes

You shouldn't need to run any of the data processing scripts, as we've already run them and uploaded the processed data to this repo. Note that this data is ~300Mb, so it will take a little while to download when you clone the repo. You also shouldn't need to run the training script `model.py`, as we already did the training and saved a pretrained version in the trained_model directory. Use the `demo.py` script to test out the model. 