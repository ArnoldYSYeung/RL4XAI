# Convolutional Neural Networks with Explainers

## Summary
The code in this directory builds the Convolutional Neural Network for classifying the <a href="https://github.com/rois-codh/kmnist">Kuzushiji-49 dataset</a> and generates explanations using the following explainers:
- LIME
- Prototypes
- Deep Taylor Saliency Maps

The main pipeline is in `CNN_with_Explainers.ipynb`.

## Setup
The following dependencies are used:
- <a href="https://www.tensorflow.org/">Tensorflow 1.15 </a> (is not compatible with 2.0)
- <a href="https://keras.io/">Keras 2.3</a>
- <a href="https://github.com/albermax/innvestigate">iNNvestigate 1.0.8.3</a>
- <a href="https://github.com/IBM/AIX360/">AIX360 0.1.0</a>
- <a href="https://github.com/marcotcr/lime">LIME 0.1.1.36</a>
- <a href="https://pypi.org/project/Pillow/">Pillow 6.2.1</a>

To run the code, download classes 0 (a) and 33 (me) from the Kuzushiji-49 dataset in `../data/` (e.g., `../data/kuzushiji-49/33/train/` and `../data/kuzushiji-49/33/test/`). You may change the locations of the data manually within `CNN_with_Explainers.ipynb`.

## Run

The pipeline in `CNN_with_Explainers.ipynb` will save a .h5 copy of the trained model in `./report/` (location may be changed manually), along with results of the training.

Explanations generated are found in `./output/` (location may be changed manually). 
