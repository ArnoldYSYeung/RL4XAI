# RL4XAI-NeurIPS
This repository contains the files and data for RL4XAI.

There are two main directories:
- `./src/` contains files for building the Convolutional Neural Network and for generating the explanations from 3 different explainers
- `./analysis/` contains the data obtained from our online experimentation and files for analysing the experimental data

Because the online experimentation was conducted on <a href="https://www.qualtrics.com/">Qualtrics</a> (a private drag-and-drop platform for designing online surveys), source code is unavailable for the RL framework and the experimentation.

Anonymized data from the online human experimentation is stored in `./analysis/batch_data/`. Explanations available for the experimentation may be found in `./src/output/` and the trained CNN model used for the experimentation is stored as a .h5 file in `./src/model/`.

To rerun the model training process or re-generate explanations, download the <a href="https://github.com/rois-codh/kmnist">Kuzushiji-49 dataset</a>. (See `./src/README.md` and `./src/CNN_with_Explainers.ipynb` for more detailed instructions.)

Dependencies for the pipeline in each directory are listed in their respective `README.md` files.

