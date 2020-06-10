# Experimental Analysis

##  Summary
The code in this directory analyzes the experimental data obtained from our online human experimentation. The main pipeline is in `Experimental_Analysis.ipynb`.

`./batch_data/` contains the data obtained for each experiment as a .json file in the format of the <a href="http://www.josephjaywilliams.com/mooclet">MOOClet framework</a>. The following notations are used for the naming of each file:
- Explanation type: `hm` for Saliency Maps, `proto` for Prototypes, `multi` for Combined Explanations
- Experiment type: `bl` for Random Selection policy (baseline), `exp` for Mental Model-Based policy

To fully run the analysis, all 3 files for each experiment (`xxx_data.json`, `xxx_learner_ids.json`, `xxx_actions.json`) must be present.

##  Setup
The following dependencies are used:
- <a href='https://numpy.org/'>NumPy 1.17.4</a>
- <a href='https://pandas.pydata.org/'>Pandas 0.25</a>
- <a href='https://matplotlib.org/'>Matplotlib 3.1.1</a>
- <a href='https://scipy.org'>SciPy 1.13</a>

To run `wallet.py` for managing and automating Amazon Mechanical Turk payments, import <a href='https://aws.amazon.com/sdk-for-python/'>AWS SDK for Python (Boto3)</a>. 

`private_variables.py` is empty by default. This is suitable for loading data directly from .json files. 

This file only needs the defined variables `url` and `key` when the user would like to directly retrieve data from a MOOClet server using REST API. The `Batch` class in `data_viz.py` contains methods for retrieving data directly from the MOOClet server (e.g., `Batch.get_data()`). `data_utils.py` contains lower-level functions for interacting with the MOOClet server using REST API.  

##  Run
The pipeline in `Experimental_Analysis.ipynb` will generate the results presented in the paper. Specifically, figures are saved in `./figures/` and tabular results are saved in `./report.txt`. Both figures and tabular results are also presented in the IPython console when `Experimental_Analysis.ipynb` is ran.

For further analysis of the data, use the methods available in the `Batch` and `Experiment` classes in `data_viz.py`.
