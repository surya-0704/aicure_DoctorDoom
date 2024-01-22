
# AI_Cure:

To make train an AI model to accurately predict heart rate using prexistying data.


## Authors

- [@surya](https://github.com/surya-0704)
- [@utkarsh](https://github.com/Utkarshgupta56)
- [@ramank](https://github.com/profresher149)
- [@shubham](https://github.com/freeradical077)




## Acknowledgements

 - [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
 - [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)


## Usage

The git consists of an ipynb file named [aicure.ipynb](https://github.com/surya-0704/aicure_DoctorDoom/blob/main/aicure.ipynb) that consists of the data research and trials.

It also includes [run.py](https://github.com/surya-0704/aicure_DoctorDoom/blob/main/run.py) file to train the model and predict results stored as 'results.csv' :

```
python run.py test_data_file.csv
```
There also exists a [check_mse.py](https://github.com/surya-0704/aicure_DoctorDoom/blob/main/check_mse.py) file used to check the mean squared error between the predicted data and the actual data (Run this script only after running the [run.py](https://github.com/surya-0704/aicure_DoctorDoom/blob/main/run.py) script) :

```
python check_mse.py output_file_name.csv
```


## Libraries

Pre-Install the following libraries before running the script:

```{python}
pip install pandas
pip install csv
pip install scikit-learn
pip install statistics
```
