# Model Card

## Model Details
A random forest classifier using the hyperparameters with `scikit-learn 0.24.2`: 

  - n_estimators: 100
  - criterion: entropy
  - max_depth: 7
  - max_features: 10

## Intended Use
This model should be used to predict the salary category of user demographic data. Users could be researchers, goverment, or hobbyists.

## Training Data
The data is the [Census Bureau data obtained from Udacity](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv). The original data had 32,561 rows and a 80-20 split was used to break this into a train-test set. No stratification was done.

Categorical features were set as:
- `workclass`
- `education`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

The remaining columns were set as continuous features. The target class was `salary`, which the value is `<=50k` and `>50k`. A one hot encoder was used on the features and a label binarizer on the `salary` target label with no normlaization or standardization for numerical feature.

## Evaluation Data
The evaluation data as mentioned above, was split with a proportion of 20% from the cleaned data. The overall model metrics computed is based off this test data.

The split was made with a random seed of `42` using data from `clean_census.csv`.

## Metrics
The model was evaluated using the precision. Its overall precision obtained by the model is ~80%, with a recall of 56% and fbeta score of 66% (see [scores.json](https://raw.githubusercontent.com/mohrosidi/mlops_census/master/logs/scores.json)). 

## Ethical Considerations
Given the census data contains sensitive, and highly charged information like education levels, gender and race, the predicted output of this model should not be taken with a high degree of confidence without further testing for bias.

## Caveats and Recommendations
The distribution of the census data is skewed heavily towards certain categorical features. There is significant under-representation of some races, workclass, education levels, marital status and native country in the data (see [slice_output.txt](https://github.com/mohrosidi/mlops_census/blob/master/logs/slice_output.txt)).

This model was not extensively tested and a hyperparameter search was not performed, it is recommended that further work to first increase the overall model performance be done such as performing feature transformation in numerical features, and then use a better dataset that is more representative of a distribution of the overall US population.


