# California Housing Value Prediction

**Junior Jele | Guided Python and machine learning project**

Estimate the recorded median house value of a California census area from eight
numerical features. This project compares a mean baseline with linear regression,
a decision tree, and a random forest, then examines the selected model's errors.

## Data and prediction target

The scikit-learn California Housing dataset contains 20,640 census block groups
from the 1990 US census. Each row represents an area, not an individual home.
The eight inputs summarize income, house age, rooms, bedrooms, population,
occupancy, latitude, and longitude. `MedHouseVal` is the target, in $100,000 units.
Results describe historical, capped recorded values, not current market prices.

Source and dataset description:
[scikit-learn California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

## Evaluation design

- Reserve 20% (4,128 areas) for final testing using `random_state=42`.
- Explore the remaining 16,512 training areas; no missing entries were found.
- Compare models on the same five shuffled KFold splits with `random_state=42`.
- Keep scaling inside the linear regression pipeline so it is learned within each fold.
- Select the 100-tree random forest before computing final test metrics.
- Fit on all training areas, evaluate the reserved test set, and inspect errors
  without changing the selected model afterward.

A brief schema preview preceded the split. The test set was otherwise reserved
for final evaluation. No hyperparameter search was performed.

## Results

Lower mean absolute error (MAE) is better. Dollars refer to area median values.

| Model | Mean five-fold CV MAE |
|---|---:|
| Mean-prediction baseline | $91,391 |
| Linear regression | $52,906 |
| Decision tree, max depth 5 | $52,658 |
| Random forest, 100 trees | **$33,504** |

The forest had the lowest error in all five folds. Its final test results were:

- **MAE: $32,754**
- **RMSE: $50,534**

MAE averages absolute errors. RMSE gives large mistakes greater weight.
These are error measures, not accuracy percentages or guarantees for any area.
The test MAE was close to the CV estimate.

Training impurity importance assigned the largest contribution to median income,
followed by average occupancy. The latter had weak individual linear correlation
with the target, illustrating that marginal correlation does not capture every
relationship used by a nonlinear model.

## Run on Windows with Anaconda

Extract the project and open Anaconda Prompt in this folder. For a new environment:

```bash
conda create -n housing-project python=3.11 -y
conda activate housing-project
python -m pip install -r requirements.txt
jupyter notebook
```

Open `01_housing_regression.ipynb` and run all cells from top to bottom.
You can also use the existing `titanic` environment if it already has these libraries.
The dataset downloads automatically on the first run and is cached by scikit-learn.
That first download requires internet access. No manual CSV or `utils.py` is needed.
Cross-validation trains 500 forest trees, plus 100 in the final fit, so allow time.
Numerical results were checked with scikit-learn 1.5.0 and NumPy 1.26.4;
small platform or version differences are possible.

## Files

- `01_housing_regression.ipynb`: checked notebook with saved charts and results.
- `requirements.txt`: dependencies.
- `LICENSE`: MIT license for this project's code; it does not relicense the dataset.

## Limitations and next questions

- The data are historical and the target is capped at approximately $500,000
  (the stored upper value is $500,001).
- Random splits can put neighboring areas in both sets. Generalization to entirely
  new geographic regions would require geographic validation.
- Extreme feature values were retained; their effects were not separately measured.
- Training impurity importance can be biased and shared among correlated features;
  it does not establish causation or the direction of an effect.
- The forest uses unrestricted-depth trees, unlike the depth-5 single tree, so the
  comparison does not isolate averaging alone.
- Only a few configurations were compared. The mean baseline is a simple benchmark;
  a median baseline could be added because it minimizes constant absolute error.
- Cross-validation was used for selection, not nested evaluation. The training data
  were explored before CV, so the folds do not independently validate the entire
  exploratory decision process.
- MAE does not describe the uncertainty of an individual prediction. No prediction
  intervals were constructed.

## Learning and authorship

Developed by Junior Jele through guided practice, with AI-assisted explanations,
code guidance, notebook review, and documentation. The focus is understanding
regression, safe validation, pipeline construction, and honest error reporting.
This is a separate project from the earlier Titanic classification rebuild.
