# Iris Species Classification

This workspace contains an end-to-end Iris classification notebook that:

- performs exploratory data analysis
- visualizes class separability
- compares k-NN, Logistic Regression, and Decision Tree classifiers
- reports accuracy, precision, recall, and confusion matrices
- saves the best trained model as `iris_best_model.joblib`

## Files

- `Iris.ipynb` - notebook with the full analysis and training workflow
- `iris_best_model.joblib` - saved best model pipeline
- `IRIS.csv` - dataset used by the notebook

## Run the notebook

Open `Iris.ipynb` and run the cells from top to bottom. The notebook will:

1. load the dataset
2. generate EDA plots
3. train and compare the classifiers
4. save the best model to `iris_best_model.joblib`

## Example inference

```python
import joblib
import pandas as pd

model = joblib.load("iris_best_model.joblib")

sample = pd.DataFrame(
    [[5.8, 2.7, 5.1, 1.9]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)

prediction = model.predict(sample)[0]
print(prediction)
```

## Observed results

On the notebook's held-out test split, the best model was k-NN with:

- accuracy: 0.9333
- precision: 0.9444
- recall: 0.9333
