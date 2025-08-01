# Personality Prediction Project

This was super fun ... I acheived a rank of **347/4329** on Kaggle Playground Series Season 5 Episode 7


This repository predicts whether a person is an **Introvert** or an **Extrovert** from social/behavioral data.  
It uses advanced data imputation, preprocessing, and multiple ensemble machine learning models for high predictive performance.

---

2 datasets were provided , the official train.csv dataset as well as an external dataset called personality_datasert.csv 
final merged dataset is called final .csv

## File Descriptions

| File                | Purpose                                                                                                    |
|---------------------|-----------------------------------------------------------------------------------------------------------|
| preprep.py          | Cleans and imputes the training dataset (`final.csv`), saves as `final_cleaned_data.csv`.                 |
| model.py            | Trains ensemble classifiers, saves the final ensemble model as `final_majority_ensemble.pkl`.              |
| test.py             | Processes `test.csv`, applies the same imputations, loads the ensemble model, and generates predictions.   |
| final.csv           | Raw full training data input.                                                                              |
| final_cleaned_data.csv | Output after data cleaning/imputation.                                                                 |
| final_majority_ensemble.pkl | Saved trained voting ensemble model.                                                              |
| test.csv            | Test data (must have `id` column).                                                                        |
| allmodelsmix.csv    | Submission file: `id`, `Personality` columns.                                                             |
| requirements.txt    | Python package dependencies.                                                                              |
| README.md           | This documentation.                                                                                       |

---

## Usage Instructions

1. **Data Preparation**

   Place your raw, merged data as `final.csv` in the project directory.

python preprep.py

text

Output: `final_cleaned_data.csv`

2. **Model Training**

python model.py

text

Output: `final_majority_ensemble.pkl`

3. **Prediction on Test Set**

Make sure `test.csv` is available in the directory (with an `id` column).

python test.py

text

Output: `allmodelsmix.csv`

---

## Requirements

All requirements are listed in `requirements.txt`.  
Install them with:

pip install -r requirements.txt

text

**Note:**  
- Python 3.8 or higher is recommended.  
- CatBoost, LightGBM, and XGBoost may require additional system dependencies.
- Results may vary depending on data content and structure in `final.csv` and `test.csv`.

---

## Contact / Issues

If you encounter any problems or need sample data formats, please open an issue or contact the repository maintainer.

requirements.txt
text
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
joblib
