# testing_prev_ml_exam
### Update 1.01

The algorithm is used to calculate an ideal Machine Learning model suiting a dataset. The user inputs a dataset in the form of a .csv as well as a label for it to predict. 

The program, after visualizing some general data,  compares and shows the results (MAE, RMSE, r2_score) for each type of model tried. The user gets a suggestion for ideal model based on these results and if agreed upon by the user, the program outputs a the trained final model.

## Requirements:
See requirements.txt (generated)

## Bugs:
-   Algorithm cannot identify digitalized files that should be used for classification. Only flag for classification-type files is occurence of dtype == 'object' in dependant label column.

## Updates 1.01: ONGOING
-   Creating test-classes
-   Improved system layout
    -   Introduced GitHub
    -   Added README.md
    -   Added requirements.txt
    -   Added respective doc and scripts folders
-   Improved documentation formatting according to PEP8

## TODO: - outside of bugfixing
-   make .get_metric - type methods static
-   make .set_values static
-   make data_report static
-   make .preproccess static
-   Change name of ModelSelection.classification_reports - confusing and too similar to imported method 'classification_report'
-   Reorder class methods for clarity
-   Put interface in complete_data within a loop - dont end program
-   Incorporate raises instead of exit
-   add parameters like 'degree' from polynomial regression to the method params (with defaults) to static methods for further control.
-   train on entire dataset before saving