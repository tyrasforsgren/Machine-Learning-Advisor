# testing_prev_ml_exam
### Update 1.01

The algorithm is used to calculate an ideal Machine Learning model suiting a dataset. The user inputs a dataset in the form of a .csv as well as a label for it to predict. 

The program, after visualizing some general data, compares and shows the results (MAE, RMSE, r2_score) for each type of model tried. The user gets a suggestion for ideal model based on these results and if agreed upon by the user, the program outputs a the trained final model.

## Requirements:
See requirements.txt (generated)

## Changelog 1.01:
-   Creating test class
-   Improved system layout
    -   Introduced GitHub
    -   Added README.md
    -   Added requirements.txt
    -   Added respective doc and scripts folders
-   Improved documentation formatting according to PEP8
-   Restructured initialization process for ModelSelectFinal.ModelSelection (SEE: demo.py or documentation)
-   Documented and clarified the demonstation file (demo.py)
-   ModelSelectFinal.ModelSelection.data_report() has been removed from the automatic algorithm. WIP to make it @static


(Note to Ammar: I had to prioritize, I could've looked for inprovements indefinetly: )

## BUGTRACKER:
- Error upon choosing classifiation for continous data due to the algorithm being unable to keep seperate a </u>classification file with digitalized target column</u> and a </u>file with continous target column</u>. It only flags for classification-type files with dtype == 'object' in target column. Reconsider potential input combinations resulting in errors, and the users ability to change options.

## TODO:
-   Fix test methods (test_ModelSelectFinal) that deal with exiting the program. They wont recognize it. (Look more deeply into raising errors and their general practice) 
-   Incorporate error raises instead of .exit

-   Move imports to top (ModelSelectFinal) - tests stopped working?
-   Put interfaces within a loop - dont end program
-   add parameters like 'degree' etc from polynomial regression to the method params (with defaults) to static methods for further control.
-   make .get_metric - type methods static
-   make .set_values static
-   make .data_report static
-   make .preproccess static
-   Change name of ModelSelection.classification_reports - confusing and too similar to imported method 'classification_report'
-   Reorder ModelSelectFinal.ModelSelection class methods for clarity
