
"""
Test Case for ModelSelectFinal

This script contains unit tests for the ModelSelection class defined in the
ModelSelectFinal module. The ModelSelection class is responsible for
selection and evaluation for both regression and classification models based
on given data. This unittest module defines and runs various test scenarios
to ensure the correctness ModelSelection class methods.

Ensure that you have set up the necessary testing environment and data
before running this script. Test file: 'project\\docs\\Advertising.

Test Methods
------------
Data handling:
- test_set_values: Tests specification of dependent and independent variables.
- test_load_data_success: Tests the successful loading of data.
- test_load_data_fail: Tests failure of data loading.
- test_complete_data_drop_incomplete: Tests dropping incomplete data.
- test_complete_data_incomplete_exit: FIXME tests exit at denial to drop.
- test_complete_data_digitalize: Tests digitalization of categorical data.
- test_complete_data_no_digitalization_when_needed: FIXME test exit at denial
to digitalize.
- test_preprocess_regression: Tests data preprocessing for regression.
- test_preprocess_classification: Test data preprocessing for classification.
- test_save_model_invalid_valid: Tests saving a model with invalid and
valid filenames.

Main Interfaces:
- test_initialize_model_reg: Tests the initialization of algrorithm for
regression cases.
- test_initialize_model_class: Tests the initialization of algorithm for
classsification cases.

- test_choose_model_invalid_then_regressor: Tests invalid selection and
selection of regression model.
- test_choose_model_classifier: Tests the selection of a classification model.
- test_choose_model_r_to_c_conversion: Tests conversion i regression is
wrongly chosen.
- test_choose_model_conversion_denied: Tests exit of conversion is denied.

- test_confirm_model_choice_invalid_input_then_valid: Tests invalid and
valid input.
- test_confirm_model_choice_exit: Tests model choice to exit.

Metrics:
- test_get_mae: Tests the calculation of Mean Absolute Error.
- test_get_rmse: Tests the calculation of Root Mean Squared Error.
- test_get_r2_score: Tests the calculation of R-squared score.

Model Creation and Suggestion:
- test_grid_model: Tests hyperparameter tuning.
- test_calc_ideal_regression_model: Tests the creation and calculation
of the ideal regression model.
- test_calc_ideal_classification_model: Tests the creation and calculation
of the ideal classification model.

Reports:
- test_regression_report: Tests the generation of regression reports.
- test_classification_reports: Tests the generation of classification reports
without plots.
- test_classification_reports_show_plots: Tests classification reports whith
plots.
- test_data_report_plot_regression: Tests data plotting
for regression tasks.
- test_data_report_plot_classification: Tests data plotting
for classification tasks.

Usage:
Before running this script, ensure that you have set up the required testing
environment and data files. You can run this script to perform unit tests
on the ModelSelection class.

Notes:
There are no parameters for any test method because it is a test class
(and 'with'-statements are used instead of decorators when mocking.)
No test method has returns because this is a test class.

Lack of parameters and returns are not explicitly stated in each method
documentation under the assumption that readers understand
this concept and that suggestions of this in method heads are enough.

"""

import unittest
from unittest.mock import patch, Mock, call

import sys
from io import StringIO
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV


from ModelSelectFinal import ModelSelection  # Local


class TestModelSelection(unittest.TestCase):

    """
    This class defines unit tests for the ModelSelection class in the
    ModelSelectFinal module. The tests cover various scenarios and
    methods in the ModelSelection class.
    """

    def setUp(self):
        '''
        Sets up the proper environment for testing each test method.

        Attributes
        ----------
        self.test_instance : ModelSelection
            An instance of the ModelSelection class for testing.
        self.test_instance.data : pd.DataFrame
            DataFrame containing test data. (Not split data)
        self.test_instance.x : pd.DataFrame
            x feature matrix.
        self.test_instance.y : pd.Series
            Dependent variable.
        self.test_instance.x_train : pd.DataFrame
            Training set of x feature matrix.
        self.test_instance.x_test : pd.DataFrame
            Testing set of x feature matrix.
        self.test_instance.y_train : pd.Series
            Training set of the target variable.
        self.test_instance.y_test : pd.Series
            Testing set of the dependent variable.

        Returns
        -------
        None
        
        '''

        self.test_instance = ModelSelection()
        self.test_instance.data = pd.read_csv(
            'docs\\Advertising.csv').head(50)  # just 50 rows
        self.test_instance.x = self.test_instance.data.drop('sales', axis=1)
        self.test_instance.y = self.test_instance.data['sales']

        self.test_instance.x_train, \
            self.test_instance.x_test, \
            self.test_instance.y_train, \
            self.test_instance.y_test = self.test_instance.preprocess()

        self.reg_test_model = LinearRegression()
        self.class_test_model = LogisticRegression()

    def tearDown(self) -> None:
        """
        Resets the testing environment to ensure independance between every
        test.

        """
        test_method_name = self._testMethodName
        print(  # Gives clarity on what tests are working
            f'\n\nTEST {test_method_name.replace("test_","").upper()} ' \
                'CLEARED\n\n'.center(40))
        self.test_instance = None

    # set_values()
    def test_set_values(self):
        """Tests the setting of values.

        This method ensures that the interface loop for target-setting works,
        both for invalid and valid inputs.

        """
        # The first try an invalid than a valid choice('sales') to check
        # execution of both scenarios. This can be done in one method because
        # they are in a loop.
        user_tries = ['invalid', 'sales']
        with patch('builtins.input', side_effect=user_tries):
            testing_x, testing_y = self.test_instance.set_values()

            # pandas builtin assert because otherwise 'the truth value of df
            # is ambigous'
            pd.testing.assert_frame_equal(self.test_instance.x, testing_x)
            pd.testing.assert_series_equal(self.test_instance.y, testing_y)

    # load_data()
    def test_load_data_success(self) -> None:
        """
        Test successful data loading scenario.

        This method ensures that data has been loaded correctly and that x
        and y values are correctly defined.

        """
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.test_instance.data
            # User will 'choose' column 'sales' as target
            with patch('builtins.input', side_effect=['sales']):
                self.test_instance.load_data('Doesnt matter')

        self.assertIsNotNone(self.test_instance.data)
        pd.testing.assert_frame_equal(
            self.test_instance.x, self.test_instance.data.drop(
                columns=['sales']))
        pd.testing.assert_series_equal(
            self.test_instance.y,
            self.test_instance.data['sales'])

    def test_load_data_fail(self) -> None:
        """
        Tests unsuccessful data loading scenario.

        This method ensures that failure to pass an existing CSV-type
        file will result in FileNotFoundError.

        """

        invalid_path = "this file doesnt exist/isn't a csv"
        with patch('pandas.read_csv', side_effect=FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):
                self.test_instance.load_data(invalid_path)

    # complete_data()
    def test_complete_data_drop_incomplete(self) -> None:
        """Test execution when data is incomplete and gets dropped.

        This method checks if the missing data is correctly dropped
        when the user asks for it to be.

        """
        self.test_instance.data.iloc[1, 1] = None  # Create a missing value
        # Prompt: Drop missing values? - mock imput yes
        with patch('builtins.input', return_value='Y'):
            # Mock the output of set_values to continue without error
            with patch('ModelSelectFinal.ModelSelection.set_values') \
                    as mock_set_values:
                mock_set_values.return_value = \
                    self.test_instance.data.dropna(), \
                    self.test_instance.y

                self.test_instance.complete_data()

        # Assert that there is no missing data
        self.assertFalse(self.test_instance.data.isna().any().any())

    def test_complete_data_incomplete_exit(self) -> None: # FIXME
        """Tests execution if incomplete data is not digitalized.

        This method ensures that the program has ended correctly
        if the incomplete data isn't decided to be dropped - aka
        if model training and testing is impossibe.

        """
        # self.test_instance.data.iloc[1, 1] = None  # Create a missing value
        # # Prompt: Drop missing values? - mock imput no
        # print('defined missing value')
        # with patch('builtins.input',return_value='Anything but Y'):
        #     # patch('builtins.print') as mock_print:
        #     print('answered wromg')
        #     with self.assertRaises(SystemExit):
        #         with patch('sys.exit') as mock_exit:
        #             print('run the test')
        #             self.test_instance.complete_data()
        # mock_exit.assert_called_once()
        # Confirm exit message
        # mock_print.assert_called_once_with('Model selection cannot be performed.')
        pass

    def test_complete_data_digitalize(self) -> None:
        """Test digitalization of data.

        This method simulates an instance of non-numerical data and ensures it
        gets properly handled after one incorrect input.

        """
        # Create column of non-numveric data
        self.test_instance.x['feature1'] = ['option1', 'option2'] * 25

        with patch('builtins.input', side_effect=['invalid', 'Y']), \
                patch('builtins.print') as mock_print:
            self.test_instance.complete_data()

        mock_print.assert_has_calls([
            call('Incorrect input.'),
            call('\nData has been digitalized and is ready for evaluation.\n')
        ])
        self.assertTrue(
            self.test_instance.x.select_dtypes(
                include=['object']).columns.empty)

    def test_complete_data_no_digitalization_when_needed(self) -> None: # FIXME
        """Tests exit execution if digitalization is refused.

        """
        # self.test_instance.x['feature1'] = ['option1', 'option2'] * 25

        # with patch('builtins.input', side_effect=['N']), \
        #     patch('builtins.print') as mock_print:
        #     with self.assertRaises(SystemExit):
        #         with patch('builtins.exit') as mock_exit:
        #             #self.test_instance.complete_data()
        #             pass

        # mock_exit.assert_called_once()

        # mock_print.assert_called_once_with("Model selection can\'t be performed.")
        pass

    # preprocess()
    def test_preprocess_regression(self) -> None:
        """Test preprocess in case of regressor.

        This method ensures proper execution of train_test_split, scaling and
        because it is regession also polynomial regression.

        """
        self.test_instance.regressor = True
        with patch('sklearn.model_selection.train_test_split') as mock_split, \
                patch('sklearn.preprocessing.PolynomialFeatures.fit_transform') as mock_fit_trans, \
                patch('sklearn.preprocessing.StandardScaler') as mock_scaler:
            mock_split.return_value = (
                self.test_instance.x_train, self.test_instance.x_test,
                self.test_instance.y_train, self.test_instance.y_test)
            # mock_scaler_instance = mock_scaler.return_value
            self.test_instance.preprocess()

        # Assertions
        self.assertTrue(self.test_instance.x_train is not None)
        self.assertTrue(self.test_instance.x_test is not None)
        self.assertTrue(self.test_instance.y_train is not None)
        self.assertTrue(self.test_instance.y_test is not None)

        # Check if polynomial regression fit_transform is called
        mock_fit_trans.assert_called()

        # Check if train_test_split is called correctly
        mock_split.assert_called_once_with(
            self.test_instance.x,
            self.test_instance.y,
            test_size=0.3,
            random_state=101)

        # Check if scaling fit and transform are called correctly
        mock_scaler.return_value.transform.assert_has_calls([
            unittest.mock.call(self.test_instance.x_train),
            unittest.mock.call(self.test_instance.x_test)
        ])
        mock_scaler.return_value.fit.assert_called_once_with(
            self.test_instance.x_train)

    def test_preprocess_classification(self) -> None:
        """Test preprocess in case of classification.

        This method ensures that train_test_split and scaling
        are done correctly and that there is no polynomial regression
        happening.

        """
        self.test_instance.regressor = False
        with patch('sklearn.model_selection.train_test_split') as mock_split, \
                patch('sklearn.preprocessing.PolynomialFeatures.fit_transform') as mock_fit_trans, \
                patch('sklearn.preprocessing.StandardScaler') as mock_scaler:
            mock_split.return_value = (
                self.test_instance.x_train, self.test_instance.x_test,
                self.test_instance.y_train, self.test_instance.y_test)
            # mock_scaler_instance = mock_scaler.return_value
            self.test_instance.preprocess()

        # Assertions
        self.assertTrue(self.test_instance.x_train is not None)
        self.assertTrue(self.test_instance.x_test is not None)
        self.assertTrue(self.test_instance.y_train is not None)
        self.assertTrue(self.test_instance.y_test is not None)

        # Check if polynomial regression fit_transform is called
        mock_fit_trans.assert_not_called()

        # Check if train_test_split is called correctly
        mock_split.assert_called_once_with(
            self.test_instance.x,
            self.test_instance.y,
            test_size=0.3,
            random_state=101)

        # Check if scaling fit and transform are called correctly
        mock_scaler.return_value.transform.assert_has_calls([
            unittest.mock.call(self.test_instance.x_train),
            unittest.mock.call(self.test_instance.x_test)
        ])
        mock_scaler.return_value.fit.assert_called_once_with(
            self.test_instance.x_train)

    # save_model()
    def test_save_model_invalid_valid(self) -> None:
        """Test axecution for both invalid and valid input.
        (Covering all outputs)

        """
        with patch('builtins.input', side_effect=[
                                        'invalid_filename.txt',
                                        'valid_filename.joblib']):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    self.test_instance.save_model('Pretend this is a model')

        mock_exit.assert_called_once()
        mock_print.assert_called_with(
            'Invalid filename. Must end with \'.joblib\'')

    # initialize_model
    def test_initialize_model_reg(self) -> None:
        """Tests initialization of the algorithm in the case of regression.

        This method ensures that every neccesary method is called and that
        the model selection process is set to regression.

        """
        self.test_instance.regressor = True
        # .preprocess needs 4-element output to avoid error.

        with patch('ModelSelectFinal.ModelSelection.choose_model') \
            as mock_choose, \
            patch('ModelSelectFinal.ModelSelection.preprocess', \
                        side_effect=[[None, None, None, None]]) \
            as mock_pre, \
            patch('ModelSelectFinal.ModelSelection.calc_ideal_regression_model') \
            as mock_calc_reg:
            self.test_instance.initialize_model()

        self.assertTrue(self.test_instance.regressor)
        mock_choose.assert_called_once()
        mock_pre.assert_called_once()
        mock_calc_reg.assert_called_once()

    def test_initialize_model_class(self) -> None:
        """Tests initialization of the algorithm in the case of classification.

        This method ensures that every neccesary method is called and that
        the model selection process is still set to classification.

        """
        self.test_instance.regressor = False
        with patch('ModelSelectFinal.ModelSelection.choose_model') \
                as mock_choose, \
                patch('ModelSelectFinal.ModelSelection.preprocess',
                      side_effect=[[None, None, None, None]]) \
                as mock_pre, \
                patch('ModelSelectFinal.ModelSelection.calc_ideal_classification_model') \
                as mock_calc_class:
            self.test_instance.initialize_model()

        mock_choose.assert_called_once()
        mock_pre.assert_called_once()
        mock_calc_class.assert_called_once()

    # choose_model()
    def test_choose_model_invalid_then_regressor(self) -> None:
        """Test that model type is set to regressor after choice.
        (ModelSelection.regressor = True)

        This method checks both for correct regressor flag definition and
        correct execution upon an invalid input.

        """
        self.test_instance.regressor = None  # Ensure reg hasnt already been defined
        with patch('builtins.input', side_effect=['invalid input', 'R']), \
                patch('builtins.print') as mock_print:
            self.test_instance.choose_model()
        mock_print.assert_called_once_with('Invalid choice')
        self.assertTrue(self.test_instance.regressor)

    def test_choose_model_classifier(self) -> None:
        """Test that model type is set to classification after choice.
        (ModelSelection.regressor = False)

        """
        self.test_instance.regressor = None
        with patch('builtins.input', return_value='C'):
            self.test_instance.choose_model()

        # Check if self.regressor is set to False
        self.assertFalse(self.test_instance.regressor)

    def test_choose_model_r_to_c_conversion(self) -> None:
        """Test conversion to classifier if rergressor is wrongly chosen.

        """
        self.test_instance.regressor = None
        # Condition for conversion : object type in target column:
        self.test_instance.y = pd.Series(['option1', 'option2'] * 25)
        # 1: Choose regressor
        # 2: Choose yes (I want to convert to classifier)
        with patch('builtins.input', side_effect=['R', 'Y']):
            self.test_instance.choose_model()
        self.assertFalse(self.test_instance.regressor)

    def test_choose_model_conversion_denied(self) -> None:
        """Test exit execution when data r->c conversion is denied.

        """
        self.test_instance.y = pd.Series(['option1', 'option2'] * 25)

        with patch('builtins.input', side_effect=['R', 'Anything but Y']):
            with self.assertRaises(SystemExit):
                with patch('sys.exit') as mock_exit:
                    self.test_instance.choose_model()
                    mock_exit.assert_called_once()

    # grid_model()
    def test_grid_model(self) -> None:
        """Test hyperparameter tuning and creating+fitting GridSearchCV model.

        This method ensures that the GridSearchCV model created in ModelSelect.grid_model
        is passed the right parameters, it is fitted properly and returned.

        """
        base_model_mock = Mock()
        param_grid_mock = {'param1': [1, 2], 'param2': [3, 4]}

        with patch('sklearn.model_selection.GridSearchCV') as mock_grid_search_cv:
            # Mock the fit method of GridSearchCV
            mock_fit = Mock()
            mock_grid_search_cv.return_value = mock_fit

            # Call the grid_model method
            grid_model = self.test_instance.grid_model(
                base_model=base_model_mock,
                param_grid=param_grid_mock,
                x_train=self.test_instance.x_train,
                y_train=self.test_instance.y_train)

            # Assert that GridSearchCV parameters were passed correctly
            mock_grid_search_cv.assert_called_once_with(
                estimator=base_model_mock,
                param_grid=param_grid_mock,
                cv=10)

            # Assert that the GridSearch is fit with the right parameters
            mock_fit.fit.assert_called_once_with(self.test_instance.x_train,
                                                 self.test_instance.y_train)

            # Assert that the ModelSelection.grid_model method returns the fitted
            # result of GridSearchCV
            self.assertEqual(grid_model, mock_fit)

    def test_calc_ideal_regression_model(self) -> None:
        """Test the calculation of the best suited regression model.

        This method ensures a report of metrics, and a call to confirm suggested
        model. It ensures the model that is suggested is a GridSearch-type meaning
        that the hyperparameters have been tuned as the original estimator was converted
        to GridSearchCV.
        ModelSelection.grid_model() rmse and r2 score are ensured to be called a
        correct amount of times.

        """
        self.test_instance.regressor = True
        # rmse & r2 need int output to be compared in the calculation. x5
        # amount of instances
        with patch('ModelSelectFinal.ModelSelection.regression_report') \
                as mock_reg_report, \
                patch('ModelSelectFinal.ModelSelection.confirm_model_choice') \
                as mock_confirm_choice, \
                patch('ModelSelectFinal.ModelSelection.get_rmse', side_effect=[1, 2, 3, 4, 5]) \
                as mock_rmse, \
                patch('ModelSelectFinal.ModelSelection.get_r2_score', side_effect=[1, 2, 3, 4, 5]) \
                as mock_r2:

            self.test_instance.calc_ideal_regression_model()

        # Assertions
        mock_reg_report.assert_called_once()
        self.assertEqual(mock_rmse.call_count, 5)
        self.assertEqual(mock_r2.call_count, 5)

        self.assertEqual(self.test_instance.regressor, True)
        self.assertIsInstance(
            mock_confirm_choice.call_args[0][0],
            GridSearchCV)
        # Lasso is the best fitting model according to r2score + other metrics

    def test_calc_ideal_classification_model(self) -> None:  # !!!!
        """Test the creation and calculation of the best suited classification model.

        This model ensures that reports are called for every type of model and that
        the confirmation model is called with a tuned model.

        """
        self.test_instance.regressor = False
        # turn data to ready for classification. It's arbitary so there's no
        # respect for original data
        self.test_instance.y = ['opt1', 'opt2'] * 25
        self.test_instance.x_train, \
            self.test_instance.x_test, \
            self.test_instance.y_train, \
            self.test_instance.y_test = self.test_instance.preprocess()

        with patch('ModelSelectFinal.ModelSelection.classification_reports', side_effect=[1, 2, 3]) as mock_report, \
                patch('ModelSelectFinal.ModelSelection.confirm_model_choice') as mock_confirm, \
                patch('ModelSelectFinal.ModelSelection.grid_model') as mock_g_model:
            self.test_instance.calc_ideal_classification_model()

        self.assertEqual(mock_report.call_count, 3)
        self.assertEqual(mock_g_model.call_count, 3)
        # self.assertIsInstance(mock_confirm.call_args[0][0], GridSearchCV)

        self.assertEqual(self.test_instance.regressor, False)

    # get_ metrics()
    def test_get_mae(self) -> None:
        """Test the proper fitting and predicting of a regression model,
        and ensureing correct return MAE from passed model.

        """
        # Get predictions
        from sklearn.metrics import mean_absolute_error
        self.reg_test_model.fit(
            self.test_instance.x_train,
            self.test_instance.y_train)
        test_pred = self.reg_test_model.predict(self.test_instance.x_test)

        # Generate comparison MAE
        expected_output = mean_absolute_error(
            self.test_instance.y_test,
            test_pred)

        with patch('sklearn.linear_model.LinearRegression.predict', return_value=(test_pred)) \
                as mock_predict:
            test_output = self.test_instance.get_mae(self.reg_test_model)

        self.assertEqual(expected_output, test_output)
        mock_predict.assert_called_once_with(self.test_instance.x_test)

    def test_get_rmse(self) -> None:
        """Test the proper fitting and predicting of a regression model,
        and ensureing correct return RMSE from passed model.
        """

        # Get predictions
        from sklearn.metrics import mean_squared_error
        self.reg_test_model.fit(
            self.test_instance.x_train,
            self.test_instance.y_train)
        test_pred = self.reg_test_model.predict(self.test_instance.x_test)

        # Generate RMSE comparison
        expected_output = np.sqrt(
            mean_squared_error(
                self.test_instance.y_test,
                test_pred))

        with patch('sklearn.linear_model.LinearRegression.predict', return_value=(test_pred)) \
                as mock_predict:
            test_output = self.test_instance.get_rmse(self.reg_test_model)

        self.assertEqual(expected_output, test_output)
        mock_predict.assert_called_once_with(self.test_instance.x_test)

    def test_get_r2_score(self) -> None:
        """Test the proper fitting and predicting of a regression model,
        and ensuring correct return r2 score from passed model.

        """
        # Get predictions
        self.reg_test_model.fit(
            self.test_instance.x_train,
            self.test_instance.y_train)

        # Make r2 score comparison
        expected_output = self.reg_test_model.score(
            self.test_instance.x_test, self.test_instance.y_test)

        test_output = self.test_instance.get_r2_score(self.reg_test_model)
        self.assertEqual(expected_output, test_output)

    # Reports
    def test_regression_report(self) -> None:
        """Test that every metric for each model in a list is being printed.

        This method ensures the metrics are calculated and show up in the terminal.

        """
        # Define a mock model for testing
        mock_model = Mock()
        string_finder_obj = StringIO()

        with patch('ModelSelectFinal.ModelSelection.get_mae') as mock_mae, \
                patch('ModelSelectFinal.ModelSelection.get_rmse') as mock_rmse, \
                patch('ModelSelectFinal.ModelSelection.get_r2_score') as mock_r2:
            with patch("sys.stdout", new=string_finder_obj):

                # Mock the get_mae, get_rmse, and get_r2_score methods
                mock_mae.return_value = str(0.2)
                mock_rmse.return_value = str(0.3)
                mock_r2.return_value = str(0.9)
                mock_model.estimator = 'LinearRegression'  # Mock estimator name

                # Call the regression_report method with the mock model
                self.test_instance.regression_report([mock_model, mock_model])

            expected_values = [mock_mae.return_value,
                               mock_rmse.return_value,
                               mock_r2.return_value,
                               mock_model.estimator]

            found_string_output = string_finder_obj.getvalue()
            for value in expected_values:
                self.assertIn(value, found_string_output)

            mock_mae.assert_called_with(mock_model)
            mock_rmse.assert_called_with(mock_model)
            mock_r2.assert_called_with(mock_model)

    def test_classification_reports(self) -> None:
        """Test to ensure reports are properly generated and displayed.

        This method ensures that metrics are being called and printed and that the right
        value is being returned. It ensures that plots arent shown by default.

        """
        self.test_instance.regressor = False
        mock_model = Mock()
        # Set up to 'catch' print returns
        string_finder_obj = StringIO()
        sys.stdout = string_finder_obj

        with patch('sklearn.metrics.ConfusionMatrixDisplay.from_estimator'), \
                patch('matplotlib.pyplot.title') as mock_title, \
                patch('matplotlib.pyplot.show') as mock_show, \
                patch('sklearn.metrics.accuracy_score') as mock_acc, \
                patch('sklearn.metrics.classification_report') as mock_c_report:

            # Since everything is mocked, we 'skip' predict
            mock_model.predict.return_value = None

            # Expected values
            mock_model.estimator = 'Mock estimator'
            mock_c_report.return_value = 'Mock classification report'
            mock_acc.return_value = 0  # It will be converted in the method, so return should be str
            mock_model.best_params_ = 'Mock best params'

            actual_return = self.test_instance.classification_reports(
                model=mock_model)

        # Gather prints and reset
        captured_output = string_finder_obj.getvalue()
        # Reset the standard output to the original value
        sys.stdout = sys.__stdout__

        # Assert return is accuracy
        self.assertEqual(actual_return, mock_acc.return_value)
        self.assertEqual(self.test_instance.regressor, False)
        # Assert that out expected values are included in the print.
        self.assertIn(mock_model.estimator, captured_output)
        self.assertIn(mock_c_report.return_value, captured_output)
        self.assertIn(str(mock_acc.return_value), captured_output)
        self.assertIn(mock_model.best_params_, captured_output)

        mock_title.assert_called()
        mock_show.assert_not_called()  # Default parameter show=False in proper use

    def test_classification_reports_show_plot(self) -> None:
        """Tests that plots in ModelSelection.classification_reports() only
        show up if explicitly stated.

        """
        # Set up values + methods irrelevant to the test
        mock_model = Mock()
        mock_model.predict.return_value = None
        mock_model.estimator = 'Mock Estimator'
        mock_model.best_params_ = 'Mock best params'
        with patch('sklearn.metrics.ConfusionMatrixDisplay.from_estimator'), \
                patch('matplotlib.pyplot.title'), \
                patch('matplotlib.pyplot.show') as mock_show, \
                patch('sklearn.metrics.accuracy_score'), \
                patch('sklearn.metrics.classification_report'):

            self.test_instance.classification_reports(
                mock_model, show=True)  # Set param True

            # Assert that plt.show was called once
            mock_show.assert_called_once()

    # data_report()
    def test_data_report_plot_regression(self) -> None:
        """Test the plotting of regression data.

        This method ensures that only the regression-friendly plots are being shown,
        and that pairplot is not being shown as data is to big. It also ensures the right
        params are being passed and that the total number of shown plots in this scenario is 2.

        """
        self.test_instance.regressor = True
        # Make data too big to meet requirements for pairplot
        self.test_instance.x = pd.concat(
            [self.test_instance.x, self.test_instance.x], axis=1)

        with patch('seaborn.histplot') as mock_histplot, \
                patch('seaborn.pairplot') as mock_pairplot, \
                patch('seaborn.countplot') as mock_countplot, \
                patch('seaborn.heatmap') as mock_heatmap, \
                patch('matplotlib.pyplot.show') as mock_show:
            self.test_instance.data_report()

            # Assertions for regression case
            mock_histplot.assert_called_once_with(
                data=self.test_instance.data,
                x='sales',
                bins=25,
                kde=True)
            mock_heatmap.assert_called_once()
            mock_pairplot.assert_not_called()
            mock_countplot.assert_not_called()
            self.assertEqual(mock_show.call_count, 2)

    def test_data_report_plot_classification(self) -> None:
        """Test the plotting of classification data.

        This method ensures that only classification-friendly plots are being shown,
        and pairplot is included data fits. It also ensures the right
        params are being passed and that the total number of shown plots in this scenario is 3.

        """
        with patch('seaborn.countplot') as mock_countplot, \
                patch('seaborn.histplot') as mock_histplot, \
                patch('seaborn.pairplot') as mock_pairplot, \
                patch('seaborn.heatmap') as mock_heatmap, \
                patch('matplotlib.pyplot.show') as mock_show:
            # Call the data_report method with regressor=False
            self.test_instance.regressor = False
            self.test_instance.data_report()

            # Assertions for classification case
            mock_countplot.assert_called_once_with(
                data=self.test_instance.data,
                x=self.test_instance.y)
            mock_heatmap.assert_called_once()
            mock_pairplot.assert_called_once()  # Since x columns < 5 (4)

            self.assertEqual(mock_show.call_count, 3)
            mock_histplot.assert_not_called()

    # confirm_model_choice()
    def test_confirm_model_choice_invalid_input_then_valid(self) -> None:
        """Test choosing to save after invalid input.

        """
        with patch('builtins.input', side_effect=['invalid', 'Y']):  # Two tries
            with patch('ModelSelectFinal.ModelSelection.save_model') as mock_save:
                with patch('builtins.print') as mock_print:
                    self.test_instance.confirm_model_choice(None)
                    mock_print.assert_called_once_with('Invalid input.')
                    mock_save.assert_called_once()

    def test_confirm_model_choice_exit(self) -> None:
        """Test not saving model and exiting program.

        """
        with patch('builtins.input', side_effect=['N']):
            with self.assertRaises(SystemExit):
                with patch('sys.exit') as mock_exit:
                    self.test_instance.confirm_model_choice(None)
                    mock_exit.assert_called_once()


if __name__ == '__main__':
    PROGRESS_BANNER = ' -' * 10 + ' TESTING IN PROGRESS ' + '- ' * 10
    print(PROGRESS_BANNER.center(40), '\n')

    unittest.main()

    EXIT_BANNER = ' -' * 10 + ' TESTING SUCCESSFUL ' + '- ' * 10
    print('\n', EXIT_BANNER.center(40))
