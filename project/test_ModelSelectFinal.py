import numpy as np
import pandas as pd
import unittest
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.model_selection import GridSearchCV
from unittest.mock import patch, Mock
from io import StringIO

from ModelSelectFinal import ModelSelection


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        self.test_instance = ModelSelection()
        self.test_instance.data = pd.read_csv('docs\\testing_Advertising.csv')
        self.test_instance.X = self.test_instance.data.drop('sales',axis=1)
        self.test_instance.y = self.test_instance.data['sales']

        self.test_instance.X_train, self.test_instance.X_test,\
            self.test_instance.y_train, self.test_instance.y_test = self.test_instance.preprocess()

        self.reg_test_model = LinearRegression()
        self.class_test_model = LogisticRegression() 

    def tearDown(self) -> None:
        test_method_name = self._testMethodName
        print(f'\n\nTEST {test_method_name.replace("test_","").upper()} CLEARED\n\n'.center(40))
        self.test_instance = None

    # load_data()
    def test_load_data_success(self) -> None:
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.test_instance.data

            # Ensure that 'input' is not mocked, so user input can be provided
            with patch('builtins.input', side_effect=['sales']):
                self.test_instance.load_data('docs\\testing_Advertising.csv')

        self.assertIsNotNone(self.test_instance.data)
        pd.testing.assert_frame_equal(self.test_instance.X, self.test_instance.data.drop(columns=['sales']))  # Update with the correct column name
        pd.testing.assert_series_equal(self.test_instance.y, self.test_instance.data['sales'])  # Update with the correct column name


    def test_load_data_fail(self) -> None:
        invalid_path = "this file doesnt exist/isn't a csv"
        # Mock .read_csv to return FileNotFound
        with patch('pandas.read_csv', side_effect=FileNotFoundError()):
            # If The error is the next code block, the assert is True,
            # AKA the test has been passed - source code is working.
            with self.assertRaises(FileNotFoundError):
                self.test_instance.load_data(invalid_path)
    
    # initialize_model
    def test_initialize_model_reg(self) -> None:
        self.test_instance.regressor = True
        with patch ('ModelSelectFinal.ModelSelection.choose_model') as mock_choose:
            with patch ('ModelSelectFinal.ModelSelection.preprocess') as mock_pre:
                with patch ('ModelSelectFinal.ModelSelection.calc_ideal_regression_model') as mock_calc_reg:
                    mock_pre.return_value = 0,0,0,0  # Replace with actual data
                    self.test_instance.initialize_model()
        mock_choose.assert_called_once()
        mock_pre.assert_called_once()
        mock_calc_reg.assert_called_once()

    def test_initialize_model_class(self) -> None:
        self.test_instance.regressor = False
        with patch ('ModelSelectFinal.ModelSelection.choose_model') as mock_choose:
            with patch ('ModelSelectFinal.ModelSelection.preprocess') as mock_pre:
                with patch ('ModelSelectFinal.ModelSelection.calc_ideal_classification_model') as mock_calc_class:
                    mock_pre.return_value = 0,0,0,0  # Replace with actual data
                    self.test_instance.initialize_model()
        mock_choose.assert_called_once()
        mock_pre.assert_called_once()
        mock_calc_class.assert_called_once()

    # set_values()
    def test_set_values(self):
        # The choices user puts in each iteration
        #user_tries = ['invalid','invalid','target']
        user_tries = ['invalid', 'sales']
        # 'target' = valid, so loop should end
        with patch ('builtins.input',side_effect=user_tries):
            # Prompt: Choose dependant value from list: [columns]
            testing_X, testing_y = self.test_instance.set_values()
            pd.testing.assert_frame_equal(self.test_instance.X,testing_X)
            pd.testing.assert_series_equal(self.test_instance.y,testing_y)

    # complete_data()
    def test_complete_data_incomplete(self) -> None:
        self.test_instance.data.iloc[1,1] = None # Create a missing value
        # Prompt: Drop missing values? - mock imput yes
        with patch ('builtins.input') as mock_input:
            mock_input.return_value = 'Y'
            # Mock the output of set_values to continue without error
            with patch('ModelSelectFinal.ModelSelection.set_values') as mock_set_values:
                mock_set_values.return_value = self.test_instance.data.dropna(), self.test_instance.y
                self.test_instance.complete_data()

        # Assert that there is no missing data
        self.assertFalse(self.test_instance.data.isna().any().any())

    def test_complete_data_digitalize(self) -> None:
        self.test_instance.X['feature1'] = ['option1','option2'] * 25

        with patch ('builtins.input',return_value='Y'):
            self.test_instance.complete_data()
        
        self.assertTrue(self.test_instance.X.select_dtypes(include=['object']).columns.empty)

    # choose_model()
    def test_choose_model_regressor(self):
        # Simulate user input 'R' using a with statement
        with patch('builtins.input', return_value='R'):
            # Call the choose_model method
            self.test_instance.choose_model()

        # Check if self.regressor is set to True
        self.assertTrue(self.test_instance.regressor)

    def test_choose_model_classifier(self):
        # Simulate user input 'C' using a with statement
        with patch('builtins.input', return_value='C'):
            # Call the choose_model method
            self.test_instance.choose_model()

        # Check if self.regressor is set to False
        self.assertFalse(self.test_instance.regressor)

    # confirm_model_choice()
    def test_confirm_model_choice_continue(self) -> None:
        with patch('builtins.input', side_effect=['invalid','Y']):
            with patch ('ModelSelectFinal.ModelSelection.save_model') as mock_save:
                with patch('builtins.print') as mock_print:
                    self.test_instance.confirm_model_choice(None)
                    mock_print.assert_called_once()
                    mock_print.assert_called_with('Invalid input.')
                    mock_save.assert_called_once()

    def test_confirm_model_choice_exit(self):
        # Create an instance of ModelSelection
        with patch('builtins.input', side_effect=['N']):
            with self.assertRaises(SystemExit):
                with patch('sys.exit') as mock_exit:
                    self.test_instance.confirm_model_choice(None)
                    mock_exit.assert_called_once()
    
    # preprocess()
    def test_preprocess_regression(self) -> None:
        self.test_instance.regressor = True
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            with patch('sklearn.preprocessing.StandardScaler') as mock_scaler:
                mock_split.return_value = (
                    self.test_instance.X_train, self.test_instance.X_test,
                    self.test_instance.y_train, self.test_instance.y_test)
                mock_scaler_instance = mock_scaler.return_value

                # Call the preprocess method
                self.test_instance.preprocess()

        # Assertions
        self.assertTrue(self.test_instance.X_train is not None)
        self.assertTrue(self.test_instance.X_test is not None)
        self.assertTrue(self.test_instance.y_train is not None)
        self.assertTrue(self.test_instance.y_test is not None)

        # Check if train_test_split is called correctly
        mock_split.assert_called_once_with(
            self.test_instance.X, self.test_instance.y, test_size=0.3, random_state=101)

        # Check if StandardScaler is called correctly
        mock_scaler_instance.fit.assert_called_once_with(self.test_instance.X_train)

        # Check if transform is called for X_train and X_test
        mock_scaler_instance.transform.assert_has_calls([
            unittest.mock.call(self.test_instance.X_train),
            unittest.mock.call(self.test_instance.X_test)
        ])

    # save_model()
    def test_save_model_invalid_valid(self):
        # Define a mock model for testing
        #mock_model = Mock()  # Replace with your model or use MagicMock

        # Mock input, sys.exit, and print
        with patch('builtins.input', side_effect=['invalid_filename.txt', 'valid_filename.joblib']):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    self.test_instance.save_model('Pretend this is a model')

        # Assert that sys.exit was called once (after the second input)
        mock_exit.assert_called_once()

        # Assert that the print method was called with the expected error message
        mock_print.assert_called_with('Invalid filename. Must end with \'.joblib\'')

    # grid_model()
    def test_grid_model(self):
        # Create an instance of ModelSelection
        model_selection = ModelSelection()

        # Define some mock data for the test
        X_train_mock = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train_mock = pd.Series([10, 20, 30])

        # Define a mock base model
        base_model_mock = Mock()

        # Define a mock param grid
        param_grid_mock = {'param1': [1, 2], 'param2': [3, 4]}

        # Mock GridSearchCV
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid_search_cv:
            # Mock the fit method of GridSearchCV
            mock_fit = Mock()
            mock_grid_search_cv.return_value = mock_fit

            # Call the grid_model method
            grid_model = model_selection.grid_model(base_model_mock, param_grid_mock, X_train_mock, y_train_mock)

            # Assert that GridSearchCV was called with the correct parameters
            mock_grid_search_cv.assert_called_once_with(estimator=base_model_mock, param_grid=param_grid_mock, cv=10)

            # Assert that the grid_model method returns the fit result of GridSearchCV
            self.assertEqual(grid_model, mock_fit)

    # get_ ... MAE,RMSE,r2Score
    def test_get_mae(self) -> None:
        from sklearn.metrics import mean_absolute_error
        self.reg_test_model.fit(self.test_instance.X_train,self.test_instance.y_train)
        test_pred = self.reg_test_model.predict(self.test_instance.X_test)

        expected_output = mean_absolute_error(self.test_instance.y_test,test_pred)

        with patch ('sklearn.linear_model.LinearRegression.predict', return_value=(test_pred)) as mock_predict:
            test_output = self.test_instance.get_mae(self.reg_test_model)

        self.assertEqual(expected_output,test_output)
        mock_predict.assert_called_once_with(self.test_instance.X_test)

    def test_get_rmse(self):
        from sklearn.metrics import mean_squared_error
        self.reg_test_model.fit(self.test_instance.X_train,self.test_instance.y_train)
        test_pred = self.reg_test_model.predict(self.test_instance.X_test)

        expected_output = np.sqrt(mean_squared_error(self.test_instance.y_test,test_pred))

        with patch ('sklearn.linear_model.LinearRegression.predict', return_value=(test_pred)) as mock_predict:
            test_output = self.test_instance.get_rmse(self.reg_test_model)

        self.assertEqual(expected_output,test_output)
        mock_predict.assert_called_once_with(self.test_instance.X_test)

    def test_get_r2_score(self) -> None:
        self.reg_test_model.fit(self.test_instance.X_train,self.test_instance.y_train)

        expected_output = self.reg_test_model.score(self.test_instance.X_test,self.test_instance.y_test)

        test_output = self.test_instance.get_r2_score(self.reg_test_model)
        self.assertEqual(expected_output,test_output)
    '''
    def test_calc_ideal_regression_model(self):

        self.test_instance.regressor = True
        with patch ('ModelSelectFinal.ModelSelection.regression_report') as mock_reg_report, \
            patch ('ModelSelectFinal.ModelSelection.confirm_model_choice') as mock_confirm_choice:
            self.test_instance.calc_ideal_regression_model()
     
        # Assertions
        mock_reg_report.assert_called_once()
        # Cannot assert the times get_rmse (etc) have been called b/c we need the real return value

        self.assertEqual(self.test_instance.regressor, True)
        self.assertIsInstance(mock_confirm_choice.call_args[0][0], GridSearchCV)
        # Lasso is the best fitting model according to r2score + other metrics
    '''
    def test_calc_ideal_classification_model(self) -> None:
        self.test_instance.regressor = False
        # turn data to ready for classification. It's arbitary so there's no respect for original data
        self.test_instance.y = ['opt1','opt2'] * 25
        self.test_instance.X_train, self.test_instance.X_test,\
            self.test_instance.y_train, self.test_instance.y_test = self.test_instance.preprocess()

        with patch ('ModelSelectFinal.ModelSelection.classification_reports',side_effect=[1,2,3]) as mock_report, \
            patch ('ModelSelectFinal.ModelSelection.confirm_model_choice') as mock_confirm:
            self.test_instance.calc_ideal_classification_model()

        self.assertEqual(mock_report.call_count,3)

        self.assertIsInstance(mock_confirm.call_args[0][0], GridSearchCV)

        self.assertEqual(self.test_instance.regressor, False)
    
    def test_classification_reports(self):

        self.test_instance.regressor = False
        mock_model = Mock()
        string_finder_obj = StringIO()
        sys.stdout = string_finder_obj

        # Patch the necessary functions and methods
        with patch('sklearn.metrics.ConfusionMatrixDisplay.from_estimator') as mock_from_est, \
                patch('matplotlib.pyplot.title') as mock_title, \
                patch('matplotlib.pyplot.show') as mock_show, \
                patch('sklearn.metrics.accuracy_score') as mock_acc, \
                patch('sklearn.metrics.classification_report') as mock_c_report:

                mock_model.predict.return_value = None # Since everything is mocked, we skip predict

                mock_model.estimator = 'Mock estimator'
                mock_c_report.return_value = 'Mock classification report'
                mock_acc.return_value = 0 # It will be converted in the method, so return should be str
                mock_model.best_params_ = 'Mock best params'

                actual_return = self.test_instance.classification_reports(model=mock_model)

        captured_output = string_finder_obj.getvalue()
        # Reset the standard output to the original value
        sys.stdout = sys.__stdout__


        self.assertEqual(self.test_instance.regressor, False)
        self.assertEqual(actual_return,mock_acc.return_value)

        self.assertIn(mock_model.estimator, captured_output)
        self.assertIn(mock_c_report.return_value, captured_output)
        self.assertIn(str(mock_acc.return_value), captured_output)
        self.assertIn(mock_model.best_params_, captured_output)
        
        mock_title.assert_called()
        mock_show.assert_not_called()
    
    def test_classification_reports_plots(self):
        mock_model = Mock()
        mock_model.predict.return_value = None
        mock_model.estimator = 'Mock Estimator'
        mock_model.best_params_ = 'Mock best params'

        with patch('sklearn.metrics.ConfusionMatrixDisplay.from_estimator'), \
                patch('matplotlib.pyplot.title'), \
                patch('matplotlib.pyplot.show') as mock_show, \
                patch('sklearn.metrics.accuracy_score'), \
                patch('sklearn.metrics.classification_report'):
            
            self.test_instance.classification_reports(mock_model, show=True)

            # Assert that plt.show was called once
            mock_show.assert_called_once()

    def test_regression_report(self):
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
                self.test_instance.regression_report([mock_model])

            expected_values = [mock_mae.return_value,
                               mock_rmse.return_value,
                               mock_r2.return_value,
                               mock_model.estimator,
                               ]
            found_string_output = string_finder_obj.getvalue()
            for value in expected_values:
                self.assertIn(value,found_string_output)

            mock_mae.assert_called_once_with(mock_model)
            mock_rmse.assert_called_once_with(mock_model)
            mock_r2.assert_called_once_with(mock_model)

    def test_data_report_plot_regression(self):

        self.test_instance.X = pd.concat([self.test_instance.X,self.test_instance.X],axis=1)
        with patch('seaborn.histplot') as mock_histplot, \
            patch('seaborn.pairplot') as mock_pairplot, \
            patch('matplotlib.pyplot.show') as mock_show:
            self.test_instance.regressor=True
            self.test_instance.data_report()

            # Assertions for regression case
            mock_histplot.assert_called_once_with(
                data=self.test_instance.data,
                x='sales',
                bins=25,
                kde=True
            )
            mock_pairplot.assert_not_called()
            self.assertEqual(mock_show.call_count,2)

    def test_data_report_plot_classification(self):
        with patch('seaborn.countplot') as mock_countplot, \
            patch('seaborn.pairplot') as mock_pairplot, \
            patch('seaborn.heatmap') as mock_heatmap, \
            patch('matplotlib.pyplot.show') as mock_show:
            # Call the data_report method with regressor=False
            self.test_instance.regressor = False
            self.test_instance.data_report()

            # Assertions for classification case
            mock_countplot.assert_called_once_with(data=self.test_instance.data, x=self.test_instance.y)
            mock_heatmap.assert_called_once()
            self.assertEqual(mock_show.call_count,3)
            mock_pairplot.assert_called_once() # Since x columns < 5 (4)

if __name__ == '__main__':
    progress_banner = ' -'*10+' TESTING IN PROGRESS '+'- '*10
    print(progress_banner.center(40),'\n')
    unittest.main()
    exit_banner = ' -'*10+' TESTING SUCCESSFUL '+'- '*10
    print('\n',exit_banner.center(40))