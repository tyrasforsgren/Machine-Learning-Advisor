import numpy as np
import pandas as pd
import unittest
from sklearn.linear_model import LinearRegression, LogisticRegression

from unittest.mock import patch, Mock


from ModelSelectFinal import ModelSelection


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        self.test_instance = ModelSelection()
        self.test_instance.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [10, 20, 30, 40]
        })
        self.test_instance.X = self.test_instance.data.drop(columns=['target'])
        self.test_instance.y = self.test_instance.data['target']

        self.reg_test_model = LinearRegression()
        self.class_test_model = LogisticRegression() 
        self.test_instance.X_train, self.test_instance.X_test,\
            self.test_instance.y_train, self.test_instance.y_test = self.test_instance.preprocess()
    
    def tearDown(self) -> None:
        self.test_instance = None

    # load_data()
    def test_load_data_success(self) -> None:
        with patch ('pandas.read_csv') as mock_read_csv:
            with patch ('ModelSelectFinal.ModelSelection.set_values') as mock_set_values:
                mock_read_csv.return_value = self.test_instance.data

                mock_X = self.test_instance.X
                mock_y = self.test_instance.y
                mock_set_values.return_value = mock_X, mock_y

                self.test_instance.load_data('path doesnt mater due to mock_read_csv')
                mock_set_values.assert_called_once()

            self.assertIsNotNone(self.test_instance.data)
            pd.testing.assert_frame_equal(self.test_instance.X, mock_X)  # Compare 
            pd.testing.assert_series_equal(self.test_instance.y, mock_y)  # Compare 

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
        user_tries = ['invalid', 'target']
        # 'target' = valid, so loop should end
        with patch ('builtins.input',side_effect=user_tries):
            # Prompt: Choose dependant value from list: [columns]
            test_X, test_y = self.test_instance.set_values()
            pd.testing.assert_frame_equal(self.test_instance.X,test_X)
            pd.testing.assert_series_equal(self.test_instance.y,test_y)

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
        self.test_instance.X['feature1'] = ['a','list','of','objects']
        self.test_instance.X['feature2'] = ['list','of','objects','a']

        with patch ('builtins.input',return_value='Y'):
            self.test_instance.complete_data()
        
        self.assertTrue(self.test_instance.X.select_dtypes(include=['object']).columns.empty
)
   
    # NOT DONE
    def test_data_report(self) -> None: 
        pass

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

    def test_calc_ideal_regression_model(self):

        with patch ('builtins.input',return_value='sales'):
            self.test_instance.data = self.test_instance.load_data('docs\Advertising.csv')
        
        self.test_instance.X_train, self.test_instance.X_test,\
            self.test_instance.y_train, self.test_instance.y_test = self.test_instance.preprocess()
        self.test_instance.calc_ideal_regression_model()

        
        # Replace the following line with your actual test predictions
        test_predictions = self.test_instance.regression_report_data['best_model'].predict(self.test_instance.X_test)

        # Assertions
        self.assertEqual(self.test_instance.regressor, True)
        self.assertIsNotNone(self.test_instance.regression_report_data['best_model'])
        self.assertIsNotNone(self.test_instance.regression_report_data['best_parameters'])
        self.assertIsNotNone(self.test_instance.regression_report_data['best_score'])

        # Calculate the MAE with real data
        MAE = mean_absolute_error(self.test_instance.y_test, test_predictions)
        self.assertIsNotNone(MAE)

if __name__ == '__main__':
    unittest.main()