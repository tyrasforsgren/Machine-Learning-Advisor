
import pandas as pd
import unittest
from unittest.mock import patch

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

    def tearDown(self) -> None:
        self.test_instance = None

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

    
    def test_regression_models(self):
        # Test the regression model selection and evaluation
        pass

    def test_classification_models(self):
        # Test the classification model selection and evaluation
        pass

    

if __name__ == '__main__':
    unittest.main()