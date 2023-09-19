
import pandas as pd
import unittest
from unittest.mock import patch

from ModelSelectFinal import ModelSelection


class TestModelSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.test_instance = ModelSelection()
        self.basic_clean_data = pd.DataFrame({
            'ind1':[1,2,3,4],
            'int2':[5,6,7,8],
            'dep':[4,5,7,8]
        })
        self.path = r'project\tests'
        
        # Test conversion to classifier
        
        # Test conversion to dummies (non numeric x vals)

    def tearDown(self) -> None:
        self.test_instance = None
        self.basic_clean_data = None
        self.path = None

    def test_load_data_success(self) -> None:
        with patch ('pandas.read_csv') as mock_read_csv:
            with patch ('ModelSelectFinal.ModelSelection.set_values') as mock_set_values:
                mock_read_csv.return_value = self.basic_clean_data

                mock_X = self.basic_clean_data.drop('dep',axis=1)
                mock_y = self.basic_clean_data['dep']
                mock_set_values.return_value = mock_X, mock_y

                self.test_instance.load_data('path doesnt mater due to mock_read_csv')
                mock_set_values.assert_called_once()

            self.assertIsNotNone(self.test_instance.data)
            pd.testing.assert_frame_equal(self.test_instance.data, self.basic_clean_data)  # Compare DataFrames
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
    

    def test_set_values_success(self):
        self.test_instance.data = self.basic_clean_data
        real_X = self.basic_clean_data.drop('dep',axis=1)
        real_y = self.basic_clean_data['dep']
        # The choices user puts in each iteration
        #user_tries = ['invalid','invalid','dep']
        user_tries = ['invalid', 'dep']
        # 'dep' = valid, so loop should end
        with patch ('builtins.input',side_effect=user_tries):
            # Prompt: Choose dependant value from list: [columns]
            test_X, test_y = self.test_instance.set_values()
            pd.testing.assert_frame_equal(real_X,test_X)
            pd.testing.assert_series_equal(real_y,test_y)

    def test_regressor_selection(self):
        # Test the regressor selection logic
        pass

    def test_classifier_selection(self):
        # Test the classifier selection logic
        pass

    def test_data_preprocessing(self):
        # Test data preprocessing methods
        pass

    def test_regression_models(self):
        # Test the regression model selection and evaluation
        pass

    def test_classification_models(self):
        # Test the classification model selection and evaluation
        pass

if __name__ == '__main__':
    unittest.main()