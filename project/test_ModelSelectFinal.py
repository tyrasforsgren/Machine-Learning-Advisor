import unittest
import os
import pandas as pd
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from ModelSelectFinal import ModelSelection
# sys.path.append(r'C:\Users\tyraf\Documents\term_3_python_ai\repos\testing_prev_ml_exam\project\ModelSelectFinal.py')
# import ModelSelection


class TestModelSelection(unittest.TestCase):


    def setUp(self) -> None:
        # Test with clean data
        self.basic_clean_data = pd.DataFrame({
            'ind1':[1,2,3,4],
            'int2':[5,6,7,8],
            'dep':[4,5,7,8]
        })
        self.path = r'project\tests'
        # Test conversion to classifier
        
        # Test conversion to dummies (non numeric x vals)

    def test_load_data_success(self) -> None:
        test_instance = ModelSelection()
        with patch ('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.basic_clean_data

            with patch ('ModelSelectFinal.ModelSelection.set_values') as mock_set_values:
                mock_X = self.basic_clean_data.drop('dep',axis=1)
                mock_y = self.basic_clean_data['dep']
                mock_set_values.return_value = mock_X, mock_y

                test_instance.load_data('path doesnt mater due to mock_read_csv')
                mock_set_values.assert_called_once()

            self.assertIsNotNone(test_instance.data)
            pd.testing.assert_frame_equal(test_instance.data, self.basic_clean_data)  # Compare DataFrames
            pd.testing.assert_frame_equal(test_instance.X, mock_X)  # Compare Series
            pd.testing.assert_series_equal(test_instance.y, mock_y)  # Compare Series


    def test_set_values(self):
        pass


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