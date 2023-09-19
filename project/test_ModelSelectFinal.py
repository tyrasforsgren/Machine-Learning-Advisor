import unittest
import os
import pandas as pd
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from ModelSelectFinal import ModelSelection
# sys.path.append(r'C:\Users\tyraf\Documents\term_3_python_ai\repos\testing_prev_ml_exam\project\ModelSelectFinal.py')
# import ModelSelection


class TestModelSelection(unittest.TestCase):
    '''

    def setUp(self):
        self.test_data = pd.DataFrame({
            'ind':[1,2,3,5,6,7],
            'ind2':[3,4,5,6,7,8],
            'dep':[4,5,6,8,9,2]
            })
        
        # mode='w+' mwan writing+reading mode simlutaneously
        # delete=True (default) means file will be deleted upon exit with-statement
        self.temp_file = NamedTemporaryFile(
            dir=os.path.abspath(r'C:\Users\tyraf\Documents\term_3_python_ai\repos\testing_prev_ml_exam\project'),
            mode='w+',
            suffix='.csv',
            delete=False)
        
        self.test_data.to_csv(self.temp_file, index=False)
        self.temp_file.close()

    def tearDown(self):
        
        # self.temp_file.close()
        os.remove(self.temp_file.name)
        #pass
    '''

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
        with patch ''

    def test_set_values(self):
        model_selection = ModelSelection()
        model_selection.data = self.basic_clean_data
        with patch('builtins.input', return_value='dep'):
            # mock the user choosing the 'dep' column as dependant value
            pass
            
    
        # Assert that the created x column is the same
        self.assertEqual(model_selection.X.columns.tolist(),
                         self.test_data.drop('dep',axis=1).columns.tolist())
        # Assert that the Dependant label name is the same
        self.assertEqual(model_selection.y.name, 'dep')

    def test_complete_data_perfect(self) -> None:
        pass

    def test_complete_data_remove_missing(self) -> None:
        with NamedTemporaryFile(dir=r'C:\Users\tyraf\Documents\term_3_python_ai\repos\testing_prev_ml_exam\project',
                            mode='w+',
                            suffix='.csv',
                            delete=False) as missing_temp_file:
        
            missing_file_data = pd.DataFrame({'ind':[1,None,5,7],'dep':[1,2,3,4]}) # Create missing value
            missing_file_data.to_csv(missing_temp_file, index=False)

            with patch('builtins.input', return_value='Y'):
                model_selection = ModelSelection(missing_temp_file.name)


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