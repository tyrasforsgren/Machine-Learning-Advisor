from project.ModelSelectFinal import ModelSelection

'''def run():
    pass

def reg_test():
    reg_test = ModelSelection('docs\Advertising.csv')
    # Dependant = 'sales'

def clif_test():
    clif_test = ModelSelection('docs\heart.csv')
    # Dependant = 'target'
    
def dummy_test():
    dummy = ModelSelection('docs\insurance.csv')
    # Dependant = 'charges'

if __name__ == '__main__':
    run()
'''
test = ModelSelection()
import pandas as pd
test = ModelSelection()
test.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [10, 20, 30, 40]
        })
test.X, test.y = test.set_values()# '''
#test.load_data('docs\heart.csv')
test.initialize_model()


# Test for regression: (dependant value is 'sales')
#reg_test = ModelSelection('docs\Advertising.csv')

# Test for classification and missing data: (dependant value is 'target')
# class_test = ModelSelection('docs\heart.csv')

# Test for dummies and regression (dependant value is 'charges')
# OBS : This takes a very long time to run.
# dummy_test = ModelSelection('docs\insurance.csv')
