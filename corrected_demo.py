from project.ModelSelectFinal import ModelSelection


def reg_test(test):
    test.load_data('docs\\Advertising.csv')
    # Dependant = 'sales'

def clif_test(test):
    test.load_data('docs\\heart.csv') # Relatively quickd
    # Dependant = 'target'
    # Drop - stops upon wrong y/n
    # Error upon choosing r instead of c

def dummy_test(test):
    test.load_data('docs\\insurance.csv')
    # Dependant = 'charges'

if __name__ == '__main__':
    test_model = ModelSelection()
    clif_test(test_model)
    print(type(test_model.y))
    # test_model.initialize_model()


# Test for regression: (dependant value is 'sales')
#reg_test = ModelSelection('docs\Advertising.csv')

# Test for classification and missing data: (dependant value is 'target')
# class_test = ModelSelection('docs\heart.csv')

# Test for dummies and regression (dependant value is 'charges')
# OBS : This takes a very long time to run.
# dummy_test = ModelSelection('docs\insurance.csv')
