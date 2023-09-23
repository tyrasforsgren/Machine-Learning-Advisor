from project.ModelSelectFinal import ModelSelection

def run():
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

    test = ModelSelection()

    test.load_data('docs\Advertising.csv')

    test.initialize_model()


# Test for regression: (dependant value is 'sales')
#reg_test = ModelSelection('docs\Advertising.csv')

# Test for classification and missing data: (dependant value is 'target')
# class_test = ModelSelection('docs\heart.csv')

# Test for dummies and regression (dependant value is 'charges')
# OBS : This takes a very long time to run.
# dummy_test = ModelSelection('docs\insurance.csv')
