
# Import the ModelSelection class from the project.ModelSelectFinal module
from project.ModelSelectFinal import ModelSelection

def regression_test(test_instance):
    """Loads data for regression testing with file 'Advertising.csv'.
    Target value: 'sales'

    Parameters
    ----------
    test_instance : ModelSelection
        An instance of the ModelSelection class.

    Returns
    -------
    None

    BUG: The program stops upon incorrect user input
    (choosing 'c' instead of 'r') because it cannot
    handle continous data in target for classification.

    """

    test_instance.load_data('docs\\Advertising.csv')

def classification_test(test_instance):
    """
    Laod data for classification testing with file 'heart.csv'.
    Target value: 'target'

    Parameters
    ----------
    test_instance : ModelSelection
        An instance of the ModelSelection class.

    Returns
    -------
    None

    """
    test_instance.load_data('docs\\heart.csv')

if __name__ == '__main__':
    # Create an instance of the ModelSelection class
    test_model = ModelSelection()

    # Choose one of the testing functions to run
    # (regression_test or classification_test)

    # See target column name in function documentation
    classification_test(test_model)

    # Initialize the model selection process
    test_model.initialize_model()
