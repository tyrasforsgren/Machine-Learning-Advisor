# Import the ModelSelection class from the project.ModelSelectFinal module
from project.ModelSelectFinal import ModelSelection

# Define a function for regression testing
def reg_test(test): # 20secs calculation
    # Load data from the 'Advertising.csv' file
    test.load_data('docs\\Advertising.csv')
    # Specify the dependent variable as 'sales'

# Define a function for classification testing
def clif_test(test):
    # Load data from the 'heart.csv' file
    test.load_data('docs\\heart.csv')
    # Specify the dependent variable as 'target'
    # The program stops upon incorrect user input (e.g., choosing 'r' instead of 'c')

if __name__ == '__main__':
    # Create an instance of the ModelSelection class
    test_model = ModelSelection()
    
    # Choose one of the testing functions to run (e.g., dummy_test)
    reg_test(test_model)
    
    # Initialize the model selection process
    test_model.initialize_model()
