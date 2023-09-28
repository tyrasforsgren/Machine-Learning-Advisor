
"""
ModelSelection Module

This module defines the ModelSelection class, which is designed to assist in
selecting an ideal machine learning model based on data in CSV form. It provides
methods for data preprocessing, model training, hyperparameter tuning, generating
classification or regression reports, and calculating the ideal model based on
rmse and r2_score for regressors and accuracy for classifiers.

Classes:
    - ModelSelection: A class for selecting ideal machine learning model based on data.

Example Usage:
    model_selector = ModelSelection()
    model_selector.load_data('data.csv')
    model_selector.data_report()
    model_selector.initialize_model()

Note:
    You should load data using 'load_data' and call 'initialize_model'
    before using other methods.

"""

# TODO : The tests stopped working when I moved imports out of
# the respective methods. To avoid redundancy/repeated code, tests should be adjusted
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
# for documentation in method heads


class ModelSelection:
    """
    ModelSelection is a class for selecting an ideal machine learning
    model based on data only in CSV form.

    It provides methods for data preprocessing, model training,
    hyperparameter tuning, generating classification or regression
    reports and calculating ideal model based om rmse and r2_score.

    Attributes
    ----------
    regressor : NoneType
        Flag for type of model (None by default).
    data : NoneType
            The dataset used the models (None by default).
    X : NoneType
            The feature matrix (None by default).
    y : NoneType
            The target values (None by default).
    x_train : NoneType
            The training feature matrix (None by default).
    x_test : NoneType
            The testing feature matrix (None by default).
    y_train : NoneType
            The training target values (None by default).
    y_test : NoneType
        The testing target values

    Methods
    -------
    preprocess():
                Applies polynomial regression, scaling, and train/test
                splitting to the data.
    grid_model():
                Performs hyperparameter tuning using grid search.
    classification_reports():
                Generates metric report for classification cases.
    regression_reports():
                Generates metric report for regression cases.
    load_data(path):
                Loads data from a CSV file.
    initialize_model():
        Initializes the model selection process.
    set_values():
                Sets feature matrix and target values from user input.
    complete_data():
                Checks for missing values or non-numerals(feature matrix only).
        Offers soloutions or ends the program depending.
    data_report():
                Plots information about the total data.
    choose_model():
                Prompts the user to select a model type.
    confirm_model_choice(chosen_model):
                Confirms the selected model.
    save_model(chosen_model):
                Saves the selected model as a .joblib file.
    get_rmse(model):
                Calculates rmse for a given model.
    get_mae(model):
                Calculates mae for a given model.
    get_r2_score(model):
                Calculates r2_score for a given model.
    regression_report(model_types):
                Prints regression reports for a list of models.
    calc_ideal_regression_model():
                Selects the best regression model.
    classification_reports(model, show=False):
            Generates report on one classification model.
    calc_ideal_classification_model()
        Selects the best classification model.

    Example usage:
    ```
    model_selector = ModelSelection()
    model_selector.load_data('data.csv')
    model_selector.data_report()
    model_selector.initialize_model()
    ```

    Note:
    You should load data using 'load_data' and call 'initialize_model' before
    using other methods.

    """

    def __init__(self) -> None:
        """
        Initializes an instance of ModelSelection.
        All attributes are defined outside of initialization.

        Attributes
        ----------
        regressor : NoneType -> bool
            defined in choose_model
        data : NoneType -> pandas.DataFrame
            defined in load_data
        X : NoneType -> pandas.Series
            defined in load_data
        y : NoneType -> pandas.Series
            defined in load_data
        x_train : NoneType -> pandas.DataFrame
            defined in initialize_model
        x_test : NoneType -> pandas.DataFrame
            defined in initialize_model
        y_train : NoneType -> pandas.Series
            defined in initialize_model
        y_test : NoneType -> pandas.Series

        Returns
        -------
        None

        Notes:
        Not to be confused with initialize_model. This method sets up the envrironment
        needed for initialize_model to function.

        """
        self.regressor = None
        self.data = None
        self.X = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, path: str) -> None:
        """Load and prepare data from CSV file.

        This method reads a CSV with the given path and initializes it as a
        pandas.DataFrame, including assigning the X features and the target
        values. It validates that the data is ready for the selection process.

        Parameters
        ----------
        path : str
            string representing the path to the data

        Returns
        -------
        None

        """
        self.data = pd.read_csv(path)
        self.X, self.y = self.set_values()
        self.complete_data()

    def initialize_model(self) -> None:
        """Initializes the model selection process under the assumption that the
        data has already been loaded.

        This method lets the user choose what model type(regression/classification)
        is desired, performs preprocessing on the data, and begins calculations on
        what model is ideal.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes:
        Not to be confused with __init__(). This model initializes
        a chain of events as opposed to __init__, which initializes
        the environment those events take place.

        """
        self.choose_model()

        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = self.preprocess()

        if self.regressor:
            self.calc_ideal_regression_model()
        else:
            self.calc_ideal_classification_model()

    def set_values(self) -> tuple:
        """Lets user assign target value from their CSV file.

        This method splits data (pandas.DataFrame) into an X feature
        matrix and a Series of target values based on the users input.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            A tuple with two elements (pandas.DataFrame, pandas.Series)
            element one (X) represents the X feature matrix
            element two (y) represents the target value

        """
        while True:
            choice = input(f'Choose dependant value from list :\n'
                           f'{list(self.data.columns)}\n')
            # Ensure chosen option exists.
            if choice in list(self.data.columns):
                X = pd.DataFrame(self.data.drop(choice, axis=1))
                y = pd.Series(self.data[choice])
                return X, y
            print('Target not in column list.')

    def complete_data(self) -> None:  # Or exit program
        """Ensures the data is ready to be preproccessed.

        This method checks for missing data, and offers so delete rows with missing
        data. It offers to digitalize X feature matrix if there are non-numerals.

        Parameters
        ----------
        None

        Returns
        -------
        None
        (or exit)

        """
        # If there is missing data :
        if self.data.isnull().sum().sum():
            print('data is missing')
            na_choice = input(
                'Data is incomplete. Do you want to drop rows with missing data?\n'
                '(all will be removed.) \n')  # TODO Specify where missing values are.
            if na_choice.upper() == 'Y':
                print('answered yes')
                # TODO : .dropna() for X and y to avoid target val reset.
                self.data = self.data.dropna()
                print('Incomplete rows have been deleted. \n'
                      'Dependant value has been reset, input again.\n')
                self.X, self.y = self.set_values()
            else:
                print('sys exit should be reaced')
                print('Model selection cannot be performed.')
                exit(1) # TODO : Make interface in loop or raise error.
                
        # If data isn't digitalized :
        
        for col in self.X:
            if self.X[col].dtype == 'object':
                while True:
                    choice = input(
                        'X values are\'nt numerical, do you want to convert them? (y/n)\n')
                    if choice.upper() == 'Y':
                        # Digitalize X data (dummies)
                        num_df = self.X.select_dtypes(exclude="object")
                        str_df = self.X.select_dtypes(include="object")
                        str_df = pd.get_dummies(str_df, drop_first=True)
                        self.X = pd.concat([num_df, str_df], axis=1)
                        print('\nData has been digitalized '
                              'and is ready for evaluation.\n')
                        return
                    if choice.upper() == 'N':
                        print('Model selection can\'t be performed.')
                        exit(1)
                    else:
                        print('Incorrect input.')

        print('\nData ready for evaluation.')

    def data_report(self):
        """Plots information about the data as a whole.

        This method shows various plots visualizing several aspects of the data
        depending on what model type the data is for.

        Shared plots:
        -   heatmap
        -   pairplot (columns < 5)
        Regression plots:
        -   histplot
        Classification:
        -   countplot

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if self.regressor:
            sns.histplot(
                data=self.data,
                x=pd.DataFrame(
                    self.y).columns[0],
                bins=25,
                kde=True)
            plt.title(f'{pd.DataFrame(self.y).columns[0]} distribution')
            plt.show()

        else:  # If classification
            sns.countplot(data=self.data, x=self.y)
            plt.title('Target Distribution')
            plt.show()

        if len(list(self.X.columns)) < 5:  # Avoid giant plots
            sns.pairplot(data=self.data)
            plt.show()

        sns.heatmap(data=self.X.corr())
        plt.title('Correlation Between X Features')
        plt.xticks(rotation=30)
        plt.show()

    def choose_model(self) -> None:  # Or exit program
        """Assigns model type (regression/classification) depending on user input.

        This model evaluates the model type the user wants. If user decides on
        regression-type, it checks if the target values are numerical, and offers
        to digitalize them if so. Otherwise quits the program.

        Parameters
        ----------
        None

        Returns
        -------
        None
        (Or exit)

        Notes:
        The method cannot identify if a df should be used for a regression instead
        of classification. As such a BUG can occur if user selects classification
        when there should be a regression model.

        """
        while True:
            choice = input('What kind of model are you looking for ? \n'
                        'Regressor (r) Classifier (c)\n')

            if 'C' == choice.upper(): # TODO : Fix potential bug^ with validation.
                self.regressor = False
                break

            if 'R' == choice.upper():

                if self.y.dtype == 'object':
                    print('Regression cannot be performed on data '
                        'with object type y values.')
                    change_choice = input(
                        'Do you want a classifier model instead?\n')
                    if change_choice.upper() == 'Y':
                        print('Model type changed to Classifier')
                        self.regressor = False
                        break
                    else:
                        print('Conversion is needed to proceed. Return with other/altered data')
                        exit()  # TODO : Raise error with message.
                else:
                    self.regressor = True
                    break

            print('Invalid choice')

    def confirm_model_choice(self, chosen_model: GridSearchCV) -> None:  # Or exit program
        """Confirms if user agrees on generated model suggestion and if they want it saved.

        If the user does not agree/does not want the suggested (trained) model to be saved,
        the program ends.

        Parameters
        ----------
        chosen_model : sklearn.model_selection.GridSearchCV
            The model suggested by the program.

        Returns
        -------
        None
        (or exit)

        """
        while True:
            choice = input('Do you agree with the model choice '
                           'and want to save it ? (y/n)\n')
            if choice.upper() == 'Y':
                self.save_model(chosen_model)
                break

            elif choice.upper() == 'N':
                exit(1)  # TODO : Raise error with message.
            else:
                print('Invalid input.')

    def preprocess(self) -> tuple:
        """Prepares data to be trained and tested.

        This model splits the data for testing/training and performs Scaling on it.
        If the model is a regressor, Polynomial (3rd degree) regression is also applied.

        Parameters
        ----------
        None

        Returns
        -------
        x_train : pandas.DataFrame
            The X feature matrix to train with.
        x_test : pandas.DataFrame
            The X feature matrix to test with.
        y_train : pandas.Series
            The y target values to train with.
        y_test : pandas.Series
            The y target values to test with.

        """

        print('Calculating...')

        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.model_selection import train_test_split


        # Polynomial Regression
        if self.regressor:
            polynomial_converter = PolynomialFeatures(
                degree=3, include_bias=False)
            self.X = polynomial_converter.fit_transform(self.X)

        # Splitting
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=101)

        # Scaling + fitting
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_model(chosen_model: GridSearchCV) -> exit:
        """Saves a trained model as a .joblib file. Static.

        The user chooses a filename for the model they want saved, and it gets saved.

        Parameters
        ----------
        chosen_model : sklearn.model_selection.GridSearchCV
            The model that will be saved  in the .joblib file

        Returns
        -------
            exit

        """
        from joblib import dump

        while True:
            filename = input('Choose filename (.joblib type) :\n')
            if filename[-7:] == '.joblib':
                dump(chosen_model, filename)
                break
            print('Invalid filename. Must end with \'.joblib\'')
        sys.exit()

    @staticmethod
    def grid_model(base_model,
                   param_grid: dict,
                   X_train: pd.DataFrame,
                   y_train: pd.Series) -> GridSearchCV:
        """Tunes hyperparameters of passed model using  GridSearhCV. Static.

        This model uses GridSearchCV from sklearn to tune the hyper parameters of
        a passed model. I then fits the data data to the model. It uses 10 as
        Kfold value.

        Parameters
        ----------
        base_model : LinearRegression, Ridge, LassoCV, ElasticNetCV,
        KNeighborsClassifier, LogisticRegression, SVC
            The model that will be used as the estimator, ie. of which
            hyperparametes are tuned.
        param_grid : dict
            The dictionary containing hyperparameters to tune and their options.
        x_train : pandas.DataFrame
            The X feature training matrix, the data that will train the model.
        y_train: pandas.Series
            THe y target training values, the data that will train the mmodel.

        Returns
        -------
        sklearn.model_selection.GridSeachrCV
            The trained GridSearchCV model with the passed model as estimator.

        """

        from sklearn.model_selection import GridSearchCV

        grid_model = GridSearchCV(estimator=base_model,
                                  param_grid=param_grid,
                                  cv=10)
        grid_model.fit(X_train, y_train)

        return grid_model

    def get_rmse(self, model) -> int:
        """Genrates rmse metric for passed model.

        Parameters
        ----------
        model : Any type of trained regressor model.
            The model to get rmse for.

        Returns
        -------
        int
            The rmse value.

        """
        from sklearn.metrics import mean_squared_error
        test_predictions = model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))
        return rmse

    def get_mae(self, model) -> int:
        """Genrates mae metric for passed model.

        Parameters
        ----------
        model : Any type of trained regressor model.
            The model to get mae for.

        Returns
        -------
        int
            The mae value.

        """
        from sklearn.metrics import mean_absolute_error
        test_predictions = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, test_predictions)
        return mae

    def get_r2_score(self, model) -> int:
        """Genrates r2 score metric for passed model.

        Parameters
        ----------
        model : Any type of trained regressor model.
            The model to get r2 score for.

        Returns
        -------
        int
            The r2 score value.

        """

        # .score and r2_score gets same result
        return model.score(self.X_test, self.y_test)

    def regression_report(self, model_types: list) -> None:
        """Prints a selection of metrics for every element in a list of
        Regressor models.

        This method takes a list of Regression models, and prints mae,
        and r2 score values as well as the best parameters for each model.

        Parameters
        ----------
        model_types : list
            The list of regressor models to report on.

        Returns
        -------
        None

        """
        for model in model_types:
            print(f'{model.estimator}\n\
            mae: {self.get_mae(model)}\n\
            rmse: {self.get_rmse(model)}\n\
            r2score: {self.get_r2_score(model)}\n\
            best parameters: {model.best_params_}\n')

    def calc_ideal_regression_model(self) -> None:
        """Calculates the best suited regression model for the data.

        This method creates and fits a model for each avilable regression model type
        (LinearRegression, Ridge, LassoCV, ElasticNetCV, SVR) with their
        hyperparameters tuned. It reports the metrics for each model and calculates
        the best fitting model based on rmse and r2 score, sugessting it to the user.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes:
        FIXME: This method takes a long time to run

        """
        from sklearn.linear_model import LinearRegression, \
            Ridge, \
            LassoCV, \
            ElasticNetCV
        from sklearn.svm import SVR
        # Dict of each param grid for every model type
        grids = {
            'linear': {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            },

            'ridge': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },

            'lasso': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'eps': [0.1],
                'random_state': [42]
            },

            'elasticnet': {
                'fit_intercept': [True, False],
                'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
                'copy_X': [True, False],
                'eps': [0.1],
                'random_state': [42]
            },

            'svr': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto']
            }
        }


        # Creating models through GridSearchCV
        linear_model = self.grid_model(LinearRegression(), grids['linear'],
                                       self.X_train, self.y_train)
        ridge_model = self.grid_model(Ridge(), grids['ridge'],
                                      self.X_train, self.y_train)
        lasso_model = self.grid_model(
            LassoCV(eps=0.1, n_alphas=100),
            grids['lasso'],
            self.X_train,
            self.y_train)
        elasticnet_model = self.grid_model(
            ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], tol=0.01),
            grids['elasticnet'],
            self.X_train,
            self.y_train)
        svr_model = self.grid_model(
            SVR(),
            grids['svr'],
            self.X_train,
            self.y_train)

        # Reporting model information
        models = [linear_model,
                  ridge_model,
                  lasso_model,
                  elasticnet_model,
                  svr_model]
        self.regression_report(models)

        # Finding right model
        # We want a low rmse and a high r2 score. to take both into account,
        # we can subtract the r2-score from rmse and then find the lowest nr as
        # the best.
        score_list = [
            self.get_rmse(linear_model) -
            self.get_r2_score(linear_model),
            self.get_rmse(ridge_model) -
            self.get_r2_score(ridge_model),
            self.get_rmse(lasso_model) -
            self.get_r2_score(lasso_model),
            self.get_rmse(elasticnet_model) -
            self.get_r2_score(elasticnet_model),
            self.get_rmse(svr_model) -
            self.get_r2_score(svr_model)]

        # Print decision based on score
        for i, scores in enumerate(score_list, start=0):
            if min(score_list) == scores:
                print(f'\n{models[i].estimator} is recommended.')
                self.confirm_model_choice(models[i])
                break

    def classification_reports(self, model, show=False):
        """Prints a selection of metrics for a given classification model.

        This method takes a classification model and prints mae, and r2 score values
        as well as the best parameters for each model.

        Parameters
        ----------
        model_types : list
            The list of regressor models to report on.

        Returns
        -------
        None

        Notes:
        TODO : Rename this method to avoid confusion with
               sklearn.metrics.classification_report.

        """
        from sklearn.metrics import classification_report, \
        ConfusionMatrixDisplay, \
        accuracy_score
        # Test the models
        y_pred = model.predict(self.X_test)

        # Show general classification information
        print(f'\n\nModel: {model.estimator}')
        print(classification_report(y_true=self.y_test, y_pred=y_pred))

        # Create confusion matrix and show if desired
        ConfusionMatrixDisplay.from_estimator(model, self.X_test, self.y_test)
        plt.title(f'{model.estimator}')
        if show:
            plt.show()

        # Show accuracy and best parameters
        acc = round(accuracy_score(y_true=self.y_test, y_pred=y_pred), 2)
        print('Model Accuracy : ', acc)
        print('Best Parameters : ', model.best_params_, '\n')

        return acc  # Return accuracy for judging

    def calc_ideal_classification_model(self) -> None:
        """Calculates the best suited classification model for the data.

        This method creates and fits a model for each avilable classification model type
        (LogisticRegression, KNeighborsClassifier, SVC) with their
        hyperparameters tuned. It reports the metrics for each model and calculates
        the best fitting model based on their accuracy, sugessting it to the user.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        # Create individual param grids for each type of model :
        grids = {'logistic': {'fit_intercept': [True, False],
                              'solver': ['lbfgs', 'liblinear', 'saga'],
                              'random_state': [42, 101]},

                 'knn': {'weights': ['uniform', 'distance'],
                         'algorithm': ['auto', 'ball_tree', 'kd_tree']},

                 'svc': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'gamma': ['scale', 'auto']}}

        # Create model with Gridsearch for each type :
        logistic_model = self.grid_model(
            LogisticRegression(),
            grids['logistic'],
            self.X_train,
            self.y_train)
        knn_model = self.grid_model(
            KNeighborsClassifier(),
            grids['knn'],
            self.X_train,
            self.y_train)
        svc_model = self.grid_model(
            SVC(),
            grids['svc'],
            self.X_train,
            self.y_train)

        models = [logistic_model, knn_model, svc_model]

        # Collect accuracies and make suggestion based of them :
        scores = []
        for i, model in enumerate(models, start=0):
            scores.append(self.classification_reports(model))

        for i, model in enumerate(models, start=0):
            if max(scores) == scores[i]:
                print(f'\n{model.estimator} is recommended.')
                self.confirm_model_choice(model)
                break
