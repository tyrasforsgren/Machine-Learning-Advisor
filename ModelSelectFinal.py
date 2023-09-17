import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelSelection:
    """
    Suggests an ML model to user based on input dataframe.
    Parameters :
        path - path to dataframe

    Reads csv to run test on.
    Prepares data.
    Reads desired ML type.
    Tries different models.
    Shows the results of different models.
    Suggests ideal model based on results.
    Saves model.
    """

    def __init__(self, path) -> None:
        self.regressor = None
        self.data = pd.read_csv(path)
        self.X, self.y = self.set_values()
        self.complete_data()
        self.choose_model()
        self.data_report()
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()
        if self.regressor:
            self.calc_ideal_regression_model()
        else:
            self.calc_ideal_classification_model()

    def set_values(self):
        """
        Takes desired targets.
        Checks if it is valid.
        Saves target and features.
        """
        while True:
            choice = input(f'Choose dependant value from list :\n'
                           f'{list(self.data.columns)}\n')
            if choice in list(self.data.columns):
                X = pd.DataFrame(self.data.drop(choice, axis=1))
                y = pd.Series(self.data[choice])
                return X, y
            else:
                print('Target not in column list.')

    def complete_data(self):
        """
        Checks werther data is ready for ML model.
        Reports eventual issues.
        Offers to remove rows with missing values.
        Offers to convert non-numerical data.
        Ends program if model selection has become impossible.
        """
        # If there is missing data :
        if self.data.isnull().sum().sum():
            na_choice = input(
                'Data is incomplete. Do you want to drop rows with missing data?\n'
                '(all will be removed.) \n')  # Missing values where?
            if na_choice.upper() == 'Y':
                self.data = self.data.dropna()
                print('Incomplete rows have been deleted. \n'
                      'Dependant value has been reset, input again.\n')
                self.X, self.y = self.set_values()
            else:
                print('Model selection cannot be performed.')
                exit()

        # If data isn't digitalized :
        for col in self.X:
            if self.X[col].dtype == 'object':
                while True:
                    choice = input(
                        'X values are\'nt numerical, do you want to convert them? (y/n)\n')
                    if choice.upper() == 'Y':
                        num_df = self.X.select_dtypes(exclude="object")
                        str_df = self.X.select_dtypes(include="object")
                        str_df = pd.get_dummies(str_df, drop_first=True)
                        self.X = pd.concat([num_df, str_df], axis=1)
                        print('\nData has been digitalized '
                              'and is ready for evaluation.\n')
                        return
                    elif choice.upper() == 'N':
                        print('Model selection can\'t be performed.')
                        exit()
                    else:
                        print('Incorrect input.')
        print('\nData ready for evaluation.')

    def data_report(self):
        """
        Visualizes information about the complete data.
        For regression a histplot.
        For classification a countplot.
        Pairplot if there are only a few X features.
        Heatmap is always shown.
        """
        import seaborn as sns

        if self.regressor:
            sns.histplot(
                data=self.data,
                x=pd.DataFrame(
                    self.y).columns[0],
                bins=25,
                kde=True)
            plt.title(f'{pd.DataFrame(self.y).columns[0]} distribution')
            plt.show()
        else:
            sns.countplot(data=self.data, x=self.y)
            plt.title('Target Distribution')
            plt.show()

        if len(list(self.X.columns)) < 5:
            sns.pairplot(data=self.data)
            plt.show()

        sns.heatmap(data=self.X.corr())
        plt.title('Correlation Between X Features')
        plt.xticks(rotation=30)
        plt.show()

    def choose_model(self):
        """
        Reads desired model type.
        WIP : Evaluates if it is possible.
        WIP - If impossible, offers to change model type.
        """
        while True:
            choice = input('What kind of model are you looking for ? \n'
                           'Regressor (r) Classifier (c)\n')
            if 'C' == choice.upper():
                self.regressor = False
                break
            elif 'R' == choice.upper():
                # Only catches dfs that have not been digitalized.
                if self.y.dtype == 'object':
                    change_choice = input(
                        'Regression cannot be performed on data '
                        'with object type y values. '
                        '\nDo you want a classifier model instead?\n')
                    if change_choice.upper() == 'Y':
                        self.regressor = False
                    else:
                        exit()
                self.regressor = True
                break
            else:
                print('Invalid choice')

    def confirm_model_choice(self, chosen_model):
        """
        Asks if user is happy with recommended model.
        Parameters:
            chosen_model - suggested model
        OBS - chosen_model should be trained on entire df, not training data.
        WIP : Offers to change model.
        """
        while True:
            choice = input('Do you agree with the model choice '
                           'and want to save it ? (y/n)\n')
            if choice.upper() == 'Y':
                self.save_model(chosen_model)
            elif choice.upper() == 'N':
                exit()
            else:
                print('Invalid input.')

    def preprocess(self):
        """
        Preprocess for ML models.
        Performs polynomial regression.
        Performs train test split.
        Performs Scaling.
        """
        print('Calculating...')

        # Polynomial Regression
        if self.regressor:
            from sklearn.preprocessing import PolynomialFeatures
            polynomial_converter = PolynomialFeatures(
                degree=3, include_bias=False)
            self.X = polynomial_converter.fit_transform(self.X)

        # Splitting
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=101)

        # Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_model(chosen_model):
        """
        Saves final model with given filename.
        Parameters:
            chosen_model - model that will be saved.
        OBS - chosen_model should be trained on entire df, not training data.
        """
        from joblib import dump
        while True:
            filename = input('Choose filename (.joblib type) :\n')
            if filename[-7:] == '.joblib':
                dump(chosen_model, filename)
                exit()
            print('Invalid filename. Must end with \'.joblib\'')

    @staticmethod
    def grid_model(base_model, param_grid, X_train, y_train):
        """
        Creates an ML model with ideal parameters by using
        GridSearchCV. Trains and returns this model.
        Parameters:
            base_model - the model type to use
            param_grid - grid of parameters that will be tested
            X_train - independent data to train over
            y_train - dependant value to train over
        """
        from sklearn.model_selection import GridSearchCV

        grid_model = GridSearchCV(estimator=base_model,
                                  param_grid=param_grid,
                                  cv=10)
        grid_model.fit(X_train, y_train)

        return grid_model

    def get_rmse(self, model):
        """
        returns RMSE error for passed model.
        Parameters:
            model - model for which to calculate RMSE
        """
        from sklearn.metrics import mean_squared_error

        test_predictions = model.predict(self.X_test)
        RMSE = np.sqrt(mean_squared_error(self.y_test, test_predictions))
        return RMSE

    def get_mae(self, model):
        """
        returns MAE error for passed model.
        Parameters:
            model - model for which to calculate MAE
        """
        from sklearn.metrics import mean_absolute_error
        test_predictions = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, test_predictions)
        return MAE

    def get_r2_score(self, model):
        """
        returns r2score error for passed model.
        Parameters:
            model - model for which to calculate r2score
        """
        # .score and r2_score gets same result
        return model.score(self.X_test, self.y_test)

    def regression_report(self, model_types):
        """
        Prints a report for each regression model in passed
        list. Report contains MAE, RMSE, r2score and best parameters.
        Parameters:
            model_types - list of models to report about.
        """
        for model in model_types:
            print(f'{model.estimator}\n\
            MAE: {self.get_mae(model)}\n\
            RMSE: {self.get_rmse(model)}\n\
            r2score: {self.get_r2_score(model)}\n\
            best parameters: {model.best_params_}\n')

    def calc_ideal_regression_model(self):
        """
        Creates different types of Regression models; LinearRegression,
        Ridge, LassoCV, ElasticNetCV and SVR. Reports information about them.
        Calculates what model is the best fit for given data.
        """
        from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
        from sklearn.svm import SVR

        # Dict of each param grid for every model type
        grids = {'linear': {'fit_intercept': [True, False],
                            'positive': [True, False],
                            'copy_X': [True, False]},

                 'ridge': {'fit_intercept': [True, False],
                           'copy_X': [True, False]},

                 'lasso': {'fit_intercept': [True, False],
                           'positive': [True, False],
                           'copy_X': [True, False],
                           'random_state': [42, 101]},

                 'elasticnet': {'fit_intercept': [True, False],
                                'positive': [True, False],
                                'copy_X': [True, False],
                                'random_state': [42, 101]},

                 'svr': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'gamma': ['scale', 'auto']}}

        # Creating models through GridSearchCV
        linear_model = self.grid_model(LinearRegression(), grids['linear'],
                                       self.X_train, self.y_train)
        ridge_model = self.grid_model(Ridge(), grids['ridge'],
                                      self.X_train, self.y_train)
        lasso_model = self.grid_model(
            LassoCV(
                eps=0.1,
                n_alphas=100),
            grids['lasso'],
            self.X_train,
            self.y_train)
        elasticnet_model = self.grid_model(ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], tol=0.01),
                                           grids['elasticnet'], self.X_train, self.y_train)
        svr_model = self.grid_model(
            SVR(), grids['svr'], self.X_train, self.y_train)

        # Reporting model information
        models = [linear_model,
                  ridge_model,
                  lasso_model,
                  elasticnet_model,
                  svr_model]
        self.regression_report(models)

        # Finding right model
        # We want a low RMSE and a high r2 score. to take both into account,
        # we can subtract the r2-score from RMSE and then find the lowest nr as
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

    def classification_report(self, model):
        """
        Reports information about classification model.
        Shows classification report, confusion matrix,
        accuracy and best parameters.
        Parameters:
            model - model on which to report on.
        """
        from sklearn.metrics import ConfusionMatrixDisplay, \
            classification_report, \
            accuracy_score

        # Test the models
        y_pred = model.predict(self.X_test)

        # Show classification report and confusion matrix
        print(f'\n\nModel: {model.estimator}')
        print(classification_report(y_true=self.y_test, y_pred=y_pred))
        ConfusionMatrixDisplay.from_estimator(model, self.X_test, self.y_test)
        plt.title(f'{model.estimator}')
        plt.show()

        # Show accuracy and best parameters
        acc = round(accuracy_score(y_true=self.y_test, y_pred=y_pred), 2)
        print('Model Accuracy : ', acc)
        print('Best Parameters : ', model.best_params_, '\n')
        return acc  # Return accuracy for judging

    def calc_ideal_classification_model(self):
        """
        Creates different types of Classification models; LogisticRegression,
        KNN and SVC. Reports information about them.
        Calculates what model is the best fit for given data.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC

        grids = {'logistic': {'fit_intercept': [True, False],
                              'solver': ['lbfgs', 'liblinear', 'saga'],
                              'random_state': [42, 101]},

                 'knn': {'weights': ['uniform', 'distance'],
                         'algorithm': ['auto', 'ball_tree', 'kd_tree']},

                 'svc': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'gamma': ['scale', 'auto']}}

        logistic_model = self.grid_model(
            LogisticRegression(),
            grids['logistic'],
            self.X_train,
            self.y_train)
        KNN_model = self.grid_model(
            KNeighborsClassifier(),
            grids['knn'],
            self.X_train,
            self.y_train)
        SVC_model = self.grid_model(
            SVC(), grids['svc'], self.X_train, self.y_train)

        models = [logistic_model, KNN_model, SVC_model]

        # accuracies will be added to scores and models are judged .
        scores = []
        for i, model in enumerate(models, start=0):
            scores.append(self.classification_report(model))

        for i, model in enumerate(models, start=0):
            if max(scores) == scores[i]:
                print(f'\n{model.estimator} is recommended.')
                self.confirm_model_choice(model)
                break
