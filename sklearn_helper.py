import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve

def plot_feature_importances(clf, X_train, y_train=None,
                             top_n=10, figsize=(6, 6), print_table=False, title="Feature Importances"):
        '''
        plot feature importances of a tree-based sklearn estimator

        Note: X_train and y_train are pandas DataFrames

        Note: Scikit-plot is a lovely package but I sometimes have issues
                  1. flexibility/extendibility
                  2. complicated models/datasets
              But for many situations Scikit-plot is the way to go
              see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

        Parameters
        ----------
            clf         (sklearn estimator) if not fitted, this routine will fit it

            X_train     (pandas DataFrame)

            y_train     (pandas DataFrame)  optional
                                            required only if clf has not already been fitted

            top_n       (int)               Plot the top_n most-important features
                                            Default: 10

            figsize     ((int,int))         The physical size of the plot
                                            Default: (8,8)

            print_table (boolean)           If True, print out the table of feature importances
                                            Default: False

        Returns
        -------
            the pandas dataframe with the features and their importance

        Author
        ------
            George Fisher
        '''

        __name__ = "plot_feature_importances"

        from xgboost.core import XGBoostError
        from lightgbm.sklearn import LightGBMError

        try:
            if not hasattr(clf, 'feature_importances_'):
                clf.fit(X_train.values, y_train.values.ravel())

                if not hasattr(clf, 'feature_importances_'):
                    raise AttributeError("{} does not have feature_importances_ attribute".
                                         format(clf.__class__.__name__))

        except (XGBoostError, LightGBMError, ValueError):
            clf.fit(X_train.values, y_train.values.ravel())

        feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
        feat_imp['feature'] = X_train.columns
        feat_imp.sort_values(by='importance', ascending=False, inplace=True)
        feat_imp = feat_imp.iloc[:top_n]

        feat_imp.sort_values(by='importance', inplace=True)
        feat_imp = feat_imp.set_index('feature', drop=True)
        feat_imp.plot.barh(title=title, figsize=(10, 6))  # Increase the figure size
        plt.xlabel('Feature Importance Score')
        plt.yticks(rotation=45, ha='right')  # Adjust the rotation and horizontal alignment
        plt.tight_layout()  # Adjust the layout to fit labels
        plt.show()

        if print_table:
            from IPython.display import display
            print("Top {} features in descending order of importance".format(top_n))
            display(feat_imp.sort_values(by='importance', ascending=False))

        return feat_imp

def plot_learning_curve(model, X_train, y_train, title, ylim=None):
    """plot the learning curves for each best model"""
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    train_scores_mean = np.sqrt(-np.mean(train_scores, axis=1))
    test_scores_mean = np.sqrt(-np.mean(test_scores, axis=1))

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(loc="best")
    plt.show()

def adjusted_r2(r2, n, p):
    """Function to calculate adjusted R-squared"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)