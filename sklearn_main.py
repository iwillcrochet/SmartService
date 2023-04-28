import math
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import pickle
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,RepeatedKFold, ParameterGrid
from sklearn.metrics import r2_score
from sklearn_helper import adjusted_r2

def main():
    RANDOM_STATE = 42
    FULL_GRID_SEARCH = False
    RANDOM_FRACTION = 0.001
    N_SPLITS = 3
    N_REPEATS = 2

    # fetch data
    from data_preparation import prepare_data
    X_train, X_test, y_train, y_test, feature_columns, scaler, imputer = prepare_data(pytorch=False)

    #########################################
    # Feature Importance
    #########################################
    print("----------------------------------------")
    print("Calculating feature importance...")

    # define regressor
    regressor = RandomForestRegressor(n_estimators=100,
                                      random_state=RANDOM_STATE)

    # specify pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("imputer", imputer),
        ("regressor", regressor)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model, using MSE and MAE as metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    # print train and test errors
    print("Regressor: Random Forest")
    print(f"Train - RMSE: {math.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, MAE: {mean_absolute_error(y_train,  y_train_pred):.4f}")
    print(f"Test - RMSE: {math.sqrt(mean_squared_error(y_test, y_pred)):.4f}, MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    # Plot feature importances
    clf = pipeline.named_steps['regressor']
    from sklearn_helper import plot_feature_importances
    plot_feature_importances(clf, pd.DataFrame(X_train, columns=feature_columns), y_train, top_n=20, print_table=True)

    #############################
    # Grid Search
    #############################
    # Define parameters for each regressor
    params_RF = {
        'regressor': [RandomForestRegressor(random_state=RANDOM_STATE)],
        'regressor__n_estimators': [50, 70, 100, 200, 500, 1000],
        'regressor__max_depth': [1, 3, 5, 7, 9, 15, None],
        'regressor__max_features': [1.0, 1, 3, 5, 7, 15, 30],
        'regressor__max_leaf_nodes': [2, 3, 5, 9, 13, 20, 50, None],
    }

    params_XGB = {
        'regressor': [XGBRegressor(random_state=RANDOM_STATE)],
        'regressor__n_estimators': [50, 100, 200, 400, 800],
        'regressor__max_depth': [1, 3, 5, 7, 9, 11, 13],
        'regressor__booster': ['gbtree'],
        'regressor__gamma': [0, 0.1, 0.5, 1],
        'regressor__reg_alpha': [0, 0.1, 0.5, 1],
        'regressor__reg_lambda': [0, 0.1, 0.5, 1],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
    }

    params_LGBM = {
        'regressor': [LGBMRegressor(random_state=RANDOM_STATE)],
        'regressor__n_estimators': [50, 100, 200, 400, 800],
        'regressor__max_depth': [-1, 3, 9, 13, 21, 32, 50, 100],
        'regressor__boosting_type': ['gbdt'],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__num_leaves': [2, 5, 9, 13, 26, 39, 50, 100],
        'regressor__reg_alpha': [0, 0.1, 0.3, 0.5, 0.7, 1],
        'regressor__reg_lambda': [0, 0.1, 0.3, 0.5, 0.7, 1],
    }

    # Put params into a list
    params_list = [params_RF, params_XGB, params_LGBM]

    best_models = {}

    print("-----------------------------")
    print("Performing grid search...")
    print("-----------------------------")

    best_parameters = {}

    # Loop over every regressor in the list
    for params in params_list:
        start_time = time.time()

        # Create a grid of parameters
        grid = ParameterGrid(params)
        n_iterations = int(len(grid) * RANDOM_FRACTION)
        print(f"{params['regressor'][0].__class__.__name__}, "
              f"Possible parameter combinations: {len(grid)}, "
              f"Search iterations: {n_iterations}")

        # Perform grid search or random search
        cv_method = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
        scoring_metric = "neg_mean_squared_error"
        n_jobs = 4

        # Removed TqdmRandomizedSearchCV and TqdmGridSearchCV
        if FULL_GRID_SEARCH == False:
            search = RandomizedSearchCV(pipeline, params, cv=cv_method, n_jobs=n_jobs, scoring=scoring_metric,
                                        verbose=0, n_iter=n_iterations, random_state=RANDOM_STATE)
        else:
            search = GridSearchCV(pipeline, params, cv=cv_method, n_jobs=n_jobs, scoring=scoring_metric, verbose=0)

        search.fit(X_train, y_train)

        # Calculate the time taken to perform the search
        elapsed_time = time.time() - start_time

        # make predictions
        best_model = search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Store the best models
        best_models[search.best_estimator_.named_steps['regressor'].__class__.__name__] = search.best_estimator_

        # Save model parameters to a dict, except for the last model parameter in the search.best_params_ dict
        best_parameters[search.best_estimator_.named_steps['regressor'].__class__.__name__] = \
            {k: v for i, (k, v) in enumerate(search.best_params_.items()) if i < len(search.best_params_) - 1}

        # print model performance
        print(f"Best model performance: "
              f"Train RMSE: {math.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, "
              f"Test RSME: {math.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}, "
              f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Best model parameters: {best_parameters[search.best_estimator_.named_steps['regressor'].__class__.__name__]}")
        print("-------------------------------------------")

    # create a dataframe to store the results
    results_df = pd.DataFrame(columns=['model', 'train_rmse', 'test_rmse', 'test_mae', 'r2_train', 'r2_test',
                                       'adj_r2_train', 'adj_r2_test','parameters'])

    # Compare the performance of all models and store results in the dataframe
    for model_name, model in best_models.items():
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)

        # Additional evaluation metrics: R-squared and Adjusted R-squared
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_pred)
        adj_r2_train = adjusted_r2(r2_train, X_train.shape[0], X_train.shape[1])
        adj_r2_test = adjusted_r2(r2_test, X_test.shape[0], X_test.shape[1])

        new_row = {'model': model_name,
                   'train_rmse': train_rmse,
                   'test_rmse': test_rmse,
                   'test_mae': test_mae,
                   'r2_train': r2_train,
                   'r2_test': r2_test,
                   'adj_r2_train': adj_r2_train,
                   'adj_r2_test': adj_r2_test,
                   'parameters': best_parameters[model_name]}

        # Use .loc[] to add a new row to the DataFrame
        results_df.loc[len(results_df)] = new_row

    # sort the dataframe by test_rmse
    results_df = results_df.sort_values(by='test_rmse')

    # save df to csv, save best_models to pickle
    cwd = os.getcwd()
    model_folder = os.path.join(cwd, 'models')
    results_df.to_csv(os.path.join(model_folder, 'sklearn_best_models_results.csv'), index=False)
    with open(os.path.join(model_folder, 'sklearn_best_models.pickle'), 'wb') as handle:
        pickle.dump(best_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ############################
    # Evaluate the best models
    ############################
    # plot a bar chart of the test RMSE for each model
    plt.figure(figsize=(10, 6))  # Increase the figure size
    plt.bar(results_df['model'], results_df['test_rmse'])
    plt.xlabel('Models')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE for the Best Regressors')
    plt.xticks(rotation=45, ha='right')  # Adjust the rotation and horizontal alignment
    plt.tight_layout()  # Adjust the layout to fit labels
    plt.show()

    # create a learning curve for each model
    from sklearn_helper import plot_learning_curve
    for model_name, model in best_models.items():
        plot_learning_curve(model, X_train, y_train, title=f"Learning Curve for {model_name}")
        print(f"Parameters for {model_name}: {best_parameters[model_name]}")

if __name__ == "__main__":
    main()