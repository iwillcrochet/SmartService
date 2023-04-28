import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def prepare_data(pytorch=True):
    RANDOM_STATE = 42
    FILE_NAME = "Kaggle_Hosue_Price.csv"
    TARGET_COL = 'Saleprice'
    MASTER_KEY = "Id"
    EXCLUDE_COLS = [None]
    SCALE = True
    IMPUTE = True

    df = pd.read_csv(FILE_NAME)

    y = df[TARGET_COL]

    df.set_index(MASTER_KEY, inplace=True)

    if EXCLUDE_COLS[0] is not None:
        EXCLUDE_COLS = EXCLUDE_COLS + [TARGET_COL]
    else:
        EXCLUDE_COLS = [TARGET_COL]
    X = df.drop(EXCLUDE_COLS, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)

    # Unit Test - data leakage: Check the indices of the train and test data after the train_test_split function to ensure that there is no overlap between them
    assert len(
        set(X_train.index).intersection(
            set(X_test.index))) == 0, "Data leakage detected: Train and test sets have common ids."

    # print dataset information
    print(f"Feature names: {X.columns.to_list()}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")

    # impute missing values
    imputer = KNNImputer(n_neighbors=5)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    if pytorch:
        # scale the data
        if SCALE:
            # scale the data
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print(f"Data scaled to mean 0 and std 1.")

        if IMPUTE:
            # impute missing values
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            # print nubmer of imputed values
            print("Number of imputed values in training set: ", pd.DataFrame(X_train).isna().sum().sum())
            print("Number of imputed values in testing set: ", pd.DataFrame(X_test).isna().sum().sum())

        return X_train, X_test, y_train, y_test, X.columns

    else:
        # return the data, scaler, and imputer
        return X_train, X_test, y_train, y_test, X.columns, scaler, imputer