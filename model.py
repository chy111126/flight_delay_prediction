# Model training functions
# Author: Tom Chau

# Import libraries
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from feature_encoder import encode_categorical_feature
from utils import print_metrics, get_regression_q1_error, get_regression_q2_error, get_classification_q1_error, get_classification_q2_error


def train_model(input_file, encode_feature_dict, training_set_output, testing_set_output, classifier_model_output, regressor_model_output):

    # Load dataset, and split into training/testing set
    data_df = pd.read_csv(input_file)
    delay_time_and_count_feats = [col_str for col_str in list(data_df.columns) if col_str.endswith('_count') or col_str.endswith('_delay_time')]
    cat_feats = ['flight_no', 'Week', 'Departure','Arrival','Airline','std_hour']
    all_feats = cat_feats + delay_time_and_count_feats

    # Get x/y columns
    x_all = data_df[all_feats]
    y_all = data_df['is_claim']

    # Split training/testing set
    train, test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

    # Save training/testing set into files first
    train_output = train.copy()
    train_output['is_claim'] = y_train
    train_output.to_csv(training_set_output)

    test_output = train.copy()
    test_output['is_claim'] = y_test
    train_output.to_csv(testing_set_output)

    # Translate dataframe categorical columns for training
    encoded_train = encode_categorical_feature(train, encode_feature_dict)
    encoded_test = encode_categorical_feature(test, encode_feature_dict)

    # Train model with classifier
    def set_is_claim(val):
        if val == 0:
            return 0
        else:
            return 1
    y_train_bool = y_train.apply(set_is_claim)
    y_test_bool = y_test.apply(set_is_claim)

    # Model training part
    # Classifier model
    print("Start training classification model ...")
    d_train = lgb.Dataset(encoded_train, label=y_train_bool)
    params = {
        "objective": "binary",
        "max_depth": 50,
        "learning_rate" : 0.1,
        "num_leaves": 900,
        "n_estimators": 300
    }
    model_feat_classifier = lgb.train(params, d_train, categorical_feature=cat_feats)

    # Predict both training/testing set and show result
    training_y_pred = model_feat_classifier.predict(encoded_train).round()
    testing_y_pred = model_feat_classifier.predict(encoded_test).round()

    print()
    print()
    print()
    print("----- Classifier model result -----")
    print("----- Training set -----")
    print_metrics(y_train_bool, training_y_pred)
    print("Q1 error:", get_classification_q1_error(y_train_bool, training_y_pred))
    print("Q2 error:", get_classification_q2_error(y_train_bool, training_y_pred))
    print("----- Testing set -----")
    print_metrics(y_test_bool, testing_y_pred)
    print("Q1 error:", get_classification_q1_error(y_test_bool, testing_y_pred))
    print("Q2 error:", get_classification_q2_error(y_test_bool, testing_y_pred))
    print()
    print()
    print()

    # Save model
    model_feat_classifier.save_model(classifier_model_output)

    # Regressor model
    print("Start training regression model ...")
    d_train = lgb.Dataset(encoded_train, label=y_train)
    params = {
        "objective": "regression",
        "max_depth": 50,
        "learning_rate" : 0.1,
        "num_leaves": 900,
        "n_estimators": 300
    }
    model_feat_regressor = lgb.train(params, d_train, categorical_feature = cat_feats)

    # Predict both training/testing set and show result
    training_y_pred = model_feat_regressor.predict(encoded_train)
    testing_y_pred = model_feat_regressor.predict(encoded_test)

    # For regression, clamp the result into range [0, 800]
    np.clip(training_y_pred, 0, 800, out=training_y_pred)
    np.clip(testing_y_pred, 0, 800, out=testing_y_pred)

    print()
    print()
    print()
    print("----- Regressor model result -----")
    print("----- Training set -----")
    print("Q1 error:", get_regression_q1_error(y_train, training_y_pred))
    print("Q2 error:", get_regression_q2_error(y_train, training_y_pred))
    print("----- Testing set -----")
    print("Q1 error:", get_regression_q1_error(y_test, testing_y_pred))
    print("Q2 error:", get_regression_q2_error(y_test, testing_y_pred))
    print()
    print()
    print()

    # Save model
    model_feat_regressor.save_model(regressor_model_output)
    return


def test_model(testing_set_input, encode_feature_dict, classifier_model, regressor_model, result_output):

    # Load testing set
    data_df = pd.read_csv(testing_set_input)
    delay_time_and_count_feats = [col_str for col_str in list(data_df.columns) if col_str.endswith('_count') or col_str.endswith('_delay_time')]
    cat_feats = ['flight_no', 'Week', 'Departure','Arrival','Airline','std_hour']
    all_feats = cat_feats + delay_time_and_count_feats

    # Get x columns
    x_all = data_df[all_feats]

    # Translate dataframe categorical columns for training
    encoded_x = encode_categorical_feature(x_all, encode_feature_dict)

    # Model prediction part
    # Classifier model
    print("Start prediction with classification model ...")
    model_feat_classifier = lgb.Booster(model_file=classifier_model)

    # Predict testing set
    y_pred = model_feat_classifier.predict(encoded_x).round()

    # Append prediction result
    x_all['classifier_y_pred'] = y_pred

    # Regressor model
    print("Start prediction with regression model ...")
    model_feat_regressor = lgb.Booster(model_file=regressor_model)

    # Predict testing set
    y_pred = model_feat_regressor.predict(encoded_x).round()

    # For regression, clamp the result into range [0, 800]
    np.clip(y_pred, 0, 800, out=y_pred)

    # Append prediction result
    x_all['regressor_y_pred'] = y_pred

    # Save result
    x_all.to_csv(result_output)
    return

