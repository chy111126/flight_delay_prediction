# flight_delay_prediction
Flight Delay Prediction for Data Science Assessment

Commands to execute:
python main.py preprocess --input=./datasets/flight_delays_data.csv --output=./datasets/flight_delays_data_transformed.csv

python main.py encode_features --input=./datasets/flight_delays_data_transformed.csv --output=./models/encode_features_dict.pkl

python main.py train_model --input=./datasets/flight_delays_data_transformed.csv --encode_feature_dict=./models/encode_features_dict.pkl --training_set_output=./datasets/training_set.csv --testing_set_output=./datasets/testing_set.csv --classifier_model_output=./models/classifier.txt --regressor_model_output=./models/regressor.txt

python main.py predict --input=./datasets/testing_set.csv --encode_feature_dict=./models/encode_features_dict.pkl --classifier_model=./models/classifier.txt --regressor_model=./models/regressor.txt --result_output=./prediction_result.csv

Required libraries:
lightgbm
numpy
pandas
scikit-learn

Libraries for notebook:
matplotlib
seaborn
