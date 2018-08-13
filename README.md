# flight_delay_prediction
Flight Delay Prediction for Data Science Assessment

The presentation material is uploaded to project root (presentation.pdf) for your reference. Also, more in-depth studies regarding the assessment can be seen under notebooks/ folder in Jupyter notebook format.


### Dependencies
------

This project is developed with Python 3.6.
To run, make sure below libraries are installed:

<br />
<span><b>- Numpy </b><pre>pip3 install numpy </pre></span>
<span><b>- Pandas </b><pre>pip3 install pandas </pre> </span>
<span><b>- SciPy </b><pre>pip install scipy </pre> </span>
<span><b>- scikit-learn </b><pre>pip install scikit-learn </pre> </span>
<span><b>- LightGBM </b><pre>Please follow the link at: [LightGBM/Installation-Guide](https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst) </pre> </span>
<br />

For files under notebooks/ folder, the following libraries are required:

<br />
<span><b>- Jupyter notebook </b><pre>pip install jupyter </pre></span> 
<span><b>- Matplotlib </b><pre>pip install matplotlib </pre> </span>
<span><b>- Seaborn </b><pre>pip install seaborn </pre> </span>
<br />

It is recommended to use Anaconda environment for simplified project setup: [Downloads - Anaconda](https://www.anaconda.com/download/)


### Running the program
------

To run the program with pre-trained models, make sure the model files exist under models/ folder, then execute as follows:

<span><b>1. Preprocess the dataset (for calculating time bin metrics)</b><pre>python main.py preprocess --input=./datasets/flight_delays_data.csv --output=./datasets/to_predict_set.csv</pre></span>

<span><b>2. Run model prediction</b><pre>python main.py predict --input=./datasets/to_predict_set.csv --encode_feature_dict=./models/encode_features_dict.pkl --classifier_model=./models/classifier.txt --regressor_model=./models/regressor.txt --result_output=./prediction_result.csv</pre></span>

<br />
To train the model from scratch, use the following steps:

<span><b>1. Preprocess the dataset (for calculating time bin metrics)</b><pre>python main.py preprocess --input=./datasets/flight_delays_data.csv --output=./datasets/to_predict_set.csv</pre></span>

<span><b>2. Get label encoder dictionary for categorical features</b><pre>python main.py encode_features --input=./datasets/flight_delays_data_transformed.csv --output=./models/encode_features_dict.pkl</pre></span>

<span><b>3. Train model</b><pre>python main.py train_model --input=./datasets/flight_delays_data_transformed.csv --encode_feature_dict=./models/encode_features_dict.pkl --training_set_output=./datasets/training_set.csv --testing_set_output=./datasets/testing_set.csv --classifier_model_output=./models/classifier.txt --regressor_model_output=./models/regressor.txt</pre></span>

<span><b>4. Run model prediction</b><pre>python main.py predict --input=./datasets/testing_set.csv --encode_feature_dict=./models/encode_features_dict.pkl --classifier_model=./models/classifier.txt --regressor_model=./models/regressor.txt --result_output=./prediction_result.csv</pre></span>

