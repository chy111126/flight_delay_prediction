
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# Parse arguments
all_args = [s for s in sys.argv if s.startswith('--')]
args_dict = dict()
for astr in all_args:
    astr_list = astr.split("=")
    args_dict[astr_list[0][2:]] = astr_list[1]

# Run argument-specified routine
if sys.argv[1] == "preprocess":

    # Preprocess routine
    from preprocess import preprocess_file
    preprocess_file(args_dict['input'], args_dict['output'])

elif sys.argv[1] == "encode_features":

    # Build dict to encode features into labels
    from feature_encoder import build_categorical_feature_dict
    build_categorical_feature_dict(args_dict['input'], args_dict['output'])

elif sys.argv[1] == "train_model":

    # Model training routine
    from model import train_model
    train_model(args_dict['input'], args_dict['encode_feature_dict'],
                args_dict['training_set_output'], args_dict['testing_set_output'],
                args_dict['classifier_model_output'], args_dict['regressor_model_output'])

elif sys.argv[1] == "predict":

    # Model prediction routine
    # Testing set should be used here
    from preprocess import preprocess_file
    preprocess_file(args_dict['input'], args_dict['output'])

