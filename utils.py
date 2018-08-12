from sklearn.metrics import confusion_matrix, accuracy_score


def get_regression_q1_error(y_test, y_pred):
    pass


def get_regression_q2_error(y_test, y_pred):
    pass


def get_classification_q1_error(y_test, y_pred):
    pass


def get_classification_q2_error(y_test, y_pred):
    pass


def get_metrics(to_check_y, to_check_pred):
    # We check the score of set result in two ways
    # Confusion matrix
    cm = confusion_matrix(to_check_y, to_check_pred)
    # Accuracy
    accuracy = accuracy_score(to_check_pred,to_check_y)

    return cm, accuracy

def print_metrics(to_check_y, to_check_pred):
    cm, accuracy = get_metrics(to_check_y, to_check_pred)

    print("Confusion matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Delay/Cancel case accuracy:", cm[1][1] / sum(cm[1]))