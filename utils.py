from sklearn.metrics import confusion_matrix, accuracy_score


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


def get_regression_q1_error(y_test, y_pred):
    # Regression Q1 error
    return sum(abs(y_pred - y_test)) / len(y_test)


def get_regression_q2_error(y_test, y_pred):
    # Regression Q2 error
    return sum(abs(y_pred - y_test) ** 2) / len(y_test)


def get_classification_q1_error(y_test, y_pred):
    # For Q1/Q2 error of classification-based model, it is simply done by multiplying true negative/false positive result by claim amount
    cm, _ = get_metrics(y_test, y_pred)

    # Q1 error
    return (cm[0][1] + cm[1][0]) * 800 / len(y_test)


def get_classification_q2_error(y_test, y_pred):
    # For Q1/Q2 error of classification-based model, it is simply done by multiplying true negative/false positive result by claim amount
    cm, _ = get_metrics(y_test, y_pred)

    # Q2 error
    return (cm[0][1] + cm[1][0]) * (800 ** 2) / len(y_test)
