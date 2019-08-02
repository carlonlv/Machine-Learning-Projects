import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

dataset = sio.loadmat('dataset.mat')
data_train_X = dataset['data_train_X']
data_train_y = dataset['data_train_y'][0]
data_test_X = dataset['data_test_X']
data_test_y = dataset['data_test_y'][0]
lambd_seq = np.random.uniform(0.02, 1.5, 50)

def shuffle_data(data):
    np.random.permutation(data)


def split_data(data, num_folds, fold):
    split = np.split(np.array(data), num_folds)
    chosen = split[fold - 1]
    rest = np.concatenate([split[i] for i in range(num_folds) if i != (fold - 1)])
    return chosen, rest


def train_model(train_data, lambd):
    train_X, train_y = zip(*train_data)
    x_t_dot_x = np.dot(np.transpose(train_X), train_X)
    lambda_identity = lambd * np.identity(x_t_dot_x.shape[0])
    beta_ridge = np.dot(np.dot(np.linalg.inv(x_t_dot_x + lambda_identity), np.transpose(train_X)), train_y)
    return beta_ridge


def predict(test_X, beta_trained):
    test_label = np.dot(test_X, beta_trained)
    return test_label


def loss(test_data, beta_trained):
    test_X, test_y = zip(*test_data)
    test_label = predict(test_X, beta_trained)
    result = np.linalg.norm(test_y - test_label, ord=2)
    return result ** 2 / test_label.shape[0]


def cross_validation(data, num_folds):
    cv_error = {}
    for i in lambd_seq:
        cv_loss_lmd = 0
        for fold in range(1, num_folds + 1):
            data_fold, data_rest = split_data(data, num_folds=num_folds, fold=fold)
            beta = train_model(data_rest, lambd=i)
            cv_loss_lmd += loss(data_fold, beta)
        cv_error[i] = cv_loss_lmd / num_folds
    lowest_err = float("inf")
    best_lambda = 0
    for i in cv_error:
        if lowest_err > cv_error[i]:
            lowest_err = cv_error[i]
            best_lambda = i
    print("Best lambda for " + str(num_folds) + " fold is", best_lambda)
    return cv_error


if __name__ == "__main__":
    combined_training = list(zip(data_train_X, data_train_y))
    combined_testing = list(zip(data_test_X, data_test_y))
    shuffle_data(combined_training)

    num_folds_seq = [5, 10]

    lines = []
    plot_shape  = {num_folds_seq[0]: 's', num_folds_seq[1]: 'o'}

    for num_folds in num_folds_seq:
        cv_error = cross_validation(combined_training, num_folds)
        x_val = []
        y_val_cv_err = []
        for i in lambd_seq:
            x_val.append(i)
            y_val_cv_err.append(cv_error[i])

        line, = plt.plot(x_val, y_val_cv_err, 'g'+ plot_shape[num_folds], label='Cross Validation Error ' +
                                                                                str(num_folds) + ' fold')
        lines.append(line,)

    train_error = {}
    test_error = {}
    for i in lambd_seq:
        beta_ridge = train_model(combined_training, i)
        train_error[i] = loss(combined_training, beta_ridge)
        test_error[i] = loss(combined_testing, beta_ridge)

    x_val = []
    y_val_train_err = []
    y_val_test_err = []
    for i in lambd_seq:
        x_val.append(i)
        y_val_train_err.append(train_error[i])
        y_val_test_err.append(test_error[i])

    line1, = plt.plot(x_val, y_val_train_err, 'b^', label='Training Error')
    line2, = plt.plot(x_val, y_val_test_err, 'r^', label='Testing Error')
    lines.append(line1)
    lines.append(line2)

    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.legend(handles=lines, loc='best')
    plt.show()
