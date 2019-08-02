from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()

import numpy as np
import os
import gzip
import struct
import array
import math

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
        col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)
    plt.close()


def predict_labels(test_set, prior, parameter, mode="theta"):
    if mode == "theta":
        # Assign data points to class with highest log of posterior probability
        # N by C matrix
        log_joint_prob = np.dot(test_set, np.log(parameter)) + np.dot((1 - test_set), np.log(1 - parameter)) + np.log(
            prior)
        # N by 1 vector
        log_marginal_prob = logsumexp(log_joint_prob, axis=1)
        # N by C matrix
        log_posterior_prob = (log_joint_prob.transpose() - log_marginal_prob).transpose()
        # N by 1 vector
        result = np.argmax(log_posterior_prob, axis=1)
        return result
    else:
        # N by C matrix
        log_exp_class_score = np.dot(test_set, parameter.transpose())
        # N by 1 vector
        result = np.argmax(log_exp_class_score, axis=1)
        return result


def evaluate_error(prediction, label):
    result = np.where((prediction - np.argmax(label, axis=1)) == 0, 1, 0)
    print("The prediction correctness is", np.sum(result) / result.shape[0])


def evaluate_avg_likelihood(dataset, true_label, prior, parameter, mode="theta"):
    # Evaluate the average likelihood for the predicted labels of N data points
    if mode == "theta":

        # N by D matrix
        theta_label = np.dot(true_label, parameter.transpose())

        # N by 1 vector
        likelihood_y = np.dot(true_label, prior)

        # 1 by 1 scalar
        log_joint_likelihood = 0
        theta_label_ivt = 1 - theta_label
        theta_label_ivt = np.log(theta_label_ivt)
        theta_label = np.log(theta_label)
        dataset_ivt = 1 - dataset
        for k in range(dataset.shape[0]):
            log_joint_likelihood += np.dot(dataset[k], theta_label[k])
        for j in range(dataset.shape[0]):
            log_joint_likelihood += np.dot(dataset_ivt[j], theta_label_ivt[j])
        log_joint_likelihood += np.sum(np.log(likelihood_y))

        # N by C vector
        log_joint_prob = np.dot(dataset, np.log(parameter)) + np.dot((1 - dataset), np.log(1 - parameter)) + np.log(
            prior)
        # 1 by 1 scalar
        log_x_likelihood = np.sum(logsumexp(log_joint_prob, axis=1))
        log_y_given_x_likelihood = log_joint_likelihood - log_x_likelihood
        print("The likelihood of predicted label given dataset is", log_y_given_x_likelihood / dataset.shape[0])
    else:

        # N by C matrix
        class_score = np.dot(dataset, parameter.transpose())
        class_score = class_score - np.max(class_score)
        # N by 1 vector
        log_total_exp_score = logsumexp(class_score, axis=1)
        # N by 1 vector
        true_label_index = np.argmax(true_label, axis=1)
        true_label_class_score = class_score[np.arange(dataset.shape[0]), true_label_index]
        # N by 1 vector
        log_likelihood_true_label = true_label_class_score - log_total_exp_score
        # 1 by 1 scalar
        avg_likelihood_predicted_label = np.sum(log_likelihood_true_label) / dataset.shape[0]
        print("The likelihood of predicted label given data is", avg_likelihood_predicted_label)


def predictive_distribution(dataset, bottom_num_pixels, parameter, prior):
    top_num_pixels = dataset.shape[1] - bottom_num_pixels
    top_pixels = dataset[:, :top_num_pixels]
    bot_pixels = dataset[:, top_num_pixels:]
    top_parameter = parameter[:top_num_pixels, :]
    bot_parameter = parameter[top_num_pixels:, :]

    # N by D_bot by C matrix
    log_prob_bot_given_y = np.zeros((dataset.shape[0], bottom_num_pixels, parameter.shape[1]))
    for j in range(log_prob_bot_given_y.shape[0]):
        log_prob_bot_given_y[j] = (bot_pixels[j] * np.log(bot_parameter).transpose()
                                   + (1 - bot_pixels[j]) * np.log(1 - bot_parameter).transpose()).transpose()

    # N by C matrix
    log_prob_joint_y_top = np.dot(top_pixels, np.log(top_parameter)) + \
                           np.dot((1 - top_pixels), np.log(1 - top_parameter)) + np.log(prior)
    # N by 1 matrix
    log_marginal_top = logsumexp(log_prob_joint_y_top, axis=1)

    # N by C matrix
    log_prob_y_given_top = (log_prob_joint_y_top.transpose() - log_marginal_top).transpose()

    # N by D_bot by C matrix
    log_prob_top_joint_bot = log_prob_bot_given_y + log_prob_y_given_top[:, None, :]

    # N by D_bot matrix
    log_prob_bot_given_top = logsumexp(log_prob_top_joint_bot, axis=2)

    return np.concatenate((top_pixels, 1 - np.exp(log_prob_bot_given_top)), axis=1)


def gradient_descent(w0, X, C):
    """
    :param w0: initial point of parameter, C by D matrix
    :param X: input dataset, N by D matrix
    :param C: input label, N by C matrix
    :return: optimal parameter, C by D matrix
    """
    wt = w0
    wt_1 = wt

    iterations = 0
    step_size = 0.5
    while True:
        wt = wt_1

        # N by C matrix
        class_score = np.dot(X, wt.transpose())
        max_class_score = np.max(class_score, axis=1)[:, None]
        shifted_class_score = class_score - max_class_score

        # N by 1 vector
        total_class_score = np.sum(np.exp(shifted_class_score), axis=1)[:, None]

        # N by C matrix
        prob_C_given_X_w = np.exp(shifted_class_score) / total_class_score

        # C by D matrix
        first_order_gradient = np.dot(C.transpose(), X) - np.dot(prob_C_given_X_w.transpose(), X)

        # second_order_gradient = -np.dot(((np.exp(shifted_class_score) * total_class_score
        #                                   - np.exp(2 * shifted_class_score))
        #                                  / np.square(total_class_score)).transpose(), np.square(X))

        wt_1 = wt + first_order_gradient * step_size

        iterations += 1
        # Decrease step size gradually
        if iterations % 5 == 0:
            step_size /= 2
        if np.allclose(wt, wt_1):
            break
    return wt


def plot_scatter(data1, data2, name="plot"):
    colors = ("red", "blue")
    groups = ("data1", "data2")
    d = (data1, data2)
    for data, color, group in zip(d, colors, groups):
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    plt.title(name)
    plt.legend(loc=2)
    plt.show()
    plt.close()


def get_distance(x_n, mu_k):
    d = x_n.shape[0]
    distance = 0
    for i in range(d):
        distance += (x_n[i] - mu_k[i]) ** 2
    return distance


def km_m_step(dataset, r):
    n = dataset.shape[0]
    d = dataset.shape[1]
    num_k = r.shape[1]
    mu = np.zeros(shape=(num_k, d))

    for k in range(num_k):
        numerator = np.zeros(d)
        denominator = 0
        for i in range(n):
            numerator += dataset[i] * r[i][k]
            denominator += r[i][k]
        if not denominator == 0:
            mu[k] = (numerator / denominator)
        else:
            mu[k] = np.zeros(d)
    return mu


def km_e_step(dataset, mu):
    n = dataset.shape[0]
    num_k = mu.shape[0]
    r = np.zeros(shape=(n, num_k))

    for i in range(n):
        minimum = math.inf
        min_index = 0
        for k in range(num_k):
            current = get_distance(dataset[i], mu[k])
            if current < minimum:
                minimum = current
                min_index = k
        r[i][min_index] = 1
    return r


def cost(dataset, r, mu):
    n = dataset.shape[0]
    num_k = mu.shape[0]
    c = 0
    for i in range(n):
        for k in range(num_k):
            c += r[i][k] * get_distance(dataset[i], mu[k])
    return c


def k_means(dataset, initial_mu):
    c = []
    r = km_e_step(dataset, initial_mu)
    c.append(cost(dataset, r, initial_mu))
    temp_r = np.zeros(shape=(r.shape[0], r.shape[1]))
    new_mu = initial_mu

    while not np.array_equal(r, temp_r):
        new_mu = km_m_step(dataset, r)
        temp_r = r
        r = km_e_step(dataset, new_mu)
        c.append(cost(dataset, r, new_mu))
    return r, new_mu, c


def normal_density(dataset, mu, sigma):
    y = multivariate_normal.pdf(dataset, mean=mu, cov=sigma)
    return y


def e_step(dataset, pi, mu, sigma):
    n = dataset.shape[0]
    num_k = mu.shape[0]
    gama = np.zeros(shape=(n, num_k))
    for i in range(n):

        denominator = 0
        for k in range(num_k):
            normal = normal_density(dataset[i], mu[k], sigma[k])
            denominator += pi[k] * normal

        for k in range(num_k):
            normal = normal_density(dataset[i], mu[k], sigma[k])
            numerator = pi[k] * normal
            gama[i][k] = numerator / denominator
    return gama


def m_step(dataset, gamma):
    n = dataset.shape[0]
    d = dataset.shape[1]
    num_k = gamma.shape[1]

    Nk = np.sum(gamma, axis=0)
    pi = Nk / n
    mu = np.zeros(shape=(num_k, d))
    for k in range(num_k):
        num = np.zeros(d)
        for i in range(n):
            num += gamma[i][k] * dataset[i]
        mu[k] = num / Nk[k]
    sigma = []
    for k in range(num_k):
        sigma_k = np.zeros(shape=(d, d))
        for i in range(n):
            sigma_k += gamma[i][k] * np.outer((dataset[i] - mu[k]), (dataset[i] - mu[k]))
        sigma.append(sigma_k / Nk[k])
    return pi, mu, sigma


def log_likelihood(dataset, pi, mu, sigma):
    n = dataset.shape[0]
    num_k = mu.shape[0]
    ln_likelihood = 0
    for i in range(n):
        sum_over_k = 0
        for k in range(num_k):
            normal = normal_density(dataset[i], mu[k], sigma[k])
            sum_over_k += pi[k] * normal
        ln_likelihood += np.log(sum_over_k)
    return ln_likelihood


def mix_gaussian(dataset, init_pi, init_mu, init_sigma):
    likelihood_list = []
    likelihood = log_likelihood(dataset, init_pi, init_mu, init_sigma)
    likelihood_list.append(likelihood)
    gamma = e_step(dataset, init_pi, init_mu, init_sigma)
    mu_updated = init_mu
    pi_updated = init_pi
    sigma_updated = init_sigma
    old_likelihood = 0

    while not np.allclose(likelihood, old_likelihood):
        pi_updated, mu_updated, sigma_updated = m_step(dataset, gamma)
        gamma = e_step(dataset, pi_updated, mu_updated, sigma_updated)
        old_likelihood = likelihood
        likelihood = log_likelihood(dataset, pi_updated, mu_updated, sigma_updated)
        likelihood_list.append(likelihood)
    return gamma, pi_updated, mu_updated, sigma_updated, likelihood_list


def plot_cluster(data, param, name):
    data1 = []
    data2 = []
    for i in range(data.shape[0]):
        if param[i][0] > param[i][1]:
            data1.append(data[i])
        else:
            data2.append(data[i])
    plot_scatter(np.array(data1), np.array(data2), name)
    return data1, data2


if __name__ == '__main__':
    # load data
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # 10000 training sets
    train_images = train_images[0:10000, :]
    train_labels = train_labels[0:10000, :]

    # encode image to binary with cutoff of 0.5
    train_images = np.where(train_images > 0.5, 1, 0)

    # Calculate MAP estimator theta
    numerator = np.dot(train_images.transpose(), train_labels) + 1
    denominator = np.sum(train_labels, axis=0) + 2
    theta_MAP = numerator / denominator
    save_images(theta_MAP.transpose(), "theta_greyscale")

    pi = np.array([0.1] * 10)
    # Evaluate on training set
    predicted_labels_training = predict_labels(train_images, prior=pi, parameter=theta_MAP)
    print("Evaluating prediction on training set:")
    evaluate_error(predicted_labels_training, train_labels)
    print("Evaluating average of log likelihood of training data:")
    evaluate_avg_likelihood(train_images, train_labels, prior=pi, parameter=theta_MAP)

    # Evaluate on test set
    predicted_labels_test = predict_labels(test_images, prior=pi, parameter=theta_MAP)
    print("Evaluating prediction on test set:")
    evaluate_error(predicted_labels_test, test_labels)
    print("Evaluating average of log likelihood of testing data:")
    evaluate_avg_likelihood(test_images, test_labels, prior=pi, parameter=theta_MAP)

    N = train_images.shape[0]
    C = train_labels.shape[1]
    D = train_images.shape[1]

    # Produce random image samples from the model
    number_of_images = 10
    sampled_Y = np.random.choice(C, number_of_images, p=pi)
    sampled_X = np.zeros((number_of_images, D))
    for i in range(sampled_Y.shape[0]):
        sampled_X[i] = [np.random.choice([1, 0], 1, p=[theta_MAP[d][sampled_Y[i]], 1 - theta_MAP[d][sampled_Y[i]]])[0]
                        for d in range(D)]
    save_images(sampled_X, "sampled_pic")

    # Use marginal distribution to recover half of the dimensions
    number_of_images = 20
    bottom_pixels_to_predict = int(D / 2)
    sampled_indices_from_training = np.random.choice(N, number_of_images)
    sampled_X_from_training = np.array([train_images[n] for n in sampled_indices_from_training])
    predicted_X = predictive_distribution(sampled_X_from_training, bottom_pixels_to_predict, theta_MAP, pi)
    save_images(predicted_X, "predicted_bottom")

    # Use Gradient Descent to find optimal w for logistic model for probability of class give x and w.
    w0 = np.zeros(shape=(C, D))
    w_opt = gradient_descent(w0, train_images, train_labels)
    save_images(w_opt, "W_trained")
    # Evaluate on training set
    predicted_labels_training = predict_labels(train_images, prior=pi, parameter=w_opt, mode="weight")
    print("Evaluating prediction on training set:")
    evaluate_error(predicted_labels_training, train_labels)
    print("Evaluating average of log likelihood of training data:")
    evaluate_avg_likelihood(train_images, train_labels,prior=pi, parameter=w_opt, mode="weight")
    # Evaluate on test set
    predicted_labels_test = predict_labels(test_images, prior=pi, parameter=theta_MAP)
    print("Evaluating prediction on test set:")
    evaluate_error(predicted_labels_test, test_labels)
    print("Evaluating average of log likelihood of testing data:")
    evaluate_avg_likelihood(test_images, test_labels, prior=pi, parameter=w_opt, mode="weight")

    # Generate some data from multivariate gaussian.
    N = 400
    D = 2
    K = 2
    mu_1 = np.array([0.1, 0.1])
    mu_2 = np.array([6.0, 0.1])
    sigma = np.array([[10, 7], [7, 10]])
    data1 = np.random.multivariate_normal(mu_1, sigma, 200)
    true_label_info = {}
    for i in data1:
        true_label_info[(i[0], i[1])] = 1
    data2 = np.random.multivariate_normal(mu_2, sigma, 200)
    for j in data2:
        true_label_info[(j[0], j[1])] = 2
    plot_scatter(data1, data2, "original")

    # Use k-means algorithm
    mu_1 = np.array([0.0, 0.0])
    mu_2 = np.array([1.0, 1.0])
    mu = np.zeros(shape=(2, 2))
    mu[0] = mu_1
    mu[1] = mu_2
    data = np.vstack((data1, data2))
    r, mu, cost_list = k_means(data, mu)
    predicted_label_data1, predicted_label_data2 = plot_cluster(data, r, name="k-means Converged")
    miss_count = 0
    for i in predicted_label_data1:
        if true_label_info[(i[0], i[1])] == 2:
            miss_count += 1
    for j in predicted_label_data2:
        if true_label_info[(j[0], j[1])] == 1:
            miss_count += 1
    miss_rate = miss_count / N
    print("Miss rate for k-means is", miss_rate)
    plt.plot(cost_list)
    plt.show()

    # Use EM algorithm for Gaussian Mixtures
    sigma_1 = np.identity(data.shape[1])
    sigma_2 = np.identity(data.shape[1])
    sigma_init = [sigma_1, sigma_2]
    pi_init = np.array([0.5, 0.5])
    mu_1 = np.array([0.0, 0.0])
    mu_2 = np.array([1.0, 1.0])
    mu_init = np.zeros(shape=(2, 2))
    mu_init[0] = mu_1
    mu_init[1] = mu_2
    gamma, pi_updated, mu_updated, sigma_updated, likelihood_list = mix_gaussian(data, pi_init, mu_init, sigma_init)
    predicted_label_data1, predicted_label_data2 = plot_cluster(data, gamma, name="Mixture of Gaussian Converged")
    miss_count = 0
    for i in predicted_label_data1:
        if true_label_info[(i[0], i[1])] == 2:
            miss_count += 1
    for j in predicted_label_data2:
        if true_label_info[(j[0], j[1])] == 1:
            miss_count += 1
    miss_rate = miss_count / N
    print("Miss rate for EM algorithm is", miss_rate)
    plt.plot(likelihood_list)
    plt.show()
