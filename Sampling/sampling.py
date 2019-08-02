import numpy as np
import math
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matlab as mb


def bernoulli_prob(x, theta):
    numerator = (math.sin(5 * (x - theta))) ** 2
    denominator = 25 * (math.sin(x - theta) ** 2)
    return numerator / denominator


def numerical_integration(x, theta):
    """
    User numerical integration to calculate p(g=0|theta)
    :param x: n by 1 vector
    :param theta: 1 by 1 scalar
    :return: 1 by 1 scalar
    """
    delta_x = np.diff(x)
    new_x = np.delete(x, 0)
    prob_g_given_x = np.zeros(new_x.shape[0])
    prob_x_given_theta = np.zeros(new_x.shape[0])
    for i in range(new_x.shape[0]):
        prob_g_given_x[i] = 1 - bernoulli_prob(new_x[i], theta)
        prob_x_given_theta[i] = sp.stats.norm(theta, 4).pdf(new_x[i])
    return np.dot(np.multiply(prob_g_given_x, prob_x_given_theta), delta_x)


def rejection_sampler(g, theta, size):
    # Sample x from Normal with mean of theta, standard deviation 4
    number_of_success = 0
    total_number_of_sampling = 0
    sampled_x_result = []
    while True:
        total_number_of_sampling += 1
        sampled_x = np.random.normal(theta, 4, 1)
        sampled_g = np.random.binomial(1, bernoulli_prob(sampled_x, theta), 1)
        if sampled_g == g:
            sampled_x_result.append(sampled_x)
            number_of_success += 1
        if number_of_success >= size:
            break
    fraction_of_success = number_of_success / total_number_of_sampling
    print("The faction of accepted samples are", fraction_of_success)
    return np.array(sampled_x_result)


def importance_sampler(g, theta, size):
    # Sample x from Normal with mean of theta, standard deviation 4
    x = np.random.normal(theta, 4, size)
    prob_g_given_x = np.zeros(size)
    for i in range(size):
        if g == 1:
            prob_g_given_x = bernoulli_prob(x[i], theta)
        else:
            prob_g_given_x[i] = 1 - bernoulli_prob(x[i], theta)
    weighted_prob_x = sp.stats.norm(theta, 4).pdf(x) / sp.stats.norm(theta, 4).pdf(x)
    weighted_prob_x = weighted_prob_x / np.sum(weighted_prob_x)
    return np.dot(weighted_prob_x, prob_g_given_x)


def joint_g_x_theta(g, x, theta):
    prob_theta = 1 / (10 * math.pi * (1 + (theta / 10) ** 2))
    prob_x_joint_theta = sp.stats.norm(theta, 4).pdf(x) * prob_theta
    if g == 1:
        return bernoulli_prob(x, theta) * prob_x_joint_theta
    else:
        return (1 - bernoulli_prob(x, theta)) * prob_x_joint_theta


def metropolis_hasting_algorithm(g, x, sd, step):
    current_theta = np.random.random(1)
    sampled_theta = [current_theta]
    current_step = 1
    while True:
        proposed_theta = np.random.normal(current_theta, sd, 1)
        pi_new_times_trans_back = joint_g_x_theta(g, x, proposed_theta) * sp.stats.norm(proposed_theta, sd).pdf(current_theta)
        pi_old_times_trans_forward = joint_g_x_theta(g, x, current_theta) * sp.stats.norm(current_theta, sd).pdf(proposed_theta)
        acceptance_ratio = min(1, pi_new_times_trans_back / pi_old_times_trans_forward)
        accept = np.random.binomial(1, acceptance_ratio, 1)
        if accept[0] == 1:
            current_theta = proposed_theta
        sampled_theta.append(current_theta)
        current_step += 1
        if current_step >= step:
            break
    return np.array(sampled_theta)


if __name__ == '__main__':
    # Using numerical integration to estimate p(g=0|theta=0)
    x = np.linspace(-20, 20, num=10000)
    print("Estimated p(g=0|theta=0) using numerical integration is", numerical_integration(x, theta=0))

    # Using Rejection Sampling to sample 10000 x
    sampled_x = rejection_sampler(1, 0, 10000)
    plt.hist(sampled_x, bins='auto')
    plt.title("Histogram of Sampled X with Rejection Sampling")
    plt.show()

    # Using normalized importance Sampling to sample 10000 x
    print("Estimated p(g=0|theta=0) using importance sampling is", importance_sampler(0, 0, 10000))

    # Plot joint density of g, x, and theta
    theta = np.arange(-15, 15, step=0.001)
    prob = np.array([joint_g_x_theta(1, 1.7, theta[k]) for k in range(theta.shape[0])])
    plt.plot(theta, prob)
    plt.title("Joint probability of g=1, x=1.7 and theta")
    plt.show()

    # Using Metropolis Hasting to sample theta
    sampled_theta = metropolis_hasting_algorithm(1, 1.7, 1, 30000)
    plt.hist(sampled_theta, bins='auto')
    plt.title("Histogram of Sampled theta with Metropolis Hasting Algorithm")
    plt.show()

    # Estimating posterior of theta given x and g
    within_region_theta = [1 for k in sampled_theta if -3 < k < 3]
    posterior_prob = sum(within_region_theta) / sampled_theta.shape[0]
    print("Estimated posterior probability from Metropolis Hasting Algorithm samples is", posterior_prob)
