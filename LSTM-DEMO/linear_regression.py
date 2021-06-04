import numpy as np


def loss_function(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x, y = points[i, 0], points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x, y = points[i, 0], points[i, 1]
        b_gradient += -(2 / n) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / n) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("data/data_lr.csv", delimiter=",")
    learning_rate = 0.0001
    init_b = 0
    init_w = 0
    num_iterations = 1000
    print(loss_function(init_b, init_w, points))
    [b, w] = gradient_descent_runner(points, init_b, init_w, learning_rate, num_iterations)
    print(b, w, loss_function(b, w, points))


run()
