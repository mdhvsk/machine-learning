from numpy import *





def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for i in range(len(points)):
        # get x value
        x = points[i, 0]
        # get y value
        y = points[i, 1]
        # get the difference, square it and total
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        # update b and m with new more acurate b and m by performing gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def main():
    print("Hello World!")

    # Step 1 - Collect data
    points = genfromtxt('data.csv', delimiter=',')

    # Step 2 - define our hyperparameters

    # how fast should model converge
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Step 3 - train model

    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('After {0} iterations, the ending point is at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == "__main__":
    main()
