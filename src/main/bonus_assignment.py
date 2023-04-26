import numpy as np

# Question 1

def func(x, x_0, tolerance):
    return (max(abs(x - x_0))) / (max(abs(x_0)) + tolerance)

def gauss_seidel(matrix, column, tolerance, iterations):
    length = len(column)
    x = np.zeros((length), dtype=np.double)
    k = 1

    while (k <= iterations):
        x_0 = x.copy()

        for i in range(length):
            sum_one = sum_two = 0

            for j in range(i):
                sum_one += (matrix[i][j] * x[j])

            for j in range(i + 1, length):
                sum_two += (matrix[i][j] * (x_0[j]))

            x[i] = (1 / matrix[i][i]) * (-sum_one - sum_two + column[i])

            if (func(x, x_0, tolerance) < tolerance):
                return k

        k = k + 1

    return k

tol = 1e-6
iters = 50
matrix = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
column = np.array([1, 3, 0])

print(gauss_seidel(matrix, column, tol, iters))
print()


# Question 2

def jacobi(matrix, column, tolerance, iterations):
    length = len(column)
    x = np.zeros((length), dtype=np.double)
    k = 1

    while (k <= iterations):
        x_0 = x.copy()

        for i in range(length):
            sum = 0

            for j in range(length):

                if j != i:
                    sum += (matrix[i][j] * x_0[j])

            x[i] = (1 / matrix[i][i]) * (-sum + column[i])

            if (func(x, x_0, tolerance) < tolerance):
                return k

        k = k + 1

    return k

tol = 1e-6
iters = 50
matrix = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
column = np.array([1, 3, 0])

print(jacobi(matrix, column, tol, iters))
print()


# Question 3

def custom_derivative(value):
    return (3 * value * value) - (2 * value)

def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):
    iteration_counter = 0
    x = initial_approximation
    f = eval(sequence)
    f_prime = custom_derivative(initial_approximation)
    approximation: float = f / f_prime

    while(abs(approximation) >= tolerance):
        x = initial_approximation
        f = eval(sequence)
        f_prime = custom_derivative(initial_approximation)
        approximation = f / f_prime
        initial_approximation -= approximation
        iteration_counter += 1

    return iteration_counter

initial_approximation: float = 0.5
tolerance: float = .000001
sequence: str = "x**3 - (x**2) + 2"
print(newton_raphson(initial_approximation, tolerance, sequence))
print()


# Question 4

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            left: float= matrix[i][j-1]
            diagonal_left: float= matrix[i-1][j-1]
            numerator: float= left-diagonal_left
            denominator = matrix[i][0]-matrix[i-j+1][0]
            operation= numerator/denominator
            matrix[i][j]=operation

    return matrix      
        
def hermite_interpolation():
    x_points = [0, 1, 2]
    y_points = [1, 2, 4]
    slopes = [1.06, 1.23, 1.55]
    num_of_points= len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))
    
    i = 0
    for x in range(0, num_of_points * 2, 2):
        matrix[x][0] = x_points[i]
        matrix[x+1][0] = x_points[i]
        i += 1


    i = 0    
    for x in range(0, num_of_points * 2, 2):
        matrix[x][1] = y_points[i]
        matrix[x+1][1] = y_points[i]
        i += 1


    i = 0   
    for x in range(1, num_of_points * 2, 2):
        matrix[x][2] = slopes[i]
        i += 1

    filled_matrix = apply_div_dif(matrix)
    
    return filled_matrix

print(hermite_interpolation())
print()


# Question 5

def function(t: float, w: float):
    return w - (t**3)


def do_work(t, w, h):
    basic_function_call = function(t, w)

    incremented_t = t + h
    incremented_w = w + (h * basic_function_call)
    incremented_function_call = function(incremented_t, incremented_w)

    return basic_function_call + incremented_function_call

def modified_eulers():
    original_w = 0.5
    start_of_t, end_of_t = (0, 3)
    num_of_iterations = 100
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        t = start_of_t
        w = original_w
        h = h
        inner_math = do_work(t, w, h)
        next_w = w + ( (h / 2) * inner_math )
        start_of_t = t + h
        original_w = next_w
        
    return next_w

print("%.5f" % modified_eulers())
print()