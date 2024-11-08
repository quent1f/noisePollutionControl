import numpy as np
import _env


def apply_gradient_descent(values, gradient, grid, learning_rate):
    (rows, cols) = np.shape(grid)
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if grid[row, col] == _env.NODE_ROBIN:
                avg_gradient = 0
                count = 0
                if grid[row + 1, col] == _env.NODE_INTERIOR:
                    avg_gradient += gradient[row + 1, col]
                    count += 1
                if grid[row, col + 1] == _env.NODE_INTERIOR:
                    avg_gradient += gradient[row, col + 1]
                    count += 1
                if grid[row, col - 1] == _env.NODE_INTERIOR:
                    avg_gradient += gradient[row, col - 1]
                    count += 1
                if grid[row - 1, col] == _env.NODE_INTERIOR:
                    avg_gradient += gradient[row - 1, col]
                    count += 1
                if count != 0:
                    avg_gradient /= count
                else:
                    avg_gradient = 0
                avg_gradient /= 4
                values[row, col] -= learning_rate * avg_gradient
            elif grid[row, col] == _env.NODE_INTERIOR:
                values[row, col] -= learning_rate * gradient[row, col]
    return values
