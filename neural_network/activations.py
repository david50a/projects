import numpy as np
from math import exp

class ReLU():
    def __call__(self, output):
        return np.maximum(output, 0)

    def derivative(self, output, grad_so_far):
        return np.where(output < 0, 0, 1) * grad_so_far

class BinaryStep():
    def __call__(self, output):
        return np.where(output <= 0, 0, 1)

    def derivative(self, output, grad_so_far):
        return np.zeros(output.shape)

class Sigmoid():
    def __call__(self, output):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return 1 / (1 + np.exp(-output))

    def derivative(self, output, grad_so_far):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return (1 / (1 + np.exp(-output))) * (1 - (1 / (1 + np.exp(-output)))) * grad_so_far

class Tanh():
    def __call__(self, output):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return (np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output))

    def derivative(self, output, grad_so_far):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return (1 - ((np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output)) ** 2)) * grad_so_far


class Softplus():
    def __call__(self, output):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return np.log(1 + np.exp(output))

    def derivative(self, output, grad_so_far):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return (1 / (1 + np.exp(-output))) * grad_so_far


class ELU():
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def __call__(self, output):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return np.where(output < 0, self.alpha * (exp(output) - 1), output)

    def derivative(self, output, grad_so_far):
        output = np.clip(output, 1000, -1000)  # Prevent exp(output) from overflowing
        return np.where(output < 0, self.alpha * exp(output), 1) * grad_so_far


class SELU(ELU):
    def __init__(self, alpha, lamda) -> None:
        super().__init__(alpha)
        self.lamda = lamda

    def __call__(self, output):
        return self.lamda * super().__call__(output)

    def derivative(self, output, grad_so_far):
        return self.lamda * super().__getitem__(output) * grad_so_far


class PReLU():
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def __call__(self, output):
        return np.where(output < 0, self.alpha * output, output)

    def derivative(self, output, grad_so_far):
        return np.where(output < 0, self.alpha, 1) * grad_so_far


class LReLU(PReLU):
    def __call__(self, output):
        return super().__call__(output)

    def derivative(self, output, grad_so_far):
        return super().__getitem__(output) * grad_so_far


class Softmax():
    def __call__(self, output):
        exp_shifted = np.exp(output - np.max(output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator

    def derivative(self, output, grad_so_far):
        output = self(output)  # Get activated outputs for formulae
        batch_size, n_classes = output.shape

        # For 1 example, the jacobian is of size NxN, so for B batches, it is BxNxN
        jacobian = np.zeros((batch_size, n_classes, n_classes))

        for b in range(batch_size):
            out = output[b].reshape(-1, 1)  # Flatten output to be an Nx1 matrix
            jacobian[b] = np.diagflat(out) - np.dot(out, out.T)  # Create Jacobian for particular example

        return np.einsum('bij,bj->bi', jacobian,
                         grad_so_far)  # Efficient batch-wise dot product using Einstein summation notation
