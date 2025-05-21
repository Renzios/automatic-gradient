from random import uniform
from scalar import Scalar

class Neuron:
    """
    A neuron with weights and a bias.

    Attributes:
        weights (list): the weights of each input
        bias (float): the bias
    """
    def __init__(self, input_size):
        """

        """
        self.weights = [Scalar(uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Scalar(uniform(-1, 1))

    def __call__(self, inputs):
        """
        
        """
        return sum((w * x for w, x in zip(self.weights, inputs)), self.bias).tanh()

    def parameters(self):
        """

        """
        return self.weights + [self.bias]