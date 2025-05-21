from neuron import Neuron

class Layer:
    """
    A layer of neurons.

    Attributes:
        neurons (list): the neurons
    """
    def __init__(self, input_size, output_size):
        """

        """
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, inputs):
        """

        """
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        """

        """
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]
    