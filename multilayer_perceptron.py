from layer import Layer

class MultilayerPerceptron:
    """
    A multilayer perceptron.

    Attributes:
        layers (list): the layers
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        """

        """
        total_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = [Layer(total_sizes[i], total_sizes[i + 1]) for i in range(len(total_sizes) - 1)]

    def __call__(self, inputs):
        """

        """
        for layer in self.layers:
            inputs = layer(inputs)

        if len(inputs) == 1:
            return inputs[0]
        
        return inputs

    def parameters(self):
        """

        """
        return [parameter for layer in self.layers for parameter in layer.parameters()]
