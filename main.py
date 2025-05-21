from multilayer_perceptron import MultilayerPerceptron
from loss import mean_squared_error
from graph import Graph

# XOR Dataset
inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

labels = [0.0, 1.0, 1.0, 0.0]

# Model
model = MultilayerPerceptron(2, [3], 1)

# Hyperparameters
EPOCHS = 10000
LEARNING_RATE = 0.01

# Train
for epoch in range(EPOCHS):
    # Forward Pass
    outputs = [model(input) for input in inputs]

    # Criterion
    criterion = mean_squared_error(outputs, labels)

    # Zero Gradient
    for parameter in model.parameters():
        parameter.gradient = 0.0

    # Backward Pass
    criterion.backward()

    # Optimize
    for parameter in model.parameters():
        parameter.value -= parameter.gradient * LEARNING_RATE

    # print(criterion)

# Test
for input, label in zip(inputs, labels):
    output = model(input)
    print(input, output, label)
    print()

Graph(criterion).render()