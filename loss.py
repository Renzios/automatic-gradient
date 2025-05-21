def mean_squared_error(outputs, labels):
    """
    
    """
    return sum((a - y) ** 2 for a, y in zip(outputs, labels)) / len(outputs)