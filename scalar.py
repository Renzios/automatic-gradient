class Scalar:
    """
    A scalar value.

    Attributes:
        value (float): the value of the scalar
        operands (set): the operands of the scalar
        operation (str): the operation of the scalar
        gradient (float): the gradient of the scalar
        differentiate (function): the function to differentiate the scalar
    """
    def __init__(self, value = 0.0, operands = set(), operation = "", gradient = 0.0, differentiate = lambda: None):
        """
        Initializes the Scalar.

        Args:
            value (float): the value of the scalar
            operands (set): the operands of the scalar
            operation (str): the operation of the scalar
            gradient (float): the gradient of the scalar
            differentiate (function): the function to differentiate the scalar
        """
        self.value = value
        self.operands = operands
        self.operation = operation
        self.gradient = gradient
        self.differentiate = differentiate

    def __repr__(self):
        """
        Returns the string representation of the Scalar.
        """
        return f"{self.value}"

    def __add__(self, other):
        """
        Returns the sum of two Scalars.

        Args:
            other (Scalar): the other addend

        Returns:
            Scalar: the sum of two Scalars
        """
        if not isinstance(other, Scalar):
            other = Scalar(other)

        result = self.value + other.value
        result = Scalar(result, {self, other}, "+")

        def differentiate():
            self.gradient += result.gradient
            other.gradient += result.gradient
        
        result.differentiate = differentiate

        return result

    def __mul__(self, other):
        """
        Returns the product of two Scalars.

        Args:
            other (Scalar): the other factor

        Returns:
            Scalar: the product of two Scalars
        """
        if not isinstance(other, Scalar):
            other = Scalar(other)

        result = self.value * other.value
        result = Scalar(result, {self, other}, "*")

        def differentiate():
            self.gradient += result.gradient * other.value
            other.gradient += result.gradient * self.value
        
        result.differentiate = differentiate

        return result

    def __pow__(self, other):
        """
        Returns the power of two Scalars.

        Args:
            other (Scalar): the exponent

        Returns:
            Scalar: the power of two Scalars
        """
        if not isinstance(other, Scalar):
            other = Scalar(other)

        result = self.value ** other.value
        result = Scalar(result, {self, other}, "**")

        def differentiate():
            self.gradient += result.gradient * other.value * self.value ** (other.value - 1)
        
        result.differentiate = differentiate

        return result

    def __neg__(self):
        """
        Returns the negative of the Scalar.

        Returns:
            Scalar: the negative of the Scalar
        """
        return self * -1

    def __sub__(self, other):
        """
        Returns the difference of two Scalars.

        Args:
            other (Scalar): the subtrahend

        Returns:
            Scalar: the difference of two Scalars
        """
        return self + -other

    def __truediv__(self, other):
        """
        Returns the quotient of two Scalars.

        Args:
            other (Scalar): the divisor

        Returns:
            Scalar: the quotient of two Scalars
        """
        return self * other ** -1

    def __radd__(self, other):
        """
        Returns the sum of two Scalars.

        Args:
            other (Scalar): the other addend

        Returns:
            Scalar: the sum of two Scalars
        """
        return self + other

    def __rsub__(self, other):
        """
        Returns the difference of two Scalars.

        Args:
            other (Scalar): the subtrahend

        Returns:
            Scalar: the difference of two Scalars
        """
        return other + -self
    
    def __rmul__(self, other):
        """
        Returns the product of two Scalars.

        Args:
            other (Scalar): the other factor

        Returns:
            Scalar: the product of two Scalars
        """
        return self * other

    def __rtruediv__(self, other):
        """
        Returns the quotient of two Scalars.

        Args:
            other (Scalar): the divisor

        Returns:
            Scalar: the quotient of two Scalars
        """
        return other * self ** -1

    def backward(self):
        """
        Topologically sorts the graph and calculates the gradients.
        """
        order = []
        visited = set()

        def depth_first_search(vertex):
            if vertex not in visited:
                visited.add(vertex)

                for operand in vertex.operands:
                    depth_first_search(operand)

                order.append(vertex)
        
        depth_first_search(self)

        self.gradient = 1.0

        for vertex in reversed(order):
            vertex.differentiate()