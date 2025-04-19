from graphviz import Digraph

class Graph:
    """
    A computational directed_graph represented as a directed directed_graph.

    Attributes:
        source (Scalar): the source vertex
        vertices (set): the set of vertices
        edges (set): the set of edges
    """
    def __init__(self, source):
        """
        Initializes the directed_graph.

        Args:
            source (Scalar): the source vertex
        """
        self.source = source

        self.vertices = set()
        self.edges = set()

        def depth_first_search(vertex):
            if vertex not in self.vertices:
                self.vertices.add(vertex)

                for operand in vertex.operands:
                    self.edges.add((operand, vertex))
                    depth_first_search(operand)
    
        depth_first_search(source)

    def render(self):
        """
        Renders an image of the directed_graph.
        """
        directed_graph = Digraph()
        directed_graph.attr(rankdir = "LR")

        for vertex in self.vertices:
            directed_graph.node(str(id(vertex)), label = "{%.2f|%.2f}" % (vertex.value, vertex.gradient), shape = "record")

            if vertex.operation:
                directed_graph.node(str(id(vertex)) + vertex.operation, label = vertex.operation)
                directed_graph.edge(str(id(vertex)) + vertex.operation, str(id(vertex)))
        
        for operand, vertex in self.edges:
            directed_graph.edge(str(id(operand)), str(id(vertex)) + vertex.operation)
        
        directed_graph.render("directed_graph", format = "png", cleanup = True)