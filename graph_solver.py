import numpy as np
from graphviz import Digraph


class Node:
    def __init__(self, identifier, reaction_matrix):
        self.identifier = identifier
        self.reaction_matrix = reaction_matrix
        self.edges = []

class Edge:
    def __init__(self, source, destination, length, diffusivity, velocities):
        self.source = source
        self.destination = destination
        self.length = length
        self.diffusivity = diffusivity
        self.velocities = velocities


    def calculate_adj_length(self,reverse_orientation=False):
        # Convert diffusivity and velocities to NumPy arrays for vectorized computation
        D_i = np.array(self.diffusivity)
        nu_i = np.array(self.velocities)
        if reverse_orientation:
            nu_i=-nu_i

        # Calculate tilde_ell_i for all i based on the provided formula
        non_zero_indices = np.where(nu_i != 0)
        tilde_ell = np.full_like(D_i,self.length, dtype=float)

        tilde_ell[non_zero_indices] = (D_i[non_zero_indices] / nu_i[non_zero_indices]) * (1 - np.exp(-nu_i[non_zero_indices] * self.length / D_i[non_zero_indices]))

        return tilde_ell

    def __str__(self):
        return f"Edge from {self.source.identifier} to {self.destination.identifier} with length {self.length}."

def calculate_xi(node, edge):
    # Calculate p(n, b)
    p_nb = 1 / len(node.edges)  # Assuming you have a list of edges for each node

    reverse_orientation=False
    if node==edge.destination: reverse_orientation=True
    tilde_ell_i = edge.calculate_adj_length(reverse_orientation)

    # Calculate xi_i(n, b)
    xi_i = (p_nb * edge.diffusivity) / tilde_ell_i

    return xi_i

def calculate_eta(node, edge):
    xi_i = calculate_xi(node, edge)

    # Calculate the sum of xi_i(n, b') for all edges b' attached to node n
    sum_xi = np.sum([calculate_xi(node, e) for e in node.edges], axis=0)

    eta_i = xi_i / sum_xi

    return eta_i

def calculate_eta_matrix(node, edge):
    eta=calculate_eta(node,edge)
    return np.diag(eta)

def calculate_tilde_K_matrix(node):
    tilde_K_matrix = np.zeros_like(node.reaction_matrix, dtype=float)

    # Calculate the sum of xi_i(n, b') for all edges b' attached to node n
    sum_xi = np.sum([calculate_xi(node, e) for e in node.edges], axis=0)

    for i in range(node.reaction_matrix.shape[0]):
        for j in range(node.reaction_matrix.shape[1]):
            K_ij = node.reaction_matrix[i, j]
            tilde_K_matrix[i, j] = K_ij / sum_xi[i]

    return tilde_K_matrix

def calculate_lambda(node1,node2):
    if node1==node2:
        K_tilde=calculate_tilde_K_matrix(node1)
        return K_tilde-np.eye(K_tilde.shape[0])

    # see if edge node1-node2 exists
    found_edge=None
    for edge in node1.edges:
        if edge.destination==node2 or edge.source==node2:
            found_edge=edge
    if found_edge==None: return np.zeros((node1.reaction_matrix.shape[0],node1.reaction_matrix.shape[0]))

    eta = calculate_eta_matrix(node1, found_edge)
    return eta

def calculate_lambda_block_matrix(graph):
    # use node ordering from graph.node_ordering variable to create matrix
    num_nodes = len(graph.node_ordering)

    # Initialize an empty block matrix
    block_matrix = np.zeros((num_nodes, num_nodes), dtype=object)

    # Fill in the blocks using the lambda_function
    for i in range(num_nodes):
        for j in range(num_nodes):
            block_matrix[i, j] = calculate_lambda(graph.node_ordering[i], graph.node_ordering[j])

    return block_matrix


def convert_block_to_regular_matrix(block_matrix):
    # Determine if the block matrix is a vector
    if block_matrix.ndim == 1:
        num_blocks_row = 1
        num_blocks_col = len(block_matrix)
        block_shape = block_matrix[0].shape  # Corrected access for a 1D array
    else:
        num_blocks_row, num_blocks_col = block_matrix.shape
        block_shape = block_matrix[0, 0].shape  # Access for a 2D array

    regular_matrix_shape = (num_blocks_row * block_shape[0], num_blocks_col * block_shape[1])
    regular_matrix = np.zeros(regular_matrix_shape)

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = block_matrix[i, j] if block_matrix.ndim > 1 else block_matrix[j]
            regular_matrix[i * block_shape[0]: (i + 1) * block_shape[0],
                           j * block_shape[1]: (j + 1) * block_shape[1]] = block

    return regular_matrix


def calculate_lambda_matrix(graph):
    block_matrix = calculate_lambda_block_matrix(graph)
    regular_matrix = convert_block_to_regular_matrix(block_matrix)

    return regular_matrix

def compute_lambda_vector(graph, exit_node_identifier):
    b_vector = []

    # Iterate through all nodes in the graph
    for node in graph.node_ordering:
        # Initialize eta_sum for the current node
        eta_sum = np.zeros_like(node.reaction_matrix,dtype=float)

        # Check if there are edges connected to the current node
        if node.edges:
            # Calculate the sum of eta(n, b) for all edges connected to the exit_node_identifier
            for edge in node.edges:
                if edge.destination.identifier == exit_node_identifier:
                    eta_sum -= calculate_eta_matrix(node, edge)

        b_vector.append(eta_sum)

    block_matrix = np.zeros((len(b_vector),), dtype=object)
    for i in range(len(b_vector)):
        block_matrix[i] = b_vector[i]

    converted_b_vector=convert_block_to_regular_matrix(block_matrix).transpose()

    return converted_b_vector

def solve_graph(graph):
    exit_node_identifier=None
    for node in graph.nodes.values():
        if node not in graph.node_ordering:
            exit_node_identifier=node.identifier
    if exit_node_identifier is None:
        raise ValueError("Exit node not defined")
    lambda_vector=compute_lambda_vector(graph,exit_node_identifier)
    lambda_matrix=calculate_lambda_matrix(graph)
    sol=np.linalg.solve(lambda_matrix,lambda_vector)
    return sol.reshape(-1,sol.shape[-1],sol.shape[-1])

class Graph:
    def __init__(self,Diff_mat=None,Velocity_mat=None):
        self.nodes = {}
        self.edges = []
        self.node_ordering = []  # List of nodes in the order they were added to the graph
        self.Diff_mat=Diff_mat
        self.Velocity_mat=Velocity_mat

    def add_node(self, identifier, reaction_matrix=None):
        if reaction_matrix is None:
            if self.Diff_mat is None:
                raise ValueError("Diffusivity matrix not defined, need to input reaction mat")
            if self.Velocity_mat is None:
                raise ValueError("Velocity matrix not defined, need to input reaction mat")
            reaction_matrix=np.zeros((self.Diff_mat.shape[0],self.Diff_mat.shape[0]))
        if identifier not in self.nodes:
            self.nodes[identifier] = Node(identifier, reaction_matrix)
            self.node_ordering.append(self.nodes[identifier])

    # dont want to add exit node to node_ordering
    def add_exit_node(self, identifier, reaction_matrix=None):
        if reaction_matrix is None:
            if self.Diff_mat is None:
                raise ValueError("Diffusivity matrix not defined, need to input reaction mat")
            if self.Velocity_mat is None:
                raise ValueError("Velocity matrix not defined, need to input reaction mat")
            reaction_matrix=np.zeros((self.Diff_mat.shape[0],self.Diff_mat.shape[0]))
        if identifier not in self.nodes:
            self.nodes[identifier] = Node(identifier, reaction_matrix)

    def add_edge(self, source, destination, length, diffusivity=None, velocities=None):
        if source in self.nodes and destination in self.nodes:
            if diffusivity is None:
                if self.Diff_mat is None:
                    raise ValueError("Diffusivity matrix not defined, need to input reaction mat")
                diffusivity=self.Diff_mat
            if velocities is None:
                if self.Velocity_mat is None:
                    raise ValueError("Velocity matrix not defined, need to input reaction mat")
                velocities=self.Velocity_mat
            new_edge = Edge(self.nodes[source], self.nodes[destination], length, diffusivity, velocities)
            self.edges.append(new_edge)
            self.nodes[source].edges.append(new_edge)
            self.nodes[destination].edges.append(new_edge)
        else:
            raise ValueError("Source or destination node not in graph.")

    def display(self):
        dot = Digraph(comment='The Reaction Network')

        for node in self.nodes.values():
            reaction_matrix_str = f"Reaction Matrix:\n{node.reaction_matrix}" if node.reaction_matrix is not None else ""
            node_label = f'Node {node.identifier}\n{reaction_matrix_str}'
            if node not in self.node_ordering:
                node_label = f'Node {node.identifier}\nExit'
            dot.node(str(node.identifier), node_label)

        for edge in self.edges:
            label = f'{edge.length}'
            dot.edge(str(edge.source.identifier), str(edge.destination.identifier), label=label)

        # dot.render('output_graph.gv', view=True)  # This will save and open the graph
        return dot
