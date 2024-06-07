import networkx as nx
import random
import numpy as np 
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations, product
import math




import cvxgraphalgs as cvxgr
from cvxgraphalgs.structures import Cut


def gw_cut(graph):
    sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(graph)
    gw_cut=sdp_cut.evaluate_cut_size(graph)
    gw_string=sdp_cut.left
    return gw_cut

def generate_pauli_strings(n):
    pauli_set = ['I', 'X', 'Y', 'Z']
    pauli_strings = []
    
    # Generate strings with k non-identity Pauli operators
    for k in range(1, n + 1):
        for positions in combinations(range(n), k):
            for pauli_operators in product(pauli_set[1:], repeat=k):  # Exclude 'I'
                string = ['I'] * n
                for pos, op in zip(positions, pauli_operators):
                    string[pos] = op
                pauli_strings.append(''.join(string))
    
    return pauli_strings


def generate_binary_strings(n, k, x):
    if k < 0 or k > n:
        raise ValueError("Invalid value for k")

    # Generate all combinations of indices with exactly k ones
    indices_combinations = list(combinations(range(n), k))

    # Generate binary strings based on the selected indices
    z_strings = []
    x_strings = []
    y_strings = []
    for indices in indices_combinations:
        binary_string_x= ['I'] * n
        binary_string_y= ['I'] * n
        binary_string_z= ['I'] * n
        for index in indices:
            binary_string_z[index] = 'Z'
            binary_string_y[index] = 'Y'
            binary_string_x[index] = 'X'
        '''
        z_strings.append("".join(binary_string_z))
        #y_strings.append("".join(binary_string_y))
        x_strings.append("".join(binary_string_x))
        '''
        z_strings.append("".join(['I'] * x) + "".join(binary_string_z))
        x_strings.append("".join(['I'] * x) + "".join(binary_string_x))
        y_strings.append("".join(['I'] * x) + "".join(binary_string_y))

    return z_strings + x_strings + y_strings

def nodes_by_color(color_dict):
    nodes_by_color_dict = {}
    for node, color in color_dict.items():
        if color not in nodes_by_color_dict:
            nodes_by_color_dict[color] = []
        nodes_by_color_dict[color].append(node)
    return nodes_by_color_dict

def vertex_to_pauli(num_vertices,num_qubits,k,num_ancillas):
    strings=[]
    for i in range (1,k+1):
        strings+=generate_binary_strings(num_qubits, i,num_ancillas)
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    hamiltonian_dict={}
    
    for i in range(num_vertices):
        hamiltonian_dict[strings[i]]=1

    obs=[]
    for o in hamiltonian_dict.items():
        obs.append(SparsePauliOp([o[0]],coeffs=o[1]))
    
    return obs


def vertex_to_pauli_qrao(num_vertices,num_qubits,k,num_ancillas):
    strings=[]
    for i in range (1,k+1):
        if len(strings)<num_vertices:
            #print(generate_binary_strings(num_qubits,i,num_ancillas))
            strings+=generate_pauli_strings(num_qubits)
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    
    return strings[:num_vertices]


def graph_to_paulis_qrao(graph):
    graph_coloring = nx.greedy_color(graph)
    dict=nodes_by_color(graph_coloring)
    list_aux=[]
    qubits=0
    for i,element in enumerate(dict.values()):

        if len(element)==1:
            list_aux.append([i,element,[qubits],'Z'])
            qubits+=1
        else:
            #n_qubits=math.ceil(math.log(len(element)/(3),2))+1
            print(len(element))
            print(math.floor(math.log(len(element),4))+1)
            n_qubits=math.floor(math.log(len(element),4))+1

            list_aux.append([i,element,list(np.arange(qubits,qubits+n_qubits)),vertex_to_pauli_qrao(len(element),n_qubits,n_qubits,0)])
            qubits+=n_qubits
    return list_aux


def vertex_to_pauli_dict(num_vertices,num_qubits,k,num_ancillas):
    strings=[]
    for i in range (1,k+1):
        if len(strings)<num_vertices:
            strings+=generate_binary_strings(num_qubits, i,num_ancillas)
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    hamiltonian_dict={}
    
    for i in range(num_vertices):
        hamiltonian_dict[strings[i]]=1
    print(hamiltonian_dict)
    
    return hamiltonian_dict


def generate_random_graph(n,p):
    # Create an empty graph
    G = nx.Graph()

    # Add vertices
    vertices = list(range(1, n + 1))
    G.add_nodes_from(vertices)

    # Initialize the dictionary for bit-strings and weights
    bit_strings_and_weights = {}

    # Add random edges with random weights
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if random.uniform(0, 1)<p:
                weight = random.uniform(0, 1)  # You can adjust the range of weights as needed
                G.add_edge(i, j, weight=1)



    return G

def cost_string(G, string):
        
        C = 0
        for edge in G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = G[edge[0]][edge[1]]["weight"]
            C += w*(1-(int(string[i-1])-1/2)*(int(string[j-1])-1/2)*4)/2
        return C
'''
def cost_string(G, string):
        
        C = 0
        for edge in G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = G[edge[0]][edge[1]]["weight"]
            C += w*(1-(int(string[i-1])+1/2)*(int(string[j-1])+1/2)*4)/2
        return C
'''
def operator_vertex_pauli(graph):
    obs=[]
    for o,coeff in operator_vertex(graph):
        obs.append(SparsePauliOp(o,coeffs=np.sqrt(coeff)))
        #obs.append(SparsePauliOp(o,coeffs=1))
    print(obs)
    return obs

def edge_pauli(graph):
    obs=operator_vertex_pauli(graph)
    edge_obs=0
    for edge in graph.edges():
        edge_obs+=SparsePauliOp(obs[int(edge[0])-1]@obs[int(edge[1])-1],coeffs=graph[edge[0]][edge[1]].get('weight')*obs[int(edge[0])-1].coeffs*obs[int(edge[1])-1].coeffs)
    print(edge_pauli)
    return edge_obs

def Hamiltonian_spectrum(G):
    n=len(G.nodes)
    bit_strings = [bin(i)[2:].zfill(n) for i in range(2**n)]
    result = {bit_string: cost_string(G,bit_string) for bit_string in bit_strings}
    return result


def counts_in_binary_with_padding(counts, n):
    # Step 1: Convert to binary
    bin_counts={}
    for num,count in counts.items():
        binary_representation = bin(num)[2:]  # [2:] removes the '0b' prefix
        
        # Step 2: Add leading zeros if necessary
        if len(binary_representation) < n:
            binary_representation = '0' * (n - len(binary_representation)) + binary_representation
        bin_counts[binary_representation]=count
    
    return bin_counts



def operator_vertex(graph):
    list_problem=graph_to_paulis_qrao(graph)
    print(graph_to_paulis_qrao)
    ops_vertex=[None]*len(graph.nodes())
    N_qubits=list_problem[-1][2][-1]+1
    for color in list_problem:
        for i in range (len(color[1])):
            string=color[-1][i]
            new_string=list('I' * (N_qubits))
            string=list(string)

            for j,qubit in enumerate(color[2]):
                if len(color[2])==1:
                    new_string[color[2][0]]=string[0]
                else:
                    new_string[qubit]=string[j]
            new_string=''.join(new_string)
            ops_vertex[int(color[1][i])-1]=[new_string,len(color[1])]
    print(ops_vertex)
    return ops_vertex