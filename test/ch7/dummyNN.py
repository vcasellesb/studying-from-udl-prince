from typing import List, Dict
import numpy as np
from copy import deepcopy

class Neuron(object):

    def __init__(self, 
                 layer: int,
                 index: int):

        self.layer = layer
        self.index = index

        self.identifier = f'{layer}.{index}'

        self.activation: float = 0
        self.grad: float = 0
    
    def __str__(self):
        return self.identifier
    
    def set_activation(self, activation: float):
        self.activation = activation

    
class NeuralNet(object):

    def __init__(self, nlayers: int,
                 input_dim: tuple, 
                 m_neurons_per_layer: List[int],
                 out_dim: tuple,
                 weights_list: List[np.ndarray] = None,
                 biases_list: List[np.ndarray] = None):
        
        net: Dict[int, List[Neuron]] = {}
        assert input_dim == m_neurons_per_layer[0]
        assert len(m_neurons_per_layer) == nlayers
        for l in range(nlayers):
            net[l] = [Neuron(l, i) for i in range(m_neurons_per_layer[l])]

        self.in_dim = input_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.net = net

        self.weights, self.biases = self.initialize_weights(weights_list=weights_list, biases_list=biases_list)
    
    def initialize_weights(self,
                           weights_list: np.ndarray = None,
                           biases_list: np.ndarray = None,
                           seed: int = 1234):
        
        if weights_list and biases_list:
            return weights_list, biases_list
        
        rndst = np.random.RandomState(seed = seed)
    
        dim1 = self.in_dim
        weights_list: List[np.ndarray] = []
        biases_list: List[np.ndarray] = []
        for l in range(self.nlayers - 1):
            dim2 = len(self.net[l+1])
            weights_list.append(rndst.randn(dim2, dim1))
            biases_list.append(rndst.randn(dim2, 1))
            dim1 = deepcopy(dim2)
                  
        return weights_list, biases_list

    def convert_list_to_np(self, layer: int):
        return np.array([i.activation for i in self.net[layer]])
    
    def convert_np_to_activations(self, layer: int, activations_as_npy: np.ndarray):
        neurons = self.net[layer]
        for i, neuron in enumerate(neurons):
            neuron.set_activation(activations_as_npy[i])            
    
    def forward(self, input: np.ndarray):
        tmp = input.copy()
        for l in range(self.nlayers - 1):
            out: np.ndarray = np.dot(self.weights[l], tmp)
            out = out + self.biases[l]
            self.convert_np_to_activations(layer = l, activations_as_npy=tmp)
            tmp = out.copy()
        
        return out


def test_net():

    weights_list = [
        np.array([[2, 2, 2],
                    [3, 3, 3,],
                    [4, 4, 4],
                    [5, 5, 5]]),
        np.array([[2,2,2,2],
                  [3,3,3,3]])
    ]

    biases_list = [
        np.array([[1],
                   [1],
                   [1],
                   [1]]),
        np.array([[1],
                  [1]])
    ]

    net = NeuralNet(nlayers = 3, 
                    input_dim = 3,
                    m_neurons_per_layer=[3, 4, 2],
                    out_dim=2,
                    weights_list=weights_list,
                    biases_list=biases_list)
    
    input = np.array([[1], 
                      [2], 
                      [3]])

    output = net.forward(input)

    print(output)

if __name__ == "__main__":
    test_net()