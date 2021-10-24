import numpy as np
import pickle
import activationFuncs
import lossFuncs

class Network():
    def __init__(self, network_file=None, activation_func=activationFuncs.identity, output_func=activationFuncs.identity, loss_func=lossFuncs.mean_squared_error):
        #########################################
        ## Set activation, output, loss functions
        #########################################
        self.activation_func = activation_func
        self.output_func = output_func
        self.loss_func = loss_func

        #########################################
        ## Set weights and bias
        #########################################
        if network_file != None:
            print("network_file exists")
            print("Loading...")
            with open(network_file, 'rb') as f:
                self.network = pickle.load(f)

        else:
            print("Building network...")
            network = {}
            network['hidden_layers'] = []
            network['bias'] = []

            network['hidden_layers'].append(np.array([
                [0.1, 0.3, 0.5],
                [0.2, 0.4, 0.6]
            ])) # input 층의 퍼셉트론 수 * output 층의 퍼셉트론 수
            network['bias'].append(np.array([0.1, 0.2, 0.3])) # 다음 층의 퍼셉트론 수 만큼

            network['hidden_layers'].append(np.array([
                [0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6]
            ]))
            network['bias'].append([0.1, 0.2])

            network['hidden_layers'].append(np.array([
                [0.1, 0.3],
                [0.2, 0.4]
            ]))
            network['bias'].append(np.array([0.1, 0.2]))

            self.depth = 3
            self.network = network

    def forward(self, input_layer): # input layer values in np.array
        hidden_layers = self.network['hidden_layers']
        bias = self.network['bias']

        depth = self.depth

        for i in range(depth - 1):
            print("Depth : ", i + 1)
            output_layer = np.dot(input_layer, hidden_layers[i]) + bias[i]
            print("result : \n", output_layer)

            output_layer = self.activation_func(output_layer)
            print("After activation result : ", output_layer)
            input_layer = output_layer
            print("------------------------------------------------------------")
    

        print("Depth : ", depth)
        output_layer = np.dot(input_layer, hidden_layers[depth - 1]) + bias[depth - 1]
        print("result : \n", output_layer)
        output_layer = self.output_func(output_layer)
        print("After activation result : ", output_layer)
        print("------------------------------------------------------------")
        return output_layer
  
    def predict(self, input_layer):
        results = forward(input_layer, self.activation_func, self.output_func)
        return np.argmax(results)

    def loss(self, x, t):
        z = self.forward(self, x, activationFuncs.identity, activationFuncs.identity)
        y = activationFuncs.softmax(x)

        
