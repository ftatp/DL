import numpy as np
import pickle
import activationFuncs
import lossFuncs
import deriative

class Network():

##############################################################################################################
## Network setting functions
##############################################################################################################
    def __init__(self, input_size, output_size, num_of_hidden_layers, weight_init_std=0.01):
        #########################################
        ## Set activation, output, loss functions
        #########################################
        self.activation_func = activationFuncs.identity
        self.output_func = activationFuncs.identity
        self.loss_func = lossFuncs.mean_squared_error

        #########################################
        ## Set weights and bias
        #########################################
        
        print("Building network...")
        network = {}
        network['weights'] = []
        network['bias'] = []

        next_input_size = input_size
        for i in range(num_of_hidden_layers):
            next_output_size = 50#np.random.randint(100)
            weight = weight_init_std * np.random.randn(next_input_size, next_output_size)
            network['weights'].append(weight)
            network['bias'].append(np.zeros(next_output_size))
            next_input_size = next_output_size

        weight = weight_init_std * np.random.randn(next_input_size, output_size)
        network['weights'].append(weight)
        network['bias'].append(np.zeros(output_size))

        self.depth = len(network['bias'])
        self.network = network


    def set_network_by_values(self):
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

        self.depth = len(network['bias'])
        self.network = network



    def set_network_by_file(self, network_file):
        if network_file != None:
            print("network_file exists")
            print("Loading...")
            with open(network_file, 'rb') as f:
                self.network = pickle.load(f)


    def set_network_training_methods(self, activation_func=activationFuncs.identity, output_func=activationFuncs.identity, loss_func=lossFuncs.mean_squared_error):
        #########################################
        ## Set activation, output, loss functions
        #########################################
        self.activation_func = activation_func
        self.output_func = output_func
        self.loss_func = loss_func

##############################################################################################################
## Prediction functions
##############################################################################################################

    def forward(self, input_layer): # input layer values in np.array
        hidden_layers = self.network['weights']
        bias = self.network['bias']

        depth = self.depth

        for i in range(depth - 1):
            #print("Depth : ", i + 1)
            output_layer = np.dot(input_layer, hidden_layers[i]) + bias[i]
            #print("result : \n", output_layer)

            output_layer = self.activation_func(output_layer)
            #print("After activation result : ", output_layer)
            input_layer = output_layer
            #print("------------------------------------------------------------")
    

        #print("Depth : ", depth)
        output_layer = np.dot(input_layer, hidden_layers[depth - 1]) + bias[depth - 1]
        #print("result : \n", output_layer)
        output_layer = self.output_func(output_layer)
        #print("After activation result : ", output_layer)
        #print("------------------------------------------------------------")
        return output_layer
  
    def predict(self, input_layer):
        results = forward(input_layer, self.activation_func, self.output_func)
        return np.argmax(results)


##############################################################################################################
## Util functions
##############################################################################################################


    def loss(self, input_layer, t):
        output_layer = self.forward(input_layer)
        loss = self.loss_func(output_layer, t)

        return loss

    def numerical_gradient(self, input_layer, t):
        loss_W = lambda W: self.loss(input_layer, t)

        grads = {}
        grads['weights'] = []
        grads['bias'] = []
        for i in range(self.depth):
            grad = deriative.numerical_gradient(loss_W, self.network['weights'][i])
            #print("grad weight {0} shape".format(i), grad.shape)
            grads['weights'].append(grad)
            grad = deriative.numerical_gradient(loss_W, self.network['bias'][i])
            #print("grad bias {0} shape".format(i), grad.shape)
            grads['bias'].append(grad)

        
        return grads
