import nums
from nums import numpy as nps
from nums.core import settings
import math
import numpy as np

settings.backend_name = "gpu"
settings.device_grid_name = "packed"
nums.init()

rand_arr1 = nps.random.rand(100, 100)
rand_arr2 = nps.random.rand(100, 100)

# print((rand_arr1 + rand_arr2).get())

# x = nps.array([1, 2, 3])
# y = nps.array([4, 5, 6])
# z = x + y
# print(z.get())

class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)
    
    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)
    
    Some layers also have learnable parameters which they update during layer.backward.
    """
    def __init__(self):
        """Here we can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        pass
    
    def forward(self, input):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.
        
        To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        
        d loss / d x  = (d loss / d layer) * (d layer / d x)
        
        Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        
        If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]
        
        d_layer_d_input = nps.eye(num_units)
        
        return nps.dot(grad_output, d_layer_d_input) # chain rule

class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass
    
    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        relu_forward = (input > 0) * input
        return relu_forward
    
    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output*relu_grad 

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate
        self.weights = nps.random.randn(input_units,output_units) / math.sqrt(2/(input_units+output_units))
        self.biases = nps.zeros(output_units)
        
    def forward(self,input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b
        
        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        # print(self.weights)
        return nps.dot(input,self.weights) + self.biases
    
    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = nps.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = nps.dot(input.T, grad_output)
        # print(grad_output, grad_output.shape)
        grad_biases = nps.mean(grad_output, axis=0)*input.shape[0]
        # grad_biases = nps.sum(grad_output, axis=0)
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input


def softmax_crossentropy_with_logits(logits,reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    # logits_for_answers = logits[nps.arange(logits.shape[0]), reference_answers]
    logits_for_answers = nps.array(logits.get()[np.arange(logits.shape[0]), reference_answers.get()])
    #  [:logits.shape[0], reference_answers]
    # logits_for_answers = logits_for_answers[reference_answers]
    print(logits)
    xentropy = - logits_for_answers
    xentropy += nps.log(nps.sum(nps.exp(logits), axis=1))
    print(logits_for_answers)
    return xentropy

def mse(logits, reference_answers):
    return nps.mean((logits - reference_answers)**2)

def grad_mse(logits, reference_answers):
    return 2*nps.mean((logits - reference_answers))

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = nps.zeros_like(logits)

    ones_for_answers.get()[np.arange(logits.shape[0]),reference_answers.get()] = 1

    ones_for_answers = nps.array(ones_for_answers)

    softmax = nps.exp(logits) / nps.sum(nps.exp(logits), axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]

def forward(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer. 
    """
    activations = []
    input = X

    # Looping through each layer
    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    """
    Compute network predictions. Returning indices of largest Logit probability
    """
    logits = forward(network,X)[-1]
    print(logits.shape)
    res = []
    for i in range(logits.shape[0]):
        res.append(nps.argmax(logits[i]).get())
    
        
    return nps.array(res)

    # return nps.argmax(logits, axis=-1)

def train(network,X,y):
    """
    Train our network on a given batch of X and y.
    We first need to run forward to get all layer activations.
    Then we can run layer.backward going from last to first layer.
    After we have called backward for all layers, all Dense layers have already made one gradient step.
    """
    
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    # print(loss_grad)

    # loss = mse(logits, y)
    # loss_grad = grad_mse(logits, y)
    
    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        
    return nps.mean(loss)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = nps.random.permutation(inputs.shape[0])
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
def load_dataset(flatten=False):
    X, y = load_digits(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    # normalize x
    X_train, X_val, y_train, y_val = nps.array(X_train), nps.array(X_val), nps.array(y_train), nps.array(y_val)

    X_train = X_train.astype(float) / 255.
    X_val = X_val.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    # X_train, X_val = X_train[:-10000], X_train[-10000:]
    # y_train, y_val = y_train[:-10000], y_train[-10000:]


    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])

    return X_train, y_train, X_val, y_val



def main():
    X_train, y_train, X_val, y_val = load_dataset(flatten=True)
    network = []
    network.append(Dense(X_train.shape[1],100))
    network.append(ReLU())
    network.append(Dense(100,200))
    network.append(ReLU())
    # network.append(Dense(200,1))
    network.append(Dense(200,10))
    train_log = []
    val_log = []
    x_batch, y_batch = X_train, y_train
    for epoch in range(25):
        # for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=32,shuffle=True):
        
        loss = train(network,x_batch,y_batch)
        print("Loss: ", loss)
        
    train_log.append(nps.mean(predict(network,X_train)==y_train))
    val_log.append(nps.mean(predict(network,X_val)==y_val))
    
    # clear_output()
    # print("Epoch",epoch)
    # print("Train accuracy:",train_log[-1])
    # print("Val accuracy:",val_log[-1])
    # plt.plot(train_log,label='train accuracy')
    # plt.plot(val_log,label='val accuracy')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()
    
    

if __name__ == "__main__":
    main()