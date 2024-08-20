
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)
E = math.e
class layer_dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.10*np.random.randn(n_inputs, n_neurons) #shaping like this so that we donot have to transpose everytime.
    self.biases = np.zeros((1,n_neurons))
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  def backward(self, dvalues): #Calculating the gradients of lodd function WRT weights, biases and inputs.
        self.dweights = np.dot(self.inputs.T, dvalues) # dvalues is the gardient of loss WRT output of the current layer i.e output
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLu:
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0,inputs)
  def backward(self, dvalues):
    self.dinputs =dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0 #Sets the gradient to 0 where the original input value was less than or equal to 0.

class Activation_Softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values /np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities
  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues) #Creating the same shape as dvalues
    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): #run for all the samples in batch
      single_output = single_output.reshape(-1, 1) #have to reshape it into column vector so that we can perform matrix operations.
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)



class Loss_CategoricalCrossEntropy():
  def forward(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

    #if given in scalar form
    if len(y_true.shape) == 1:
      correct_confidence = y_pred_clipped[range(samples), y_true]
    #if given in one-hot encoding
    elif len(y_true.shape) == 2:
      correct_confidence = np.sum(y_pred_clipped*y_true, axis =1)

    negative_log_likelihoods = -np.log(correct_confidence)
    return negative_log_likelihoods

  def backward(self, dvalues, y_true):
    samples = len(dvalues)
    labels = len(dvalues[0])


    if len(y_true.shape) == 1:
      y_true = np.eye(labels)[y_true] #if y_true is not one-hot encoded, this converts it

    self.dinputs = -y_true / dvalues
    self.dinputs = self.dinputs / samples

  def calculate(self, output, y):
    sample_losses = self.forward(output, y)
    data_loss = np.mean(sample_losses)
    return data_loss

def create_data(points, classes):
  X = np.zeros((points*classes,2))
  Y = np.zeros(points*classes,dtype='uint8')
  for class_number in range(classes):
    ix = range(points*class_number, points *(class_number+1))
    r = np.linspace(0.0,1,points)
    t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
    X[ix] =np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
    Y[ix] =class_number
  return X,Y
X,y = create_data(100,3)

layer1 = layer_dense(2, 5)  #Setting inputs to 4 as we have 4 inputs


dense1 = layer_dense(2,3)
activation1 = Activation_ReLu()

dense2 =layer_dense(3,3)
activation2 = Activation_Softmax()



for i in range(0,1000):
  dense1.forward(X)
  activation1.forward(dense1.output)

  dense2.forward(activation1.output)
  activation2.forward(dense2.output)

  #print(activation2.output)

  loss_function = Loss_CategoricalCrossEntropy()
  loss = loss_function.calculate(activation2.output,y)

  print("loss : " ,loss)


  loss_function.backward(activation2.output, y)
  activation2.backward(loss_function.dinputs)
  dense2.backward(activation2.dinputs)
  activation1.backward(dense2.dinputs)
  dense1.backward(activation1.dinputs)

  learning_rate = 0.1

  dense1.weights += -learning_rate * dense1.dweights
  dense1.biases += -learning_rate * dense1.dbiases
  dense2.weights += -learning_rate * dense2.dweights
  dense2.biases += -learning_rate * dense2.dbiases
