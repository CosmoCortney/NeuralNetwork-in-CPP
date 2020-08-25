# NeuralNetwork-in-CPP

This neural network design is written in C++. I've created this by following a book called "Make Your Own Neural Network" by Tariq Rashid (ISBN: 9781530826605) and this article https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

The XOR problem can be solved with a 2x2x1 network running 10'000 iteration in about 100ms (LR = 0.3).

## Usage
The network is initialized by the following:
The given topology as an std::vector<int>
Input data via std::vector<Matrix>
Target data via std::vector<Matrix>
Learning rate (float)
  
  NeuralNetwork nn(topology, LR, xorInputs, xorTargets); 
  
The Network uses the standard Sigmoid activation function. You can select different activation functions. Create an std::vector<int> and push back the activation function aliases.

std::vector<int> aTypes;
aTypes.push_back(SIGMOID); //use Sigmoid function for layer 1
aTypes.push_back(TANH);    //use TanH function for layer 2
nn.setActivationTypes(aTypes);
