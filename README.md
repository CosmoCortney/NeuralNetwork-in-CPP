# NeuralNetwork-in-CPP
WORK IN PROGRESS

This neural network design is written in C++. I've created this by following a book called "Make Your Own Neural Network" by Tariq Rashid (ISBN: 9781530826605). Later on is given an example in Python. I tried to implement the whole logic in C++ which works fine at some point. Feeding forward with the same example values resulted in the same results as given in the book. However, there seems to be something wrong with the implementation of the back-propagation method or any function this method calls.

I trained it with the XOR problem which leads to strange results:
Utilizing [0, 1], [1, 0] or [1, 1] as input values results in very similar, very small values. Using [0, 0] results in a value much bigger (still smaller than 1.0).

The source code is split into 3 parts:
The AnnMaths namespace featuring useful functions,
The NeuralNetwork class for creating instances of the NeuralNetwork
and the main.cpp which contains a training example.
