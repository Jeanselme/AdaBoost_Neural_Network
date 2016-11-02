# AdaBoost_Neural_Network
Test the adaBoost algorithm on multiple neural networks.

## Theory
In order to improve the backpropagation, it is more accurate to associate a stronger weight to inputs which add the most information. In other words, an input which has a wrong computed output.
In order to do so, we developp an algorithm inspired by AdaBoost.

## Execution
```
python3.5 MNIST_Analysis.py
```

## Results
The current example uses three neural networks with 25 hidden nodes with 10 iterations for backpropagation for each :  
Training set : 54 831 / 60 000  
Testing set : 9 091 / 10 000  
The result is a little less impressive than a unique neural network with 30 iterations, however, the proof shows that for a huge number of weak classifier the result should exponentially decrease.  

The second test is with 5 identic neural networks :  
Training set : 55 031 / 60 000  
Testing set : 9 176 / 10 000  

## Libraries
Needs struct, urllib.request, io, gzip, numpy and os. Executed with python3.5
