# Neural-Network

This project is an easy tool to creat and train neurals networks. Fully modular, you will be able to create a personalized network with several layers and many types of perceptrons.

## Create your network

First, you must define the number of internal layers and the number of perceptrons in each of them. For this we use a vector in which each element represents a layer and the value of the element represents the number of perceptrons in this layer.
In the example below, we have three internal layers. The layer closest to the input has two perceptrons and the furthest one has four.
```
vector<int> nb = {2, 6, 4 };
```
n.b : If this vector is empty, the network will only have one layer, that of output.

Now, you have to describe the kind of perceptrons you want to use on each layer. To do it, different methods are available.
```
// We describe the kind of perceptrons for each layer (internals + output)
vector<perceptron_type_t> type = {tanH, rectifier_param, logistic, identities };
// All internal layers will have logistic perceptrons and the output layer will have identities perceptrons
vector<perceptron_type_t> type = { logistic, identities };
// All layers will have identities perceptrons
vector<perceptron_type_t> type = { identities };
// All layers will have defaults perceptrons (logistic for internal and identities for output)
vector<perceptron_type_t> type = { };
```
n.b : 14 kind of perceptrons are available

If you use rectifier_param or ELU perceptrons, you need to set a parameter related to the mathematical activation function.
```
// We set the parameter for each layer (internals + output)
vector<double> param = { 2., 1., 1.5, 0};
// All internal layers will have the first parameter and the output layer will have the second
vector<double> param = { 1.5., 0.};
// All layers will have this parameter
vector<double> param = { 1. };
// All layers will have defaults parameter
vector<double> param = { };
```
n.b : Perceptrons that do not need parameters will not take into account what is defined

Depending on what you want, you do not have to create all these variables and you can use one of these methods to create your network.
```
Network net();
Network net(nb);
Network net(type);
Network net(type, param);
Network net(nb, type);
Network net(nb, type, param);
```
