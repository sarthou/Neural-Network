[![Build Status](https://travis-ci.org/sarthou/Neural-Network.svg?branch=master)](https://travis-ci.org/sarthou/Neural-Network) 

# Neural-Network

This project is an easy tool to create and train neurals networks in C++. Fully modular, you will be able to create a personalized network with several layers and many types of perceptrons.

The SNN V2 is six times faster than the previous one and can perform 16,000 individual data in less than 200ms without optimization on i5 6300hq.

![SNN training](https://github.com/sarthou/Neural-Network/blob/master/images/ele.gif "SNN during outline training")
![training example](https://github.com/sarthou/Neural-Network/blob/master/images/training_example.png "training example") <!-- .element height="30%" width="30%" -->

## Create your network

First, you must define the number of internal layers and the number of perceptrons in each of them. For this we use a vector in which each element represents a layer and the value of the element represents the number of perceptrons in this layer.
In the example below, we have three internal layers. The closest layer to the input has two perceptrons and the furthest one has four.
```C++
vector<int> nb = {2, 6, 4 };
```
n.b : If this vector is empty, the network will only have one layer, that of output.

Now, you have to describe the kind of perceptrons you want to use on each layer. To do that, different methods are available.
```C++
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
```C++
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
```C++
Network net();
Network net(nb);
Network net(type);
Network net(type, param);
Network net(nb, type);
Network net(nb, type, param);
```

## Configure your training

First, you must create a configuration structure:
```C++
trainig_config_t config;
```

### Configure the debug

Three levels of debug are available.
* Level 0 : No debugging. Just print the errors.
* Level 1 : For each epoch of training, the training error is printed.
* Level 2 : The training error will be saved in a text file at each epoch.

If you use the debug level 2, you can also set the debug file name. The default one is "debug.txt".

```C++
config.debug_level = 2;
config.debug_file = "name.txt";
```

### Configure the stop condition

You can either set a maximum number of epoch or set the target training error to reach or allow the trainer to detect an evolution stop.
When one of these conditions is reached, the training will stop.

Regarding the error of training, there are two ways to calculate this error: mean square error (mse) and average mean error (mae).

```C++
config.error_type = mae;
config.stop_error = 0.00001;

config.nb_epochs = 5000;

config.stop_evolution = true;
```

### Configure the training process

Six training processes are available. For each of them, the "step" parameter can be configured.
For the GD_momentum and GD_nesterov processes, a momentum factor can also be configured.

```C++
config.training_type = GD_adagrad;
config.step = 0.1;
config.momentum_factor = 0.05;
```
n.b : If the parameters "step" and "momentum_factor" are not specified, default values will be used depending on the training process.

### Set the configuration

The configuration must be set on a Trainer object. If you want a default training, you don't need to set any configuration.
```C++
Trainer trainer;
trainer.set_config(config);
```

## Train your network

You must specify the input training data (P) and the expected output (T) related to the input.
The example below is to learn how to count in four-bit binary.
```C++
vector<float> d = { 0, 0, 0, 0, 0, 1, 1, 0 };
vector<float> c = { 0, 0, 0, 0, 1, 1, 0, 1 };
vector<float> b = { 0, 0, 1, 1, 0, 0, 0, 1 };
vector<float> a = { 0, 1, 0, 1, 0, 0, 0, 1 };
vector<vector<float> > P = { a, b, c, d};

vector<float> Ta = { 0, 1, 2, 3, 4, 12, 8, 7 };
vector<vector<float> > T = {Ta};

Matrix<float> P_mat(P.size(), P.at(0).size(), P);
Matrix<float> T_mat(T.size(), T.at(0).size(), T);
```

Or :

```C++
Matrix<float> P_mat(4, 8, {
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 1 },
		{ 0, 0, 1, 1, 0, 0, 0, 1 },
		{ 0, 1, 0, 1, 0, 0, 0, 1 }
	});
Matrix<float> T_mat(1, 8, {{ 0, 1, 2, 3, 4, 12, 8, 7 }});
```

Once you you have created your data, you just have to train your network.
```C++
trainer.train(&net, P_mat, T_mat);
```

n.b : You can continue to train your network as many times as you want.

## Use your training network

To use the network, you just have to put your input data into a vector and run the network.
```C++
Matrix<float> P(4, 9, {
		{ 1, 0, 0, 0, 1, 0, 0, 1, 1 },
		{ 0, 1, 0, 0, 0, 1, 1, 0, 1 },
		{ 0, 0, 1, 0, 1, 0, 1, 0, 1 },
		{ 0, 0, 0, 1, 0, 1, 0, 1, 1 }
	});

net.sim(&P);
```

You can get the output by using the get_output or get_output_cpy functions.
If you want to work only with integer data, you can use the round_output function.
Finally you can print the output data with the print_output function.

```C++
net.round_output();
net.print_output();
vector<vector<float> > out = net.get_output_cpy();
```

## Additionals features

* Use the print function on your network to have a textual description of the network.
```C++
net.print();
```
* You can copy your network into an other by using the operator = or the copy constructor.
```C++
Network cpy_net;
cpy_net = net;
```
