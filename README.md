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

## Configure your training

First, you must create a configuration structure:
```
trainig_config_t config;
```

### Configure the debug

Three level of debug are available.
* Level 0 : No debugging. Just print the errors.
* Level 1 : For each epoch of training, the training error is printed.
* Level 2 : The training error will be saved in a text file at each epoch.

If you use the debug level 2, you can also set the debug file name. The default one is "debug.txt".

```
config.debug_level = 2;
config.debug_file = "name.txt";
```

### Configure the stop condition

You can either set a maximum number of epoch or set the target training error to reach.
When one of these conditions is reached, the training will stop.

With respect to the error of training, there are two ways to calculate this error: mean square error (mse) and average mean error (mae).

```
config.error_type = mae;
config.stop_error = 0.00001;

config.nb_epochs = 5000;
```

### Configure the training process

Six training processes are available. For each of them, the "step" parameter can be configured.
For the GD_momentum and GD_nesterov processes, a momentum factor can also be configured.

```
config.training_type = GD_adagrad;
config.step = 0.1;
config.momentum_factor = 0.05;
```
n.b : If the parameters "step" and "momentum_factor" are not specified, default values will be used depending on the training process.

### Set the configuration

The configuration must be set on a trainer object. If you want a default training, you don't need to set any configuration.
```
Trainer trainer;
trainer.set_config(config);
```
