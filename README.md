# Neural-Network-in-Go

This repository contains a simple artificial neural network implemented in Go from scratch. The network is built to recognize handwritten digits from the MNIST dataset. It was created following the concepts from this tutorial:
[Simple artificial neural network with Go](https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/)

To test the network, you can use the MNIST dataset in CSV format, which can be downloaded here:
[CSV for train/test the neural network](https://pjreddie.com/projects/mnist-in-csv/)

## Program Flags

The program accepts several command-line flags to customize its execution:

* `-csv <path>`: Specifies the directory where the MNIST CSV files (`mnist_train.csv` and `mnist_test.csv`) are located. Defaults to `"csv"`.
* `-t`: If provided, the program will **skip** the training phase and proceed directly to testing the network.
* `-s`: Tells the program to save the trained weights to a file (`savedData`) for future iterations.
* `-d`: Tells the program to delete any previously saved data (`savedData`) before starting. This forces the network to start training from scratch.