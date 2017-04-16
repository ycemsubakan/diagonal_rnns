# Software requirements: 
* Python v > 3.5
* Numpy v > 1.12 
* Tensorflow v > 1.0 
* Pandas > v -0.18
* scikit-learn v > 0.17.1 (for feature extraction in text task) 

# SYNOPSIS for version 0.1
* This is the code used in the paper *Diagonal RNNs in Symbolic Music Modeling*. This code can has regular Vanilla RNN, LSTM, GRU and their diagonal versions. We also have the bi-directional implementations of these models. We have sequence to sequence (suitable for e.g. sequence prediction tasks) and sequence to vector (suitable for e.g. sequence classification)    

* In this version we have support for the following experimental setups: 
	1. Sequence prediction on symbolic music datasets downloaded from this [link](http://www-etud.iro.umontreal.ca/~boulanni/icml2012), 
	2. Digit classification on the MNIST dataset with bi directional RNNs. We use tensorflow's default MNIST data.   
	3. Phoneme classification on the TIMIT dataset. We use the  data preparation code for the TIMIT dataset, which save the data in a .pickle file. 
	4. The text prediction task will soon be implemented. 

## Summary of the input and outputs of the code: 
* The specifications for the particular run, including the task, dataset, model, form of the recurrent matrices, the device to run on, initialization, optimizer are given inside the dictionary named 'input_dictionary' as an argument to the main function. The outputs are written in a .npy file after being done with each random hyper-parameter configuration. 

* The input interface is in main.py. The RNN classes and functions are in rnns.py 

## Loading custom dataset:  
* The function 'load_data()' in rnns.py handles the data loading. The users can customize this function accordingly.    
