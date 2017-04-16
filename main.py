import sklearn.feature_extraction.text as ft 
import numpy as np
import tensorflow as tf
import time
import pdb 
import os 
import socket
import multiprocessing
from multiprocessing import Process
from multiprocessing.pool import Pool
from rnns import *

def model_driver(d,data):
    """This function builds the computation graph for the specified model
    The input is the dictionary d with fields:
        model (model to be learnt), wform (form of the W matrices - full/diagonal/scalar/constant) 
        K (number of states), L1 (input dimensions), L2 (output dimensions), numlayers (number of layers)
    """
    # Reset graph
    tf.reset_default_graph()
    
    #build the first graph
    if d['wform_global'] == 'diag_to_full':
        d['wform'] = 'diagonal' # set the first rnn to be diagonal
        rnn1 = rnn( model_specs = d) 
        rnn1_handles = rnn1.build_graph()  
        print('This model has ',rnn1.tnparams, ' parameters')
        
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True

        sess = tf.Session(config = config) 
        sess.run(tf.initialize_all_variables())

        #train the first model 
        all_times, tr_logls, test_logls, valid_logls = rnn1.optimizer(data = data, rnn_handles = rnn1_handles, sess = sess) 

        vars_np = rnn1.save_modelvars_np(sess)
       

        print("Switching from diagonal to full") 
        #build the second graph
        d['wform'] = 'diag_to_full' # set the wform right for the 2nd rnn
        with tf.variable_scope('rnn2'):
            rnn2 = rnn( model_specs = d, initializer = vars_np)  
            rnn2_handles = rnn2.build_graph() 
        print('The first model had ',rnn1.tnparams,
              ' ,the second model has ',rnn2.tnparams,' parameters') 


        sess.run(tf.initialize_all_variables()) #initializes the first model
        
        all_times2, tr_logls2, test_logls2, valid_logls2 = rnn2.optimizer(data = data, rnn_handles = rnn2_handles, sess = sess, model_n = 2) 

        all_times.extend(all_times2), tr_logls.extend(tr_logls2)
        test_logls.extend(test_logls2), valid_logls.extend(valid_logls2)

        max_valid = np.max(np.array(valid_logls))
        max_test = np.max( np.array(test_logls))
        res_dictionary = {'valid':  np.array(valid_logls), 
                      'max_valid':max_valid , 
                      'tst':  np.array(test_logls), 
                      'max_test':max_test, 
                      'tr': np.array(tr_logls), 
                      'all_times':all_times, 
                      'tnparams':rnn1.tnparams}
    else:
        rnn1 = rnn( model_specs = d, initializer = d['init']) 
        rnn1_handles = rnn1.build_graph()  
       
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True

        with tf.Session(config = config) as sess:
            sess.run(tf.initialize_all_variables())

            #train the first model 
            all_times, tr_logls, test_logls, valid_logls = rnn1.optimizer(data = data, rnn_handles = rnn1_handles, sess = sess) 

            
        max_valid = np.max(np.array(valid_logls)) #kind of unnecessary
        max_test = np.max( np.array(test_logls))
        res_dictionary = {'valid':  np.array(valid_logls), 
                      'max_valid':max_valid , 
                      'tst':  np.array(test_logls), 
                      'max_test':max_test, 
                      'tr': np.array(tr_logls), 
                      'all_times':all_times, 
                      'tnparams':rnn1.tnparams}

    res_dictionary.update(d)

    return res_dictionary

def main(dictionary):

    # first get the data and the resulting model parameters  
    data, parameters = load_data(task = dictionary['task'], data = dictionary['data'] )
    dictionary.update(parameters) # add the necessary model parameters to the dictionary here
    
    ### next thing is determining the hyperparameters
    np.random.seed( dictionary['seedin'][0] )
    tf.set_random_seed( dictionary['seedin'][1] ) 
    timestamp = str(round(time.time()))

    # lower and upper limits for hyper-parameters to be sampled
    lr_min ,lr_max = dictionary['lr_min'], dictionary['lr_max']
    num_layers_min, num_layers_max = dictionary['num_layers_min'], dictionary['num_layers_max']
    K_min, K_max, min_params, max_params = return_Klimits(
            model = dictionary['model'], 
            wform = dictionary['wform'], 
            data = dictionary['data']) 

    records = [] #information will accumulate in this
    for i in range(dictionary['num_configs']):
        while True:  
            try:
                lr, K, num_layers, momentum = generate_random_hyperparams(
                        lr_min =  lr_min, lr_max = lr_max, 
                        K_min = K_min, K_max = K_max , 
                        num_layers_min = num_layers_min, 
                        num_layers_max = num_layers_max,
                        load_hparams = (dictionary['load_hparams'],i))
                
                print("Configuration ",i,
                      "K = ",K, 
                      "num_layers = ", num_layers,
                      "Learning Rate = ", lr,
                      "Momentum = ", momentum)  
                #this if clause enables the user to restart an experiment from a specific point the experiment
                if i < dictionary['start']:
                    break
                
                try: # Sometimes resources may get exhausted, this exception handles that 
                    dictionary.update({'LR': lr, 
                                        'K': K, 
                                        'num_layers': num_layers,
                                        'min_params': min_params,
                                        'max_params': max_params,
                                        'momentum' : momentum} ) 
                    run_info = model_driver(d = dictionary, data = data) 
                       
                    #append the performance records
                    records.append(run_info)
                except KeyboardInterrupt:
                    raise 
                except:
                    print('Resouces exhausted for this configuration, moving on')
                    #raise
                break
        
            except num_paramsError:
                print('This parameter configuration is not valid!!!!!!!') 

        # Save in directory
        savedir = 'experiment_data/'
        np.save( savedir + dictionary['server'] 
                + '_data_' + dictionary['data'] 
                + '_model_' + dictionary['model'] 
                + '_'+ dictionary['wform_global'] 
                + '_optimizer_' + dictionary['optimizer']
                +'_device_'+ dictionary['device']
                + '_' + timestamp, records)
    
    return records


#import matplotlib.pyplot as plt
wform = 'diagonal'
input_dictionary = {'seedin' : [1144, 1521], 
            'task' : 'music', 
            'data' : 'Nottingham',
            'model': 'mod_rnn',
            'wform': wform,
            'wform_global' : wform,
            'num_configs' : 60, 
            'start' : 0,  
            'EP' : 300,
            'dropout' : [0.9, 0.9],
            'device' : 'gpu:1', 
            'server': socket.gethostname(),
            'verbose': False,
            'load_hparams': False, #this loads hyper-parameters from a results file
            'count_mode': False, #if this is True, the code will stop after printing the number of trainable parameters
            'init':'xavier',
            'lr_min':-4, 'lr_max':-2,
            'num_layers_min':2, 'num_layers_max':3,
            'optimizer':'RMSProp',
            'notes':'I am trying RMS prop here. I am sampling momentums uniformly in this one. (This is what differs in this experiment from other RMSprop experiments) The upper limit for number of parameters is open. I am doing the gradient centralization thing also '}

perfs = main(input_dictionary)
