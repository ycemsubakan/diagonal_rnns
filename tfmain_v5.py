import sklearn.feature_extraction.text as ft 
import numpy as np
import tensorflow as tf
import time
import pdb 
import os 
import itertools
import socket
import multiprocessing
from multiprocessing import Process
from multiprocessing.pool import Pool
import _pickle as cPickle
from tfrnns_v5 import *

def model_driver(d,data):
    """This function builds the computation graph for the specified model
    The input is the dictionary d with fields:
        model (model to be learnt), wform (form of the W matrices - full/diagonal/scalar/constant) 
        K (number of states), L1 (input dimensions), L2 (output dimensions), numlayers (number of layers)
    """
    # Reset graph
    tf.reset_default_graph()
    

    #build the first graph
    if d['wform_global'] == 'diagonal_to_full':
        d['wform'] = 'diagonal' # set the first rnn to be diagonal
        rnn1 = rnn( model_specs = d) 
        rnn1_handles = rnn1.build_graph()  
       
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
        d['wform'] = 'diagonal_to_full' # set the wform right for the 2nd rnn
        with tf.variable_scope('rnn2'):
            rnn2 = rnn( model_specs = d, initializer = vars_np)  
            rnn2_handles = rnn2.build_graph()  


        #vars_to_init = [var for var in tf.all_variables() if 'Wfull' in var.name]  
        
        #before = sess.run(tf.trainable_variables()[1]) 
        #sess.run(tf.variables_initializer(vars_to_init))

        sess.run(tf.initialize_all_variables())
        #after = sess.run(tf.trainable_variables()[1])
        
        #diff = np.sum( np.abs( before - after ) ) 
        

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
        rnn1 = rnn( model_specs = d) 
        rnn1_handles = rnn1.build_graph()  
       
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True

        sess = tf.Session(config = config) 
        sess.run(tf.initialize_all_variables())

        #train the first model 
        all_times, tr_logls, test_logls, valid_logls = rnn1.optimizer(data = data, rnn_handles = rnn1_handles, sess = sess) 

        max_valid = np.max(np.array(valid_logls))
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
    data, parameters = load_data(task = dictionary['task'], data = dictionary['data'])
    dictionary.update(parameters) # add the necessary model parameters to the dictionary here
    
    ### next thing is determining the hyperparameters
    np.random.seed( dictionary['seedin'][0] )
    tf.set_random_seed( dictionary['seedin'][1] ) 
    timestamp = str(round(time.time()))

    # lower and upper limits for hyper-parameters to be sampled
    lr_min ,lr_max = -4, -2 
    num_layers_min, num_layers_max = 1, 3
    K_min, K_max, min_params, max_params = return_Klimits(
            model = dictionary['model'], 
            wform = dictionary['wform'], 
            data = dictionary['data']) 

    countmode = False

    ##################This part is just for counting##################
    ### deprecated  - update this 
    if countmode:
        n1 = parameter_counter(model, wform, lr_min, K_min, num_layers_max)  
        pdb.set_trace()
        n2 = parameter_counter(model, wform, lr_min, K_max, num_layers_min) 
        print('For this model (' +model + wform+ 
                ') number of parameters Lower limit is ' + str(n1) + 
                ' upper limit is ' + str(n2) + 
                '\n|||||||| by the way'+ ' min_params= ' + str(min_params) +
                ' max_params= ' + str(max_params) +  
                ' for this dataset which is ' + data 
                + '|||||\n  The code will run on gpu' + str(gpus) ) 
        pdb.set_trace()
    ##################################################################    

    records = [] #information will accumulate in this
    for i in range(dictionary['num_configs']):
        while True:  
            try:
                random_lr, random_K, random_num_layers = generate_random_hyperparams(
                        lr_min =  lr_min, lr_max = lr_max, 
                        K_min = K_min, K_max = K_max , 
                        num_layers_min = num_layers_min, 
                        num_layers_max = num_layers_max)
                
                dictionary.update({'LR':random_lr, 
                                    'K':random_K, 
                                    'num_layers': random_num_layers,
                                    'min_params': min_params,
                                    'max_params': max_params} ) 
                run_info = model_driver(d = dictionary, data = data) 
                   
                #append the performance records
                records.append(run_info)
            
                break
        
            except num_paramsError:
                print('This parameter configuration is not valid!!!!!!!') 

        # Save in directory
        savedir = 'experiment_data/'
        np.save( savedir + dictionary['server'] 
                + '_data_' + dictionary['data'] 
                + '_model_' + dictionary['model'] 
                + '_'+ dictionary['wform'] 
                +'_gpu_'+ str(dictionary['gpus'][0]) 
                + '_' + timestamp, records)
    
    return records

def parameter_counter(model, wform, random_lr, random_K, random_num_layers, Trainseq, Validseq, Testseq, mbatchsize, gpus, task, data):
        #this function is to be used for getting a sense for parameter ranges
        nparams = run_model_v2(random_lr, random_K, random_num_layers, Trainseq, Validseq, Testseq, batchsize, gpu, task, data, min_params = 4e5, max_params = 6e5, count_mode = True, model = model, wform = wform)
        print('This model has ' + str(nparams) + ' parameters')
        return nparams

desktop = 1
if desktop:
    import matplotlib.pyplot as plt
    print('WARNING YOU ARE NOT DOING PARALLEL COMPUTATION!!!!!!!!!')
    dictionary = {'seedin' : [1144, 1521], 
                'task' : 'digits', 
                'data' : 'mnist',
                'model': 'bi_mod_lstm',
                'wform': 'full',
                'wform_global' : 'full',
                'num_configs' : 60, 
                'EP' : 60,
                'dropout' : [0.9, 0.9],
                'gpus' : [2], 
                'server': socket.gethostname(),
                'verbose':False }
    perfs = main(dictionary)

else:
    #this is all deprecated
    with multiprocessing.pool.Pool(5) as pool:
        mydict0 = {'seedin' : [1532,61245], 'task' : 'music', 'mode' : 'start', 'num_configs' : 60, 'gpus' : [0], 'model':model,'wform':'full' ,'server':server }
        mydict1 = {'seedin' : [855,8256], 'task' : 'music', 'mode' : 'start', 'num_configs' : 60, 'gpus' : [1], 'model':model, 'wform':'diagonal','server':server}
        #mydict2 = {'seedin' : [8125,539], 'task' : 'music', 'mode' : 'start', 'num_configs' : 30, 'gpus' : [2]}
        #mydict3 = {'seedin' : [215,41236], 'task' : 'music', 'mode' : 'start', 'num_configs' : 30, 'gpus' : [3]}



        perf0 = pool.apply_async(mainwrapper, args = [mydict0])
        perf1 = pool.apply_async(mainwrapper, args = [mydict1])
        #perf2 = pool.apply_async(mainwrapper, args = [mydict2])
        #perf3 = pool.apply_async(mainwrapper, args = [mydict3])

        p0 = perf0.get()
        p1 = perf1.get()
        #p2 = perf2.get()
        #p3 = perf3.get()
