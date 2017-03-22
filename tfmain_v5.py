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
    
    rnn1 = rnn( model_specs = d) 
    rnn1_handles = rnn1.build_graph()  
   
    # load the data
    tr = SimpleDataIterator(data['Train'])
    tst = SimpleDataIterator(data['Test'])
    valid = SimpleDataIterator(data['Validation'])

    config = tf.ConfigProto(log_device_placement = True)
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    all_times, tr_logls, test_logls, valid_logls = [], [], [], [] 
    with tf.Session(config = config) as sess:
        sess.run(tf.initialize_all_variables())
        for ep in range(d['EP']):
            t1, tr_logl = time.time(), []
            while tr.epochs == ep:
                trb = tr.next_batch(n = d['batchsize'], task = d['task'], verbose = d['verbose'])      
                feed = {rnn1_handles['x']:trb[0], rnn1_handles['y']:trb[1], rnn1_handles['seq_lens']:trb[2], rnn1_handles['dropout_kps']:d['dropout'] }         
                tr_cost, tr_logl_temp, _ = sess.run( [rnn1_handles['cost'], rnn1_handles['accuracy'], rnn1_handles['train_step']], feed) 
                tr_logl.append(tr_logl_temp)

                if d['verbose']:
                    print("Training cost = ", tr_cost, " Training Accuracy = ", tr_logl_temp)
            t2 = time.time()

            #get training and test accuracies
            tsb = tst.next_batch( n = tst.size, task = d['task'])  
            vlb = valid.next_batch( n = valid.size, task = d['task'])  

            tst_feed = {rnn1_handles['x']: tsb[0], rnn1_handles['y']: tsb[1], rnn1_handles['seq_lens']: tsb[2], rnn1_handles['dropout_kps']:np.array([1,1])} 
            vld_feed = {rnn1_handles['x']: vlb[0], rnn1_handles['y']: vlb[1], rnn1_handles['seq_lens']: vlb[2], rnn1_handles['dropout_kps']:np.array([1,1])} 
   
            tr_logl = np.mean(tr_logl)
            tst_logl = sess.run( rnn1_handles['accuracy'], tst_feed ) 
            vld_logl = sess.run( rnn1_handles['accuracy'], vld_feed ) 
    
            print("Iteration = ", ep, 
                  "Training Accuracy", np.mean(tr_logl),
                  ",Test Accuracy = ", tst_logl, 
                  ",Validation Accuracy = ", vld_logl, 
                  ",Elapsed Time = ", t2-t1) 

            all_times.append(t2-t1)
            tr_logls.append(tr_logl)
            test_logls.append(tst_logl)
            valid_logls.append(vld_logl)

        max_valid = np.max(np.array(valid_logls))
        max_test = np.max( np.array(test_logls))
        res_dictionary = {'valid':  np.array(valid_logls), 
                      'max_valid':max_valid , 
                      'tst':  np.array(test_logls), 
                      'max_test':max_test, 
                      'tr': np.array(tr_logls), 
                      'all_times':all_times, 
                      'tnparams':rnn1.tnparams}

        d['wform'] = 'diagonal_to_full'
        rnn2 = rnn( model_specs = d) 
        rnn2_handles = rnn2.build_graph()  


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
                'wform':'diagonal',
                'num_configs' : 60, 
                'EP' : 150,
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
