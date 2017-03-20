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

def run_model(random_lr, random_K, random_num_layers, Trainseq, Validseq, Testseq, mbatchsize, gpus, task, data,vnum_steps = 2000, tnum_steps = 2000, min_params = 1e5, max_params = 1e6, count_mode = False, wform = None, model = None, p1 = 0.9, p2 = 0.9, EP = 1002,server = 'malleus'):
    
    #set the experiment parameters
    model = model #model, options are 'lstm', 'gated_w', 'gru' (to be implemented), 'conv_w' (to be added), 'vanilla_RNN', 'mod_lstm' 
    wform = wform #form of the W matrix in the model
    EP = EP#number of iterations
    LR = random_lr #learning rate
    num_layers = random_num_layers #number of network layers
    K = random_K #number of hidden units per layer 
    if task in ['text','music','speech']:
        num_steps = 200 #this is the mini-batchsize 
    elif task in ['digits']:
        num_steps = 28
    ntaps = 100 #this is the filter length for the conv_w model (only required is model = 'conv_w') 
    build_with_dropout = True 
    p1 = p1 # Input keep-probability
    p2 = p2 # Output keep_probability
    Tgen = 1293012931#length of validation sequence in the text task 
    if task == 'music':
        outstage = 'sigmoid' #output stage non-linearity 
    elif task in ['text','digits','speech']:
        outstage = 'softmax'
    save = True #if this is false the code does not save learned model variables  
    train_fun = 'sgd' #? 
    mbatchsize = mbatchsize - (mbatchsize%num_steps)
    
    if task in ['music','text']:
        X_train = Trainseq[:,:-1]
        Y_train = Trainseq[:,1:]

    elif task in ['digits','speech']:
        X_train = Trainseq[0] #concatenated images
        Y_train = Trainseq[1] #labels
        L2 = Y_train.shape[0] #this is the number of classes

    L = X_train.shape[0]

    end_idx = (X_train.shape[1])%len(gpus)
    if end_idx!=0:
        X_train = X_train[:,:-end_idx]
        Y_train = Y_train[:,:-end_idx]

    # Define network parameters
    if len(gpus)==1 and train_fun=='sgd':
        batchsize = mbatchsize 
        T = batchsize
    elif len(gpus)!=1 and train_fun=='sgd':
        batchsize = mbatchsize*len(gpus)
        T = int( batchsize/len(gpus))

    # Reset graph
    tf.reset_default_graph()


    # Create class object
    myrnn = infinite_tap_rnn(K,L,ntaps, T, gpus, 'CrossEnt')
    myrnn.num_steps = num_steps
    myrnn.num_layers = num_layers 
    myrnn.outstage = outstage
    myrnn.build_with_dropout = build_with_dropout
    myrnn.p1 = p1
    myrnn.p2 = p2
    myrnn.wform = wform
    myrnn.case = model
    myrnn.task = task
    if task == 'digits':
        myrnn.L2 = L2
        myrnn.y = tf.placeholder(tf.float32, [L2,int(mbatchsize/28)], name = "out_placeholder") #overwrite on self.y definition for the digit case 
    elif task == 'speech': 
        myrnn.L2 = L2
        myrnn.y = tf.placeholder(tf.float32, [L2,int(mbatchsize)], name = "out_placeholder") #overwrite on self.y definition for the digit case 

    # list of recursive models implemented in the code
    myrnn.recursive_models = ['lstm','gated_w','vanilla_rnn','gated_wf','mod_lstm']
    myrnn.bidirectional_models = ['bi_lstm', 'bi_gated_w', 'bi_vanilla_rnn', 'bi_gated_wf', 'bi_mod_lstm'] 
    myrnn.vary_penalty = True

    
    # Initialize parameters
    myrnn.initialize_parameters(distribution = tf.random_normal, m = 0.0, c = 0.1, case = model )

    # Define model and updates
    myrnn.multi_gpu_grad(g1 = tf.tanh, g2 = tf.nn.softmax, opt = 'Adam', learning_rate = LR, case = model, count_mode = count_mode )

    if task == 'text':
        #for test sequence
        myrnn.final_xtest = tf.placeholder( tf.float32, [L,Testseq.shape[1]-1] , name = 'x_test')
        with tf.device('/gpu:'+str(myrnn.gpus[0])):
            myrnn.final_ytest, _ = myrnn.model( myrnn.final_xtest , case = model , train_mode = False, finalvalid_mode = True, custom_numsteps = tnum_steps) 
            myrnn.final_longytest, _ = myrnn.model( myrnn.final_xtest , case = model , train_mode = False, finalvalid_mode = True, custom_numsteps = Testseq.shape[1]-1) 
     
        #for validation sequence
        myrnn.final_xvalid = tf.placeholder( tf.float32, [L,Validseq.shape[1]-1] , name = 'x_valid')
        with tf.device('/gpu:'+str(myrnn.gpus[0])):
            myrnn.final_yvalid, _ = myrnn.model( myrnn.final_xvalid, case = model , train_mode = False, finalvalid_mode = True, custom_numsteps = vnum_steps) 
            myrnn.final_longyvalid, _ = myrnn.model( myrnn.final_xvalid, case = model , train_mode = False, finalvalid_mode = True, custom_numsteps = Validseq.shape[1]-1) 
    elif task in ['digits','speech']:
        #for test sequence
        myrnn.xtest = tf.placeholder( tf.float32, [L,Testseq[0].shape[1]] , name = 'x_test')
        with tf.device('/gpu:'+str(myrnn.gpus[0])):
            myrnn.ytest = myrnn.model( myrnn.xtest , case = model , train_mode = False) 
     

        #for validation sequence
        myrnn.xvalid = tf.placeholder( tf.float32, [L,Validseq[0].shape[1]] , name = 'x_valid')
        with tf.device('/gpu:'+str(myrnn.gpus[0])):
            myrnn.yvalid = myrnn.model( myrnn.xvalid, case = model , train_mode = False)


    ##########

    init = tf.initialize_all_variables()
    #saver = tf.train.Saver(myrnn.model_variables.extend(myrnn.output_stage_vars ))

    #init = tf.global_variables_initializer() # Use for older version of tensorflow in nmf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    sess = tf.InteractiveSession(config = config)
    sess.run(init)
    #with tf.Session() as sess:
    
    trainable_vars = tf.trainable_variables()
    nparams = [np.prod(trainable_vars[i].get_shape().as_list()) for i in range(len(trainable_vars))]
    tnparams = np.sum(nparams)
    print('Total number of parameters to be trained = ' + str(tnparams) )
    if count_mode == True:
        return tnparams#If the count mode is on just count the parameters and return  

    if (tnparams < min_params) | (tnparams > max_params):
        raise ValueError
    
    # Train the network 
    start_time = time.time()
    print('Training ' + model + ' with learning rate '+ str(LR) + ' K = ' + str(K) +'num_layers = ' + str(num_layers)) 
    valid_logls, tst_logls, costs, all_times = myrnn.sgd_train(X_train, Y_train, sess, ep = EP, batchsize = batchsize, verbose = True, plot = False, Validseq = Validseq, Testseq = Testseq, task = task, Tgen = Tgen)
    end_time = time.time() - start_time
    print('Computation time: ' + str(end_time))
   
    #get the trainable variables after training
    trainable_vars = tf.trainable_variables()

        
    ##
    max_valid = np.max(np.array(valid_logls))
    max_test = np.max( np.array(tst_logls))
    dictionary = {'valid':  np.array(valid_logls), 
                  'max_valid':max_valid , 
                  'tst':  np.array(tst_logls), 
                  'max_test':max_test, 
                  'tr': np.array(costs), 
                  'K':K,
                  'L':L,
                  'num_layers': num_layers, 
                  'LR':LR, 
                  'all_times':all_times, 
                  'build_with_dropout':build_with_dropout, 
                  'p1':p1, 
                  'p2':p2, 
                  'wform':wform,
                  'num_steps': num_steps, 
                  'model':model, 
                  'data':data, 
                  'EP':EP,
                  'vnum_steps': vnum_steps,
                  'tnum_steps': tnum_steps,
                  'nparams':nparams,  
                  'tnparams':tnparams,
                  'server': server,
                  'vary_penalty':myrnn.vary_penalty}
                    
    #np.save( savedir+data+'_'+model + '_'+ str(round(time.time()))  , dictionary) 
    
    return dictionary, model, data, build_with_dropout, wform

def model_driver(d):
    'This function builds the computation graph for the specified model
    The input is the dictionary d with fields:
        model (model to be learnt), wform (form of the W matrices - full/diagonal/scalar/constant) 
        K (number of states), L1 (input dimensions), L2 (output dimensions), numlayers (number of layers) '
    # Reset graph
    tf.reset_default_graph()
    
    rnn1 = rnn( model = d['model'], 
                wform = d['wform'],
                K = d['K'], 
                L1 = d['L1'], 
                L2 = d['L2'], 
                numlayers = d['numlayers']) 
    pdb.set_trace()

    


###################### Main Wrapper Function begins here ########################

def main(dictionary):

    # first get the data and the resulting model parameters  
    data, parameters = load_data(task = dictionary['task'], data = dictionary['data'])
    dictionary.update(parameters) # add the necessary model parameters to the dictionary here
    
    ### next thing is determining the hyperparameters
    performance_records = []
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
                                    'num_layers': random_num_layers} ) 
                run_info = model_driver(dictionary) 
                   
                #append the performance records
                performance_records.append(run_info)
            
                break
        
            except ValueError:
                print('This parameter configuration is not valid!!!!!!!') 

        # Save in directory
        savedir = 'experiment_data/'
        np.save( savedir + server+ '_data_' + data + '_model_' + model + '_'+ wform +'_dropout_'+str(build_with_dropout)+'_gpu_'+str(gpus[0])+ '_' + timestamp, performance_records)
    return performance_records

def parameter_counter(model, wform, random_lr, random_K, random_num_layers, Trainseq, Validseq, Testseq, mbatchsize, gpus, task, data):
        #this function is to be used for getting a sense for parameter ranges
        nparams = run_model_v2(random_lr, random_K, random_num_layers, Trainseq, Validseq, Testseq, batchsize, gpu, task, data, min_params = 4e5, max_params = 6e5, count_mode = True, model = model, wform = wform)
        print('This model has ' + str(nparams) + ' parameters')
        return nparams

################# Main Processing function begins here ################

desktop = 1
server = socket.gethostname() 
if desktop:
    import matplotlib.pyplot as plt
    print('WARNING YOU ARE NOT DOING PARALLEL COMPUTATION!!!!!!!!!')
    dictionary = {'seedin' : [1144, 1521], 
                'task' : 'digits', 
                'data' : 'mnist',
                'model': 'bi_mod_lstm',
                'wform':'full',
                'num_configs' : 60, 
                'EP' : 151,
                'dropout' : [0.9, 0.9],
                'gpus' : [1], 
                'server': socket.gethostname}
    perfs = main(dictionary)

else:

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
