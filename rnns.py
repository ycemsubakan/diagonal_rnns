# This file has the rnn functions and classes  
import numpy as np
import tensorflow as tf
import pdb
import time
import pandas as pd
import _pickle as pickle
import itertools
import sys
import os


class ModRNNCell(tf.contrib.rnn.RNNCell):
    """Vanilla RNN Cell"""

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform = True, seed = 2, dtype=tf.float32), wform = 'full'):
        self._num_units = num_units
        self.init = initializer
        self.wform = wform 

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            init = self.init
            self.L1 = inputs.get_shape().as_list()[1]

            if self.wform == 'full':
                mats = tf.get_variable( "mats", 
                        shape = [self._num_units + self.L1, self._num_units],
                        initializer = init)
                bias = tf.get_variable("bias",
                        shape = [self._num_units],
                        initializer = init)

                res = tf.matmul( tf.concat([state,inputs], axis = 1),mats)
                new_h = tf.nn.tanh( tf.nn.bias_add( res,bias) )
            elif self.wform == 'diagonal':
                Wmat = tf.get_variable("W", 
                        shape = [self._num_units],
                        initializer = init)
                Umat = tf.get_variable( "U", 
                        shape = [self.L1, self._num_units],
                        initializer = init)
                bias = tf.get_variable("b",
                        shape = [self._num_units],
                        initializer = init)

                res = (Wmat*state) + tf.matmul(inputs, Umat)  
                new_h = tf.nn.tanh( tf.nn.bias_add( res, bias) ) 

            return new_h, new_h


class ModLSTMCell(tf.contrib.rnn.RNNCell):
    """Modified LSTM Cell """

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32), wform = 'diagonal'):
        self._num_units = num_units
        self.init = initializer
        self.wform = wform 

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            
            c, h = state
            init = self.init
            self.L1 = inputs.get_shape().as_list()[1]
           
            mats, biases = self.get_params_parallel()
            if self.wform == 'full' or self.wform == 'diag_to_full':
                
                res = tf.matmul(tf.concat([h,inputs],axis=1),mats)
                res_wbiases = tf.nn.bias_add(res, biases)
           
                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1) 
            elif self.wform == 'diagonal':
                h_concat = tf.concat([h,h,h,h],axis=1)

                W_res = tf.multiply(h_concat,mats[0])

                U_res = tf.matmul(inputs,mats[1])

                res = tf.add(W_res,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1) 

            elif self.wform == 'constant':
                    
                h_concat = tf.concat([h,h,h,h],axis=1)

                U_res = tf.matmul(inputs,mats)

                res = tf.add(h_concat,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1) 

            
            new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i)*tf.nn.tanh(j))

            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


    def get_params_parallel(self):
        if self.wform == 'full':
            mats = tf.get_variable("mats", 
                    shape = [self._num_units+self.L1,self._num_units*4], 
                    initializer = self.init )   
            biases = tf.get_variable("biases", 
                    shape = [self._num_units*4], 
                    initializer = self.init )   
        elif self.wform == 'diagonal':
            Ws = tf.get_variable("Ws", 
                    shape = [1,self._num_units*4], 
                    initializer = self.init )   
            Umats = tf.get_variable("Umats", 
                    shape = [self.L1,self._num_units*4], 
                    initializer = self.init )   
            biases = tf.get_variable("biases", 
                    shape = [self._num_units*4], 
                    initializer = self.init )   
            mats = [Ws, Umats] 
        elif self.wform == 'constant':
            mats = tf.get_variable("mats", 
                    shape = [self.L1,self._num_units*4], 
                    initializer = self.init )   
            biases = tf.get_variable("biases", 
                    shape = [self._num_units*4], 
                    initializer = self.init )   
        elif self.wform == 'diag_to_full':
            #get the current variable scope
            var_scope = tf.get_variable_scope().name.replace('rnn2/','')

            #first filtering 
            vars_to_use = [var for var in self.init if var_scope in var[0]]  

            #next, assign the variables   
            for var in vars_to_use:
                if '/Ws' in var[0]:
                    Ws = np.split(var[1], indices_or_sections = 4, axis = 1)
                    diag_Ws = np.concatenate([np.diag(w.squeeze()) for w in Ws], axis = 1)

                elif '/Umats' in var[0]:
                    Us = var[1] 
                
                elif '/biases' in var[0]:
                    biases_np = var[1]
                    
            mats_np = np.concatenate([diag_Ws, Us], axis = 0) 
            mats_init = tf.constant_initializer(mats_np) 

            mats = tf.get_variable("mats", 
                    shape = [self._num_units+self.L1,self._num_units*4], 
                    initializer = mats_init )   

            biases_init = tf.constant_initializer(biases_np)
            biases = tf.get_variable("biases", 
                    shape = [self._num_units*4], 
                    initializer = biases_init )   




        return mats, biases


class GatedWFCell(tf.contrib.rnn.RNNCell):    
    """GRU Cell """

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32), wform = 'full' ):
        self._num_units = num_units
        self.init = initializer
        self.wform = wform 

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            #with tf.variable_scope("w"):  # w gate.
            #    w = tf.nn.sigmoid(self.get_nextstate(state, inputs))  
            #with tf.variable_scope("f"):  # forget gate
            #    f = tf.nn.sigmoid(self.get_nextstate(state, inputs))  

            #with tf.variable_scope("Candidate"):
            #    cand = tf.nn.tanh(self.get_nextstate(state*w, inputs))   
            
            init = self.init
            self.L1 = inputs.get_shape().as_list()[1]
           
            mats, biases = self.get_params_parallel()
            
            if self.wform == 'full' or self.wform == 'diag_to_full':

                res = tf.matmul(tf.concat([state,inputs],axis=1),mats)
                res_wbiases = tf.nn.sigmoid(tf.nn.bias_add(res, biases))
           
                i,f = tf.split(res_wbiases,num_or_size_splits=2,axis=1) 

                with tf.variable_scope("candidate"):
                    cmats = tf.get_variable("cmats", 
                            shape = [self._num_units+self.L1,self._num_units],
                            initializer = self.init)
                    cbias = tf.get_variable("cbias",
                            shape = [self._num_units],
                            initializer = self.init)

                    cand = tf.matmul(tf.concat([state*i,inputs],axis=1),cmats)
                    cand = tf.nn.bias_add(cand, cbias)
            elif self.wform == 'diagonal':
                h_concat = tf.concat([state,state],axis=1)

                W_res = tf.multiply(h_concat,mats[0])

                U_res = tf.matmul(inputs,mats[1])

                res = tf.add(W_res,U_res)
                res_wbiases = tf.nn.sigmoid(tf.nn.bias_add(res, biases))

                i,f = tf.split(res_wbiases,num_or_size_splits=2,axis=1) 

                with tf.variable_scope("candidate"):
                    cW = tf.get_variable("cW",
                            shape = [1,self._num_units],
                            initializer = self.init)
                    cU = tf.get_variable("cU", 
                            shape = [self.L1,self._num_units],
                            initializer = self.init)
                    cbias = tf.get_variable("cbias",
                            shape = [self._num_units],
                            initializer = self.init)

                    Wcand = tf.multiply(tf.multiply(cW, state),i)
                    Ucand = tf.matmul(inputs, cU)
                    cand = tf.add( Wcand, Ucand )
                    cand = tf.nn.bias_add( cand, cbias)
            
            new_h = tf.nn.tanh(cand)*(1-f) + f*state 
        return new_h, new_h

    def get_params_parallel(self):
        if self.wform == 'full':
            mats = tf.get_variable("mats", 
                    shape = [self._num_units+self.L1,self._num_units*2], 
                    initializer = self.init )   
            biases = tf.get_variable("biases", 
                    shape = [self._num_units*2], 
                    initializer = self.init )   
        elif self.wform == 'diagonal':
            Ws = tf.get_variable("Ws", 
                    shape = [1,self._num_units*2], 
                    initializer = self.init )   
            Umats = tf.get_variable("Umats", 
                    shape = [self.L1,self._num_units*2], 
                    initializer = self.init )   
            biases = tf.get_variable("biases", 
                    shape = [self._num_units*2], 
                    initializer = self.init )   
            mats = [Ws, Umats] 

        return mats, biases 


class rnn(object):
    """This class has the build_graph function  that builds the rnn computation graph, and the optimizer that optimizes the model parameters given the graph handle and the data""" 
    def __init__(self, model_specs, initializer = 'xavier'):
        'model specs is a dictionary'
        self.model_specs = model_specs
        self.initializer = initializer
    
    def build_graph(self): 
        'this function builds a graph with the specifications in self.model_specs'
        
        d = self.model_specs #unpack the model specifications
        with tf.device('/' + d['device']):
            if d['mapping_mode'] == 'seq2seq':
                x = tf.placeholder(tf.float32, [None, d['batchsize'], d['L1']],"x")
                mask = tf.placeholder(tf.float32, [None])
                if d['outstage'] == 'softmax':
                    y = tf.placeholder(tf.int32, [None]) 
                else:
                    y = tf.placeholder(tf.float32, [None, d['L1']],"y") 

            elif d['mapping_mode'] == 'seq2vec': 
                x = tf.placeholder(tf.float32, [None, None, d['L1']])
                mask = tf.placeholder(tf.float32, [None])
                y = tf.placeholder(tf.int32, [None]) 
            dropout_kps = tf.placeholder(tf.float32, [2], "dropout_params")
            seq_lens = tf.placeholder(tf.int32, [None])
            
            yhat = self.define_model(x, seqlens = seq_lens, dropout_kps = dropout_kps)
                    
            #compute the number of parameters to be trained
            tvars = tf.trainable_variables()
            tnparams = np.sum([np.prod(var.get_shape().as_list()) for var in tvars])
            saver = tf.train.Saver(tvars) 
           

            if d['count_mode']:
                print('The total number of parameters to be trained =', tnparams,
                      'The model is: ', d['model'])
                os._exit(0)

            #raise an error if we are outside the allowed range
            if d['min_params'] > tnparams or d['max_params'] < tnparams:
                raise num_paramsError
            else:
                self.tnparams = tnparams
                self.tvars_names = [var.name for var in tvars] 

            #define the cost         
            if d['outstage'] == 'softmax':
                temp_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = yhat, labels = y )
                masked_cost = temp_cost*mask 
                cost = tf.reduce_mean( masked_cost )  
            elif d['outstage'] == 'sigmoid': 
                temp_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = yhat, 
                        labels = y) 
                masked_cost = tf.reduce_sum(temp_cost,1)*mask 
                cost = tf.reduce_mean( masked_cost )  

            #define the optimizer
            if d['optimizer'] == 'Adam':
                train_step = tf.train.AdamOptimizer(d['LR']).minimize(cost)
            elif d['optimizer'] == 'RMSProp':
                train_step = tf.train.RMSPropOptimizer(d['LR'], 
                                    momentum = d['momentum'],
                                    centered = True).minimize(cost)   
            elif d['optimizer'] == 'Adadelta':
                train_step = tf.train.AdadeltaOptimizer(d['LR']).minimize(cost)   

            #compute the accuracies #somehow check the second line? 
            if d['task'] == 'digits' or d['task'] == 'speech':
                relevant_inds = tf.squeeze(tf.where(tf.cast(mask,tf.bool)))
                preds = tf.gather(tf.nn.softmax(yhat),relevant_inds) 
                targets = tf.gather(y,relevant_inds) 
                correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), targets)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            elif d['task'] == 'music':
                relevant_inds = tf.squeeze(tf.where(tf.cast(mask,tf.bool)))
                preds = tf.gather(yhat,relevant_inds) 
                targets = tf.gather(y,relevant_inds) 
                logl = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                                     logits = preds, labels = targets),1)
                accuracy = tf.reduce_mean(logl)


            #return the graph handles 
            graph_handles = {'train_step':train_step,
                             'x':x,
                             'y':y,
                             'mask':mask,
                             'cost':cost,
                             'dropout_kps':dropout_kps,
                             'seq_lens':seq_lens,
                             'accuracy':accuracy,
                             'saver':saver,
                             'preds':preds,
                             'relevant_inds':relevant_inds,
                             'targets':targets,
                             'logl':logl}
                                           
                             
            return graph_handles


    def define_model(self, x, seqlens ,dropout_kps = tf.constant([1,1])):  
        p1 = dropout_kps[0]
        p2 = dropout_kps[1]
        onedir_models = ['lstm','gru', 'gated_w', 'vanilla_rnn',  
                            'gated_wf', 'mod_lstm','mod_rnn'] #gated_w is to be implemented
        bidir_models = ['bi_lstm', 'bi_mod_lstm', 'bi_gated_w', 'bi_gated_wf' ]
       
        # unpack model specifications 
        d = self.model_specs
        wform, model, K, num_layers, mapping_mode, L1, L2 = d['wform'], d['model'], d['K'], d['num_layers'], d['mapping_mode'], d['L1'], d['L2']

        if model in bidir_models:
            #bidirectional rnns
            if self.initializer == 'xavier':
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, 
                        seed=2, dtype=tf.float32)
            else:
                initializer = self.initializer
            
            if model == 'bi_lstm': 
                fw_cell = tf.contrib.rnn.BasicLSTMCell(K, forget_bias=1.0)
                bw_cell = tf.contrib.rnn.BasicLSTMCell(K, forget_bias=1.0)
            elif model == 'bi_mod_lstm':
                fw_cell = ModLSTMCell(K, initializer = initializer, wform = wform)
                bw_cell = ModLSTMCell(K, initializer = initializer, wform = wform)
            elif model  == 'bi_gated_w':
                fw_cell = GatedWCell(K, initializer = initializer, wform = wform)
                bw_cell = GatedWCell(K, initializer = initializer, wform = wform)
            elif model  == 'bi_gated_wf':
                fw_cell = GatedWFCell(K, initializer = initializer, wform = wform)
                bw_cell = GatedWFCell(K, initializer = initializer, wform = wform)


            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=p1)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=p1)

            fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * num_layers)
            bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * num_layers)

            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=p2)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=p2)

            outputs, _= tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32, time_major = True, sequence_length = seqlens )
            outputs = tf.concat(outputs, axis = 2) #This is concatenating fw - bw rnns

            if mapping_mode == 'seq2vec':
                mean_output = tf.reduce_mean(outputs, axis = 0)  
                outputs = mean_output
            elif mapping_mode == 'seq2seq': #this part requires work 
                outputs = tf.transpose(outputs, [1,0,2] ) 
                outputs = tf.unstack(outputs,axis = 0)
                outputs = tf.concat(outputs, axis = 0)

            with tf.variable_scope("output_stage"):
                if d['wform'] == 'diag_to_full':
                    vars_to_use = [var for var in self.initializer if 'output_stage' in var[0]] 
                    for var in vars_to_use:
                        if '/V' in var[0]:
                            initializer = tf.constant_initializer(var[1])
                            V = tf.get_variable("V", dtype= tf.float32, 
                                shape = [2*K, L2], initializer = initializer)  
                        else:
                            initializer = tf.constant_initializer(var[1])
                            b = tf.get_variable("b", dtype= tf.float32, 
                                shape = [L2], initializer = initializer)  

                else:
                    initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                    V = tf.get_variable("V", dtype= tf.float32, 
                            shape = [2*K, L2], initializer = initializer)  
                    b = tf.get_variable("b", dtype= tf.float32, 
                            shape = [L2 ], initializer = initializer)  

            yhat = tf.matmul(outputs,V) + tf.reshape(b, (1, L2))
            return yhat 


        elif model in onedir_models:
            if self.initializer == 'xavier':
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, 
                        seed=2, dtype=tf.float32)
            elif self.initializer == 'random_uniform':
                fac = 2*np.random.rand()
                initializer = tf.random_uniform_initializer(
                        minval = -fac*np.sqrt(6)/np.sqrt(2*K),
                        maxval = fac*np.sqrt(6)/np.sqrt(2*K),
                        seed = d['seedin'][0])

            else:
                initializer = self.initializer
            
            if model == 'lstm': 
                cell = tf.contrib.rnn.BasicLSTMCell(K, forget_bias=1.0)
            if model == 'gru':
                cell = tf.contrib.rnn.GRUCell(K )  
            elif model == 'mod_lstm':
                cell = ModLSTMCell(K, initializer = initializer, wform = wform)
            elif model  == 'gated_w':
                cell = GatedWCell(K, initializer = initializer, wform = wform)
            elif model  == 'gated_wf':
                cell = GatedWFCell(K, initializer = initializer, wform = wform)
            elif model == 'mod_rnn':
                cell = ModRNNCell(K, initializer = initializer, wform = wform)

            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p1)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=p2)

            outputs, _= tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, 
                    time_major = True, sequence_length = seqlens )

            if mapping_mode == 'seq2vec':
                mean_output = tf.reduce_mean(outputs, axis = 0)  
                outputs = mean_output
            elif mapping_mode == 'seq2seq': #this part requires work 
                outputs = tf.transpose(outputs, [1,0,2] ) 
                outputs = tf.unstack(outputs,axis = 0)
                outputs = tf.concat(outputs, axis = 0)

            with tf.variable_scope("output_stage"):
                if d['wform'] == 'diag_to_full':
                    vars_to_use = [var for var in self.initializer if 'output_stage' in var[0]] 
                    for var in vars_to_use:
                        if '/V' in var[0]:
                            V_initializer = tf.constant_initializer(var[1])
                        else:
                            b_initializer = tf.constant_initializer(var[1])
                            
                else:
                    V_initializer = b_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                
                V = tf.get_variable("V", dtype= tf.float32, 
                        shape = [K, L2], initializer = V_initializer)  
                b = tf.get_variable("b", dtype= tf.float32, 
                        shape = [L2], initializer = b_initializer)  

            yhat = tf.matmul(outputs,V) + tf.reshape(b, (1, L2))
            return yhat 


    def optimizer(self, data, rnn_handles, sess, model_n = 1):
        """This function runs the optimizer for the given data and given rnn graph referenced by rnn_handles """

        d = self.model_specs # unpack the variables 
        iterator_dict = {'BucketedDataIterator':BucketedDataIterator,
                         'SimpleDataIterator':SimpleDataIterator}

        tr = iterator_dict[d['iterator']](data['Train'], 
                num_buckets = d['num_buckets'])
        tst = SimpleDataIterator(data['Test'])
        valid = SimpleDataIterator(data['Validation'])

        all_times, tr_logls, test_logls, valid_logls = [], [], [], [] 
        for ep in range(d['EP']):
            t1, tr_logl = time.time(), []
            while tr.epochs == ep:
                trb = tr.next_batch(
                        n = d['batchsize'], 
                        task = d['task'], 
                        verbose = d['verbose'])      
                
                feed = {rnn_handles['x']:trb[0], 
                        rnn_handles['y']:trb[1], 
                        rnn_handles['mask']:trb[2],
                        rnn_handles['seq_lens']:trb[3], 
                        rnn_handles['dropout_kps']:d['dropout'] }  

                tr_cost, tr_logl_temp, _ = sess.run( 
                        [rnn_handles['cost'], 
                        rnn_handles['accuracy'], 
                        rnn_handles['train_step']], feed) 
                tr_logl.append(tr_logl_temp)
                
                if d['verbose']:
                    print("Training cost = ", tr_cost, 
                          " Training Accuracy = ", tr_logl_temp)
            t2 = time.time()

            #get test and validation accuracies (Implement the for loop over batches) 
            tsb = tst.next_batch( n = d['batchsize'], task = d['task'], 
                    verbose = d['verbose'])  
            vlb = valid.next_batch( n = d['batchsize'], task = d['task'], 
                    verbose=d['verbose'])  

            tst_feed = {rnn_handles['x']: tsb[0], 
                    rnn_handles['y']: tsb[1], 
                    rnn_handles['mask']:tsb[2],
                    rnn_handles['seq_lens']: tsb[3], 
                    rnn_handles['dropout_kps']:np.array([1,1])} 
            
            vld_feed = {rnn_handles['x']: vlb[0], 
                    rnn_handles['y']: vlb[1], 
                    rnn_handles['mask']: vlb[2],
                    rnn_handles['seq_lens']: vlb[3], 
                    rnn_handles['dropout_kps']:np.array([1,1])} 
   
            tr_logl = np.mean(tr_logl)
            tst_logl = sess.run( rnn_handles['accuracy'], tst_feed ) 
            vld_logl = sess.run( rnn_handles['accuracy'], vld_feed ) 
    
            print("Model is ",d['model'],d['wform'],
                  "Optimizer is ",d['optimizer'],
                  " ,Iteration = ", ep, 
                  " ,Training Accuracy", np.mean(tr_logl),
                  ",Test Accuracy = ", tst_logl, 
                  ",Validation Accuracy = ", vld_logl, 
                  ",Elapsed Time = ", t2-t1) 


            all_times.append(t2-t1)
            tr_logls.append(tr_logl)
            test_logls.append(tst_logl)
            valid_logls.append(vld_logl)


        return all_times, tr_logls, test_logls, valid_logls

    def save_modelvars_np(self, sess):
        """ This function saves the variables in diagonal to full transition """ 

        variables = tf.trainable_variables()
        vars_np = [(var.name,sess.run(var)) for var in variables]

        return vars_np 


def return_Klimits(model, wform, data):
    """We use this function to select the upper and lower limits of number of 
    hidden units per layer depending on the task and the dataset. The user can also choo    se to limit the upper and lower limit of allowable number of trainable parameters"""

    if model == 'mod_lstm':
        min_params = 1e1; max_params =  7e7 # in our waspaa paper we basically did not use lower and upper bounds for number of parameters
        K_min, K_max = 50, 300   

    elif model == 'gated_wf': #this is essentially gru
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 50, 350
   
    elif model == 'mod_rnn':
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 50, 400

    return K_min, K_max, min_params, max_params 

def generate_random_hyperparams(lr_min, lr_max, K_min, K_max, num_layers_min, num_layers_max,load_hparams):
    """This function generates random hyper-parameters for hyperparameter search"""

    #this is for new random parameters
    if not load_hparams[0]:
        lr_exp = np.random.uniform(lr_min, lr_max)
        lr = 10**(lr_exp)
        K = np.random.choice(np.arange(K_min, K_max+1),1)[0]
        num_layers = np.random.choice(np.arange(num_layers_min, num_layers_max + 1),1)[0]
        #momentum_exp = np.random.uniform(-8,0) 
        momentum = np.random.uniform(0,1) #(2**momentum_exp)

    #this loads hyperparameters from an existing file
    else:
        exp_data = np.load('experiment_data/nmf_data_timit_model_bi_mod_lstm_diag_to_full_device_cpu:0_1490813245.npy')[load_hparams[1]]
        lr = exp_data['LR']
        K = exp_data['K']
        num_layers = exp_data['num_layers']
        try:
            momentum = exp_data['num_layers']
        except:
            momentum = None

    return lr, K, num_layers, momentum

def load_data(task, data):
    """this function loads the data, and sets the associated parameters (such as output and input dimensionality and batchsize) according to the specified task, which are either text, music, speech or digits """

    if task == 'text':
        print('Task is text')

        if data == 'ptb':
            #deprecated
            tr = 'simple-examples/data/ptb.train.txt'
            vld = 'simple-examples/data/ptb.valid.txt'
            test = 'simple-examples/data/ptb.test.txt'

            [X, fmap ] = load_multiple_textdata( [tr,vld,test] )  
            Trainseq = X[0]
            
            #chop off the tail of validation and test sequences, I am hard coding 200 as num_steps  
            Validseq = X[1]
            vslen = round(Validseq.shape[1])
            vnum_steps = 2000
            Validseq = Validseq[:, 0:(vslen - (vslen%vnum_steps) + 1)]

            Testseq = X[2]
            tslen = round(Testseq.shape[1])
            tnum_steps = 2000 
            Testseq = Testseq[:, 0:(tslen - (tslen%tnum_steps) + 1)] 

            mbatchsize = 200000 
        else:
            Tseq = 10000
            [X, fmap] = load_textdata(data+'.txt')

            offset = 5000
            Trainseq = X[:, offset:offset+Tseq] # To remove leading gaps in text
            Testseq = X[:, offset + Tseq + 50: offset + 3*Tseq+50]  
            Validseq = Testseq 

            mbatchsize = Tseq - 1 #this is the batch size. 

    elif task == 'music':
        print('Loading Music task ' + data + ' data')
        filename = data + '.pickle'

        #this if-else determines the maximum allowable sequence length depending on the dataset
        if data == 'JSB Chorales':
            len_th = 99999 #basically no sequence chopping
        elif data == 'Piano-midi.de':
            len_th = 200
        elif data == 'Nottingham':
            len_th = 200
        elif data == 'MuseData':
            len_th = 200
        else:
            len_th = 9e9

        dataset = load_musicdata( filename, len_th ) 
         
        d = {'data':dataset[0][0], 'lengths':dataset[0][1]}         
        df_train = pd.DataFrame( d ) 

        d = {'data':dataset[1][0], 'lengths':dataset[1][1]}
        df_test = pd.DataFrame( d )

        d = {'data':dataset[2][0], 'lengths':dataset[2][1]}
        df_valid = pd.DataFrame( d ) 

        if data == 'JSB Chorales':
            iterator = 'SimpleDataIterator'
            batchsize = len(df_train) 
            num_buckets = None
        if data == 'Piano-midi.de':
            iterator = 'SimpleDataIterator'
            batchsize = round( 0.5*len(df_train) ) 
            num_buckets = None 
        elif data == 'Nottingham':
            iterator = 'SimpleDataIterator'
            batchsize = round( 0.5*len(df_train) ) 
            num_buckets = None
        elif data == 'MuseData':
            iterator = 'SimpleDataIterator'
            batchsize = round( 0.5*len(df_train) )
            num_buckets = None

        L1 = L2 = df_train['data'][0].shape[0]
        outstage = 'sigmoid'
        mapping_mode = 'seq2seq'
        num_steps = None

    elif task == 'digits':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #Train
        Trainsize = mnist.train.images.shape[0]
        images = mnist.train.images[:Trainsize, :]
        Trainims = np.split(images.reshape(Trainsize*28,28),Trainsize,0)

        Trainlabels = list(np.argmax(mnist.train.labels[:Trainsize,:],1))
        lengths = [28]*Trainsize

        d = {'data':Trainims, 'labels':Trainlabels, 'lengths':lengths }
        df_train = pd.DataFrame( d )    

        #Test
        Testsize = mnist.test.images.shape[0]
        images = mnist.test.images[:Testsize, :] 
        Testims = np.split(images.reshape(Testsize*28,28),Testsize,0)
    
        Testlabels = list(np.argmax(mnist.test.labels[:Testsize,:],1))
        lengths = [28]*Testsize

        d = {'data':Testims, 'labels':Testlabels, 'lengths':lengths }
        df_test = pd.DataFrame( d )    


        #Validation
        Validsize = mnist.validation.images.shape[0]
        images = mnist.validation.images[:Validsize,:] 
        Validims = np.split(images.reshape(Validsize*28,28),Validsize,0 )

        Validlabels = list(np.argmax(mnist.validation.labels[:Validsize,:],1))
        lengths = [28]*Validsize

        d = {'data':Validims, 'labels':Validlabels, 'lengths':lengths }
        df_valid = pd.DataFrame( d )

        batchsize = 5000
        L1 = Trainims[0].shape[0]
        L2 = np.max(Trainlabels) + 1
        outstage = 'softmax'
        mapping_mode = 'seq2vec'
        num_steps = 28
        iterator = 'SimpleDataIterator'
        num_buckets = None
        len_th = None

    elif task == 'speech':
        filehandle = open('timit39.pickle','rb')
        dataset = pickle.load(filehandle)

        d = {'data':dataset[0][0],'labels':dataset[0][1],'lengths':dataset[0][2] }
        df_train = pd.DataFrame( d ) 

        d = {'data':dataset[1][0],'labels':dataset[1][1],'lengths':dataset[1][2] }
        df_test = pd.DataFrame( d )

        d = {'data':dataset[2][0],'labels':dataset[2][1],'lengths':dataset[2][2] }
        df_valid = pd.DataFrame( d )

        batchsize = 200#round(len(df_train)/5)
        L1 = dataset[0][0][0].shape[0]
        L2 = 39
        outstage = 'softmax'
        mapping_mode = 'seq2seq'
        num_steps = None
        iterator = 'BucketedDataIterator'
        num_buckets = 5
        len_th = None
        
    parameters = {'batchsize':batchsize,
                  'L1':L1,
                  'L2':L2,
                  'outstage':outstage,
                  'mapping_mode':mapping_mode,
                  'num_steps':num_steps,
                  'iterator':iterator,
                  'num_buckets':num_buckets,
                  'len_th':len_th}

    return {'Train':df_train, 'Test':df_test, 'Validation':df_valid}, parameters


class SimpleDataIterator():
    """
    This class is adapted (ripped off) from r2rt.com, in this version, the next_batch function uses a pandas dataframe, and outputs the batch in a reshaped format ready to input to the tensorflow function 
    """
    def __init__(self, df, num_buckets = None):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n, task, verbose = False):
        if verbose:
            print("The current cursor points to ",self.cursor," Data size is",self.size)

        part = self.df.ix[self.cursor:self.cursor+n-1]
        
        if task == 'digits': 
            #this part needs to be updated
            temp = list(part['data'].values)
            data = np.transpose(np.asarray(temp),[2,0,1])  
            labels = part['labels'].values
            mask = np.ones(labels.shape) 
            lengths = part['lengths'].values
        elif task == 'speech':
            max_len = np.max(part['lengths'].values)
            L1 = part['data'].values[0].shape[0]
           
            lengths = np.zeros(n) 
            mask = np.zeros((n,max_len))
            labels = np.zeros((n,max_len))
            data = np.zeros((L1, max_len, n)) 
            iterables = zip(part['lengths'].values, part['labels'].values,part['data'].values)
            for i,vals in enumerate(iterables):
                lengths[i] = vals[0]
                mask[i,0:vals[0]] = 1
                labels[i,0:vals[0]] = vals[1] 
                data[:,0:vals[0],i] = vals[2]

            #finally reshape things
            labels = labels.reshape(-1)
            mask = mask.reshape(-1)
            data = np.transpose(data, [1,2,0]) 
        elif task == 'music':
            max_len = np.max(part['lengths'].values)
            L = part['data'].values[0].shape[0] 
           
            lengths = np.zeros(n) 
            mask = np.zeros((n,max_len-1))
            labels = [np.zeros((max_len-1, L)) for i in range(n)] # better name is 'targets' 
            data = np.zeros((L, max_len -1, n)) 
            iterables = zip(part['lengths'].values, part['data'].values)
            for i,vals in enumerate(iterables):
                ln = vals[0] - 1
                lengths[i] = ln
                mask[i,0:ln] = 1
                labels[i][0:ln,:] = vals[1][:,1:].transpose()
                data[:,0:ln,i] = vals[1][:,:-1]

            #finally reshape things
            mask = mask.reshape(-1)
            labels = np.concatenate(labels, axis = 0)
            data = np.transpose(data, [1,2,0]) 

        if self.cursor+n >= self.size:
            self.epochs += 1
            self.shuffle()
        else:
            self.cursor += n

        return data, labels, mask, lengths


class BucketedDataIterator():
    """This one is the bucketed version of the simple data iterator, that is the sequences are ordered and put into buckets to minimize the total padding"""
    def __init__(self, df, num_buckets = 2):
        df = df.sort_values('lengths').reset_index(drop=True)
        self.size = len(df) / num_buckets
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.ix[bucket*self.size: (bucket+1)*self.size - 1])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
            #if i != 0: 
            #    self.cursor[i] = 800

    def next_batch(self, n, task, verbose = False):
        if verbose:
            print("The current cursor points to ",self.cursor," Data size is",self.size)
       
        relevant_j = np.where(self.cursor <= self.size)[0]
        j =  np.random.choice(relevant_j)

        ## length adaptation
        part = self.dfs[j].ix[self.cursor[j]:self.cursor[j]+n-1]
        #if np.max( temp_part['lengths'] ) > 1000:
        #    n = 8
        #    part = self.dfs[j].ix[self.cursor[j]:self.cursor[j]+n-1]
        #else:
        #    part = temp_part

        
        self.cursor[j] += n #increase the cursor once we choose our part
       
        if task == 'digits': 
            #this part needs to be updated
            temp = list(part['data'].values)
            data = np.transpose(np.asarray(temp),[2,0,1])  
            labels = part['labels'].values
            mask = np.ones(labels.shape)
            lengths = part['lengths'].values
        elif task == 'speech':
            max_len = np.max(part['lengths'].values)
            L1 = part['data'].values[0].shape[0]
           
            lengths = np.zeros(n) 
            mask = np.zeros((n,max_len))
            labels = np.zeros((n,max_len))
            data = np.zeros((L1, max_len, n)) 
            iterables = zip(part['lengths'].values, part['labels'].values,part['data'].values)
            for i,vals in enumerate(iterables):
                lengths[i] = vals[0]
                mask[i,0:vals[0]] = 1
                labels[i,0:vals[0]] = vals[1] 
                data[:,0:vals[0],i] = vals[2]

            #finally reshape things
            mask = mask.reshape(-1)
            labels = labels.reshape(-1)
            data = np.transpose(data, [1,2,0]) 
        elif task == 'music':
            max_len = np.max(part['lengths'].values)
            L = part['data'].values[0].shape[0] 
           
            lengths = np.zeros(n) 
            mask = np.zeros((n,max_len-1))
            labels = [np.zeros((max_len-1, L)) for i in range(n)] # better name is 'targets' 
            data = np.zeros((L, max_len -1, n)) 
            iterables = zip(part['lengths'].values, part['data'].values)
            for i,vals in enumerate(iterables):
                ln = vals[0] - 1
                lengths[i] = ln
                mask[i,0:ln] = 1
                labels[i][0:ln,:] = vals[1][:,1:].transpose()
                data[:,0:ln,i] = vals[1][:,:-1]

            #finally reshape things
            mask = mask.reshape(-1)
            labels = np.concatenate(labels, axis = 0)
            data = np.transpose(data, [1,2,0]) 

            #sanity_check = (1-mask.reshape(len(mask),1))*labels #sum should be 0 

        
        #np.any(self.cursor+n+1 > self.size):
        if np.sum(self.cursor > self.size) == self.num_buckets  :
            self.epochs += 1
            self.shuffle()

        return data, labels, mask, lengths 

def load_multiple_textdata(filenames):
    """This file reads multiple text files given in filenames, and it returns a list containing the one hot coded representation of these files"""
    nfiles = len(filenames)
    f = [open(filenames[i],'r') for i in range(nfiles)] 

    texts =  [' '.join( f[i].read().split('\n') ) for i in range(nfiles)] 
    textlens = [len(texts[i]) for i in range(nfiles)] 
    textlens.insert(0, 0) 
    textstarts = np.cumsum( textlens ) #starting points of the text files within the concatenated text 

    ftext = ''.join(texts) #the concatenated text  

    v = ft.CountVectorizer(analyzer = 'char')
    Y = v.fit_transform(list(ftext)).toarray().transpose().astype('float32')
    fmapping = v.get_feature_names()

    Ys = [Y[:,textstarts[i]:textstarts[i+1]] for i in range(nfiles)] 

    return Ys,fmapping


def load_musicdata(fl, len_th):
    """this function is used to load the symbolic music files in piano roll format"""

    filename = open(fl,'rb')
    dataset = pickle.load(filename)
    
    dataset_list = [dataset['train'],dataset['test'],dataset['valid']]
    
    #this part extracts the max and min from the data
    lst = []
    lens = []
    for i, dataset in enumerate(dataset_list):
        for sequence in dataset:
            T = len(sequence) #length of the sequence  
            lens.append(T)
            lst.extend( list(itertools.chain.from_iterable( sequence )) )                     
    max_val, min_val = max(lst), min(lst) 

    #get the statistics regarding lengths
    sorted_lens = np.sort(lens)
    lendist = np.cumsum(sorted_lens)/np.sum(sorted_lens) 


    #this part puts the data in binary matrix format 
    sets = []
    for i, dataset in enumerate(dataset_list):
        lens, mats = [], []
        for sequence in dataset:
            T = len(sequence) 
            mat = np.zeros((max_val + 1, T)) # +1 just to be safe
            for t,vals in enumerate(sequence):
                mat[vals, t] = 1    
            mat = mat[min_val-1:max_val,:] #eliminate the empty parts of the data

            #split the sequence if it is too long
            if T > len_th:
                split_indices = np.arange(len_th, T, len_th) 
                divided_mat = np.split( mat, indices_or_sections = split_indices, axis = 1 )
                mats.extend(divided_mat)
                lens.extend([submat.shape[1] for submat in divided_mat])
            else:
                mats.append(mat)
                lens.append(T)
        
        sets.append([mats,lens])
            
    return sets


class Error(Exception):
    pass

class num_paramsError(Error):
    pass


