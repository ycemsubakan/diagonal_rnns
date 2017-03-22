import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from IPython.display import clear_output
from tensorflow.python.client import timeline
import pdb
import time
import pandas as pd

def fliplr(x):
    height, width = x.get_shape().as_list() # x is a tensor
    xr = tf.reshape(x, [height, width, 1])
    y = tf.squeeze(tf.image.flip_left_right(xr))
    return y

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
            with tf.variable_scope("i"):  
                i = self.get_nextstate(h,inputs) 
            with tf.variable_scope("j"):  
                j = self.get_nextstate(h,inputs)
            with tf.variable_scope("f"):  
                f = self.get_nextstate(h,inputs)
            with tf.variable_scope("o"):  
                o = self.get_nextstate(h,inputs)
                  
            new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i)*tf.nn.tanh(j))

            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state

    def get_nextstate(self, state, x):

        if self.wform == 'full':
            W = tf.get_variable("W", shape = [self._num_units, self._num_units], initializer = self.init )   
            Wh = tf.matmul(state,W)   
        elif self.wform == 'diagonal':
            W = tf.get_variable("W", shape = [self._num_units], initializer = self.init )   
            Wh = W*state
        elif self.wform == 'scalar':
            W = tf.get_variable("W", shape = [1], initializer = self.init ) 
            Wh = W*state
        elif self.wform == 'constant':    
            Wh = state
        elif self.wform == 'diagonal_to_full':
            tf.get_variable_scope().reuse_variable()
            Wdiag = tf.get_variable("W", shape = [self._num_units]) 
            W = tf.Variable( tf.diag(Wdiag), name = "Wfull") 

        U_shape = [x.get_shape().as_list()[1], self._num_units] 
        U = tf.get_variable("U", shape = U_shape, initializer = self.init) 
        b = tf.get_variable("b", shape = [self._num_units], initializer = self.init)
        
        Ux = tf.matmul( x, U)  
        next_state = Wh + Ux + b

        return next_state 



class GatedWFCell(tf.contrib.rnn.RNNCell):
    """Gated W cell """

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
            with tf.variable_scope("w"):  # w gate.
                w = tf.nn.sigmoid(self.get_nextstate(state, inputs))  
            with tf.variable_scope("f"):  # forget gate
                f = tf.nn.sigmoid(self.get_nextstate(state, inputs))  

            with tf.variable_scope("Candidate"):
                cand = tf.nn.tanh(self.get_nextstate(state*w, inputs))   

            new_h = cand*f + (1-f)*state 
        return new_h, new_h

    def get_nextstate(self, state, x):
        if self.wform == 'full':
            W = tf.get_variable("W", shape = [self._num_units, self._num_units], initializer = self.init )   
            Wh = tf.matmul(state,W)   
        elif self.wform == 'diagonal':
            W = tf.get_variable("W", shape = [self._num_units], initializer = self.init )   
            Wh = W*state
        elif self.wform == 'scalar':
            W = tf.get_variable("W", shape = [1], initializer = self.init ) 
            Wh = W*state
        elif self.wform == 'constant':    
            Wh = state

        U_shape = [x.get_shape().as_list()[1], self._num_units] 
        U = tf.get_variable("U", shape = U_shape, initializer = self.init) 
        b = tf.get_variable("b", shape = [self._num_units], initializer = self.init)
        
        Ux = tf.matmul( x, U)  
        next_state = Wh + Ux + b

        return next_state 



class GatedWCell(tf.contrib.rnn.RNNCell):
    """Gated W cell """

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32), wform = 'diagonal'):
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
            with tf.variable_scope("Gate"):  # Reset gate and update gate.
                #pdb.set_trace()
                U_shape = [inputs.get_shape().as_list()[1], self._num_units] 

                init = self.init
                Uw = tf.get_variable("Uw", shape = U_shape, initializer = init )
                bw = tf.get_variable("bw", shape = [self._num_units], initializer = init)
                if self.wform == 'constant': 
                    w = tf.nn.sigmoid( state + tf.matmul( inputs, Uw) + bw) 
                elif self.wform == 'scalar': 
                    Ww = tf.get_variable("Ww", shape = [1], initializer = init) 
                    w = tf.nn.sigmoid( Ww*state + tf.matmul( inputs, Uw) + bw) 
                elif self.wform == 'diagonal':
                    Ww = tf.get_variable("Ww", shape = [self._num_units], initializer=init) 
                    w = tf.nn.sigmoid( Ww*state + tf.matmul( inputs, Uw) + bw) 
                elif self.wform == 'full':
                    Ww = tf.get_variable("Ww", shape = [self._num_units,self._num_units], initializer = init)
                    w = tf.nn.sigmoid( tf.matmul(state,Ww) + tf.matmul(inputs, Uw) + bw) 

            with tf.variable_scope("Candidate"):
                U = tf.get_variable("U", shape = U_shape, initializer = init)
                b1 = tf.get_variable("b1", shape = [self._num_units], initializer = init ) 
                if self.wform == 'constant':
                    new_h =  tf.nn.tanh( state + w*tf.matmul( inputs, U  ) + b1 )  
                elif self.wform == 'scalar':
                    W = tf.get_variable("W", shape = [1], initializer = init)
                    new_h = tf.nn.tanh( W*state + w*tf.matmul( inputs, U  ) + b1 )  
                elif self.wform == 'diagonal':
                    W = tf.get_variable("W", shape = [self._num_units], initializer = init)
                    new_h = tf.nn.tanh( W*state + w*tf.matmul( inputs, U  ) + b1 )  
                elif self.wform == 'full':
                    W = tf.get_variable("W", shape = [self._num_units,self._num_units], initializer = init)
                    new_h = tf.nn.tanh( tf.matmul(state,W) + w*tf.matmul( inputs, U  ) + b1 )  
 
                    
                
        return new_h, new_h

# Start class definition
class rnn(object):
    def __init__(self, model_specs):
        'model specs is a dictionary'
        self.model_specs = model_specs
    
    def build_graph(self ): 
        'this function builds a graph with the specifications in self.model_specs'

        d = self.model_specs #unpack the model specifications
        with tf.device('/gpu:' + str(d['gpus'][0])):
            if d['mapping_mode'] == 'seq2seq':
                x = tf.placeholder(tf.float32, [None, d['batchsize'], d['L1']])
                y = tf.placeholder(tf.int32, [None,d['batchsize']]) 
            elif d['mapping_mode'] == 'seq2vec': 
                x = tf.placeholder(tf.float32, [None, None, d['L1']])
                y = tf.placeholder(tf.int32, [None]) 
            dropout_kps = tf.placeholder(tf.float32, [2], "dropout_params")
            seq_lens = tf.placeholder(tf.int32, [None])
            
            yhat = self.define_model(x, seqlens = seq_lens, dropout_kps = dropout_kps)
                    
            #compute the number of parameters to be trained
            tvars = tf.trainable_variables()
            tnparams = np.sum([np.prod(var.get_shape().as_list()) for var in tvars])
           
            #raise an error if we are outside the allowed range
            if d['min_params'] > tnparams or d['max_params'] < tnparams:
                raise num_paramsError
            else:
                self.tnparams = tnparams
                self.tvars_names = [var.name for var in tvars] 
                pdb.set_trace()

            #define the cost         
            if d['outstage'] == 'softmax':
                temp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = yhat, labels = y )
                cost = tf.reduce_mean( temp )  
            elif d['outstage'] == 'sigmoid': 
                cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = yhat, labels = y) 
            
            #define the optimizer
            train_step = tf.train.AdamOptimizer(d['LR']).minimize(cost)
            #compute the accuracies
            preds = tf.nn.softmax(yhat) 
            correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        #return the graph handles 
        graph_handles = {'train_step':train_step,
                         'x':x,
                         'y':y,
                         'cost':cost,
                         'dropout_kps':dropout_kps,
                         'seq_lens':seq_lens,
                         'accuracy':accuracy}
                         

        return graph_handles


    def define_model(self, x, seqlens ,dropout_kps = tf.constant([1,1])):  
        p1 = dropout_kps[0]
        p2 = dropout_kps[1]
        onedir_models = ['lstm', 'gated_w', 'vanilla_rnn', 'gated_wf', 'mod_lstm']
        bidir_models = ['bi_lstm', 'bi_mod_lstm', 'bi_gated_w', 'bi_gated_wf' ]
       
        # unpack model specifications 
        d = self.model_specs
        wform, model, K, num_layers, mapping_mode, L1, L2 = d['wform'], d['model'], d['K'], d['num_layers'], d['mapping_mode'], d['L1'], d['L2']

        if model in bidir_models:
            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
            
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
            outputs = tf.concat(outputs, axis = 2) 

            if mapping_mode == 'seq2vec':
                mean_output = tf.reduce_mean(outputs, axis = 0)  
                outputs = mean_output
            elif mapping_mode == 'seq2seq': #this part requires work 
                outputs = tf.transpose(outputs, [2,0,1] ) 
                outputs = tf.unstack(outputs,axis = 2)
                outputs = tf.concat(outputs, axis = 1)

            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
            with tf.variable_scope("output_stage"):
                if self.model_specs['wform'] == 'diagonal_to_full':
                    tf.get_variable_scopei().reuse_variables()

                V = tf.get_variable("V", dtype= tf.float32, shape = [2*K, L2], initializer = initializer)  
                b = tf.get_variable("b", dtype= tf.float32, shape = [L2 ], initializer = initializer)  

            yhat = tf.matmul(outputs,V) + tf.reshape(b, (1, L2))
            return yhat 


        elif d in onedir_models:
            with tf.variable_scope("onedir_rnn") as vs: 
                # pdb.set_trace()
                if finalvalid_mode:
                    num_steps = custom_numsteps 
                else:
                    num_steps = self.num_steps 
                num_layers = self.num_layers
               
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                wform = self.wform
                #define the cell
                
                if case == 'lstm':
                    cell = tf.nn.rnn_cell.LSTMCell(self.K, state_is_tuple=True, initializer =  initializer)
                elif case == 'gated_w':
                    cell = GatedWCell(self.K, initializer =  initializer, wform = wform)
                elif case == 'vanilla_rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(self.K) 
                elif case == 'gated_wf':
                    cell = GatedWFCell(self.K, initializer =  initializer, wform = wform) 
                elif case == 'mod_lstm':
                    cell = ModLSTMCell(self.K, initializer = initializer, wform = wform)

                
                # Input Dropout
                if build_with_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=p1)

                
                #make the cell multilayer     
                if not(case in ['lstm', 'mod_lstm'] ): 
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
                else:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
                   
                # Output Dropout
                if build_with_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=p2)

                if train_mode | finalvalid_mode :
                    self.batchsize = int(x.get_shape().as_list()[1]/num_steps)

                    x = tf.split(1, self.batchsize, x)
                    x = tf.pack(x)
                    x = tf.transpose(x, [2, 0, 1] ) 

                    init_state = cell.zero_state(self.batchsize, tf.float32)

                else:
                    self.batchsize = 1
                    x = tf.reshape(x, [1,1,self.L])  
                    init_state = self.state

                if not(train_mode): #if we are not in the train mode we always reuse the variables
                    tf.get_variable_scope().reuse_variables()
               
                                    
                #h_hat, _ = tf.scan( lambda a, x: cell(x,a[1]), elems = x, initializer = (tf.zeros([self.batchsize, self.K]), init_state), parallel_iterations=10)  
                h_hat, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major = True) 
                h_hat = tf.unpack( h_hat, axis = 1) 
                h_hat = tf.concat( concat_dim = 0, values =  h_hat) 

                if self.outstage == 'softmax':
                    y_hat = tf.nn.softmax( tf.matmul(self.V,tf.transpose(h_hat)) + self.b2, dim=0)
                elif self.outstage == 'sigmoid':
                    y_hat = tf.sigmoid( tf.matmul(self.V,tf.transpose(h_hat)) + self.b2)
                elif self.outstage == 'linear': 
                    y_hat = tf.matmul(self.V,tf.transpose(h_hat)) + self.b2

                self.model_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)
                if train_mode:
                    return y_hat
                else:
                    return y_hat, final_state 

        elif model == 'multi_layer_ff':
            
            if not(train_mode):
                tf.get_variable_scope().reuse_variables()

            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
            with tf.variable_scope("first_layer"):
                V1 = tf.get_variable("V1", dtype = tf.float32, shape = [self.K, self.L], initializer = initializer ) 
                b1 = tf.get_variable("b1", dtype = tf.float32, shape = [self.K,1], initializer = initializer)
            with tf.variable_scope("second_layer"):
                V2 = tf.get_variable("V2", dtype = tf.float32, shape = [self.K, self.K], initializer = initializer ) 
                b2 = tf.get_variable("b2", dtype = tf.float32, shape = [self.K,1], initializer = initializer)

            with tf.variable_scope("output_layer"):
                V = tf.get_variable("V", dtype= tf.float32, shape = [self.L2, self.K ], initializer = initializer)  
                b = tf.get_variable("b", dtype= tf.float32, shape = [self.L2, 1 ], initializer = initializer)  

            h1 = tf.tanh(tf.matmul(V1,x) + b1)
            h1 = tf.nn.dropout(h1, 0.75, noise_shape=None, seed=None, name=None)

            h2 = tf.tanh(tf.matmul(V2,h1) + b2) 
            h2 = tf.nn.dropout(h2, 0.75, noise_shape=None, seed=None, name=None)

            yhat = tf.matmul(V, h2) + b#tf.reshape(b, (self.L2,1))
            return yhat



        elif model == 'vector_w_conv':
            #heavily deprecated
            K = self.K
            T = self.T
            ntaps = self.ntaps

            Ux = (tf.matmul(self.U,x))
            
            #flip Ux
            Ux_flip = fliplr(Ux)
            Ux_flip_pad = tf.concat(1, [Ux_flip, tf.zeros( [K, ntaps -1 ] )] )  

            Uxt = tf.transpose(Ux_flip_pad) 
            Uxr = tf.reshape(Uxt,[1, 1, T + ntaps - 1, K])
            
            wt = tf.transpose(self.w)
            wr = tf.reshape(wt,[1, ntaps, K, 1])
            Z = tf.nn.depthwise_conv2d(Uxr, wr, strides=[1,1,1,1], padding = 'VALID')

            Z = fliplr(tf.transpose(tf.squeeze(Z))) + self.b1 
            Hhat = tf.tanh( Z ) 
            
            Yhat = tf.nn.softmax( tf.matmul( self.V, Hhat ) + self.b2  ,dim=0 )
            
            # Transpose and flip
            #y_hat = fliplr( tf.transpose(tf.reshape( y, [Uxth,Uxtw])))
            return Yhat
            
    
    def multi_gpu_grad(self, g1 = tf.tanh, g2 = tf.sigmoid, opt = 'Adam', learning_rate = 0.01, case = 'scalar_w', use_gpu = True, count_mode = False):
        # Placeholders for input and output
#         self.x = tf.placeholder(tf.float64, [L,None], name = 'in_placeholder')
#         self.y = tf.placeholder(tf.float64, [L,None], name = 'out_placeholder')
        self.g1 = g1
        self.g2 = g2
        
        # Define optimizer
        if opt == 'RMSProp':
            print('Using RMSProp Optimizer')
            optimizer = tf.train.RMSPropOptimizer( learning_rate, .5, .9)
        else:
            print('Using Adam Optimizer')
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
        # Split the input data to one batch per GPU
        self.ins = tf.split( self.x, len(self.gpus), axis = 1) # self.ins = tf.split( self.x, len(gpus), 1)
        self.out = tf.split( self.y, len(self.gpus), axis = 1) # self.out = tf.split( self.y, len(gpus), 1)

        # Cost and gradient from each tower
        device = '/gpu:' if use_gpu else '/cpu:' 
        grads = []
        cost = 0
        for i in range(len(self.gpus)):
            with tf.device(device + str( self.gpus[i])):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                c = self.cost_function( self.out[i], self.model( self.ins[i], case))
                if not(count_mode):
                    grads.append( optimizer.compute_gradients( c))
                cost += c
                
        # Average the gradients for all towers
        if not(count_mode):
            avgrad = []
            for gv in zip( *grads):
                gg = []
                for g,_ in gv:
                    gg.append( tf.expand_dims( g, 0))
                gr = tf.reduce_sum( tf.concat( values = gg, axis = 0), 0)
                avgrad.append( (gr, gv[0][1]))

            # Return the weight update and cost ops
            self.updates = [optimizer.apply_gradients( avgrad),cost] 
        

    def sgd_train(self, X, Y, sess, ep = 200, batchsize = 50, verbose = True, plot = False, Validseq = None, Testseq = None, task = 'text', Tgen = 1000):
        batchsize = int(batchsize)
        batches = np.arange(0, X.shape[1], round(batchsize )) #this defines the batch start points 
        batch_ends = (batches + batchsize) <= (X.shape[1]) 
        nbatches = np.sum(batch_ends)
        if task == 'music':
            validlens = [Validseq[i].shape[1] for i in range(len(Validseq))]
            testlens = [Testseq[i].shape[1] for i in range(len(Testseq))]
            totallen_valid = np.sum(validlens) 
            totallen_test = np.sum(testlens) 
            period = 100 #period in which we get validation values
        elif task == 'text':
            period = 2 
        else:
            period = 10

        eps = 1e-9
        cst = []
        test_logls = [] 
        valid_logls = []
        all_times = []
        for jj in range(ep):
            if jj < 41:
                pen = 10000
            else:
                pen = 0 

            c_sum = 0
            start_time = time.time()
            for ii in range(nbatches):
                Xb = X[:,batches[ii]:batches[ii]+batchsize]
                if task == 'digits':
                    Yb = Y[:,int((batches[ii]/28)):int((batches[ii]+batchsize)/28)]
                else:
                    Yb = Y[:,batches[ii]:batches[ii]+batchsize]

                d = {self.x: Xb, self.y: Yb, self.lam: pen}
                c = sess.run( self.updates, feed_dict=d)
                c_sum += c[1]
            cst.append(c_sum)
            time_elapsed = time.time() - start_time
            all_times.append(time_elapsed) 

            if jj%1==0 or jj==(ep-1):
                if (verbose):
                    if self.vary_penalty: #if we have the off-diagonal penalty
                        penalty = sess.run(self.penalty, feed_dict = {self.lam: pen })
                        print('Penalty = ' + str(penalty) )
                        c_sum = c_sum - nbatches*penalty 

                    print( 'Iteration: ' + str(jj) + ' Cost = ' + str( c_sum/nbatches )+ ' Elapsed Time = ' +str(time_elapsed) + '_gpu_' + str(self.gpus[0]))
                    
                    if (jj%period == 0) | (jj==(ep-1)):
                       
                        if task == 'music':
                            # t1 = time.time()
                            # test_logl = 0
                            # for nn in range(len(Testseq)):
                            #     xtest = tf.placeholder(tf.float32, [self.L, Testseq[nn].shape[1]-1], name = 'xtest')
                            #     with tf.device('/gpu:' + str(self.gpus[0])):
                            #         ytest,_ = self.model( xtest, case = self.case, train_mode = False, finalvalid_mode = True, custom_numsteps = Testseq[nn].shape[1] - 1)

                            #     y_probs = sess.run(ytest, feed_dict = {xtest:Testseq[nn][:,0:-1] })
                            #     temp_logl = np.sum(Testseq[nn][:,1:]*np.log(y_probs+eps) + (1-Testseq[nn][:,1:])*np.log(1-y_probs+eps)) 
                            #     test_logl = test_logl + temp_logl
                            # test_logl = test_logl/totallen_test 
                            # test_logls.append(test_logl)  
                            # t2 = time.time()
                            # print( 'Test likelihood = ' + str(test_logl) + ' Time Elapsed = ' + str(t2-t1) + ' secs' ) 
                           
                            ########test###################
                            t1 = time.time()
                            tnumsteps = 1000 
                            Testseqconcat = np.concatenate( Testseq,axis = 1)
                            tslen = Testseqconcat.shape[1] - Testseqconcat.shape[1]%tnumsteps
                            xtest = tf.placeholder(tf.float32, [self.L, tslen], name = 'xtest')
                            Testseqconcat = Testseqconcat[:,0:tslen + 1]
                            with tf.device('/gpu:' + str(self.gpus[0])):
                                ytest,_ = self.model( xtest, case = self.case, train_mode = False, finalvalid_mode = True, custom_numsteps = tnumsteps)

                            y_probs = sess.run(ytest, feed_dict = {xtest:Testseqconcat[:,0:tslen] })
                            test_logl = np.sum(Testseqconcat[:,1:]*np.log(y_probs+eps) + (1-Testseqconcat[:,1:])*np.log(1-y_probs+eps))/tslen
                            test_logls.append(test_logl)    
                            t2 = time.time()
                            print( 'Test likelihood parallel method = ' + str(test_logl) + ' Time Elapsed = ' + str(t2 - t1) + ' secs' ) 

                            ########validation##############
                            t1 = time.time()
                            vnumsteps = 1000 
                            Validseqconcat = np.concatenate( Validseq,axis = 1)
                            vlen = Validseqconcat.shape[1] - Validseqconcat.shape[1]%vnumsteps
                            xvalid = tf.placeholder(tf.float32, [self.L, vlen], name = 'xtest')
                            Validseqconcat = Validseqconcat[:,0:vlen + 1]
                            with tf.device('/gpu:' + str(self.gpus[0])):
                                yvalid,_ = self.model( xvalid, case = self.case, train_mode = False, finalvalid_mode = True, custom_numsteps = vnumsteps)

                            y_probs = sess.run(yvalid, feed_dict = {xvalid:Validseqconcat[:,0:vlen] })
                            valid_logl = np.sum(Validseqconcat[:,1:]*np.log(y_probs+eps) + (1-Validseqconcat[:,1:])*np.log(1-y_probs+eps))/vlen
                            valid_logls.append(valid_logl)    
                            t2 = time.time()
                            print( 'Validation likelihood parallel method = ' + str(valid_logl) + ' Time Elapsed = ' + str(t2 - t1) + ' secs' ) 


                        elif task == 'text': 
                            reportmode = 2 

                            if reportmode == 1:
                                _, test_logl = self.tfgenerate_text(sess, Tseq = Tgen, x = Testseq[:, 0:Tgen+1])
                                test_logls.append(test_logl)  
                                print(  'Test log2BPC = ' + str(test_logl) ) 


                                _, valid_logl = self.tfgenerate_text(sess, Tseq = Tgen, x = Validseq[:, 0:Tgen+1])
                                valid_logls.append(valid_logl)  
                                print(  'Valid log2BPC = ' + str(valid_logl) ) 
                            elif reportmode == 2:  
                                ##TEST##
                                testlen = Testseq.shape[1]
                                feed_dict = { self.final_xtest: Testseq[:,0:-1] }
                                y_probs = sess.run(self.final_ytest, feed_dict ) 
                                test_bpc = np.sum(  Testseq[:,1:]*np.log2(y_probs + eps)  )/testlen
                                test_logls.append(test_bpc)  
                                print( 'Test log2BPC = ' + str(test_bpc) + 'gpu:'+ str(self.gpus[0]) )
                                
                                ##VALIDATION##
                                validlen = Validseq.shape[1]
                                feed_dict = { self.final_xvalid: Validseq[:,0:-1] }
                                y_probs = sess.run(self.final_yvalid, feed_dict ) 
                                valid_bpc = np.sum(  Validseq[:,1:]*np.log2(y_probs + eps)  )/validlen
                                valid_logls.append(valid_bpc)
                                print( 'Valid log2BPC = ' + str(valid_bpc)+ 'gpu:'+ str(self.gpus[0]) )

                        elif task in ['digits','speech']:
                                feed_dict = {self.xtest : Testseq[0]}
                                y_probs = sess.run(self.ytest, feed_dict = feed_dict)     
                                true_classes = np.argmax(Testseq[1], axis = 0)
                                est_classes = np.argmax(y_probs, axis = 0) 
                                test_accuracy = np.mean(true_classes == est_classes)
                                test_logls.append( test_accuracy ) 

                                feed_dict = {self.xvalid: Validseq[0]} 
                                y_probs = sess.run(self.yvalid, feed_dict = feed_dict) 
                                true_classes = np.argmax(Validseq[1], axis = 0)
                                est_classes = np.argmax(y_probs, axis = 0)
                                valid_accuracy = np.mean(true_classes == est_classes)
                                valid_logls.append( valid_accuracy ) 


                                print('Test Accuracy = ' + str(test_accuracy) + ' Validation Accuracy = ' + str(valid_accuracy) ) 


                else:
                    plt.subplot( 2, 1, 1)
                    pl(cst)

        #final validation/test for text
        if task == 'text':
            ##TEST##
            testlen = Testseq.shape[1]
            feed_dict = { self.final_xtest: Testseq[:,0:-1] }
            y_probs = sess.run(self.final_longytest, feed_dict ) 
            final_test_bpc = np.sum( Testseq[:,1:]*np.log2(y_probs + eps)  )/testlen
            test_logls.append(final_test_bpc)  
            print( 'Final Test log2BPC = ' + str(final_test_bpc) )
            
            ##VALIDATION##
            validlen = Validseq.shape[1]
            feed_dict = { self.final_xvalid: Validseq[:,0:-1] }
            y_probs = sess.run(self.final_longyvalid, feed_dict ) 
            final_valid_bpc = np.sum( Validseq[:,1:]*np.log2(y_probs + eps)  )/validlen
            valid_logls.append(final_valid_bpc)
            print( 'Final Valid log2BPC = ' + str(final_valid_bpc))
                                    


        return valid_logls,test_logls,cst,all_times   

    def tfgenerate_text(self, sess, Tseq = 100, x = None): 
        L = self.L
        E = np.eye(L) 
        
        y = np.zeros((L,Tseq) ) 
        y_probs = np.zeros((L,Tseq))
        y_init = E[:,np.random.choice(L)].reshape(L,1)   
        state = None
        for t in range(Tseq):
            if state == None:
                if x == None:
                    feed_dict = { self.xtest: y_init }
                else: 
                    feed_dict = { self.xtest: x[:, t:t+1] } 

            else:
                if x == None:
                    feed_dict = {self.xtest: y[:,t-1:t], self.state: state} 
                else:
                    feed_dict = {self.xtest: x[:, t:t+1], self.state: state }
           
            pnext,state = sess.run( [self.ytest, self.nstate], feed_dict ) 
            
            y_probs[:,t] = pnext.squeeze()

            if self.outstage == 'softmax':
                y[:,t] = E[:,np.random.choice(L, p = pnext.squeeze())]
            elif self.outstage == 'sigmoid':
                y[:,t] = (np.random.rand(L) < pnext.squeeze())    
            elif self.outstage == 'linear':
                y[:,t] = pnext.squeeze()


        eps = 1e-9
        if not(x == None):
            
            if self.outstage == 'softmax':
                log2bpc = np.sum( x[:,1:Tseq+1]*np.log2(y_probs + eps)  )/Tseq

                logl = log2bpc 
            else:
                logl = np.sum(  x[:,1:Tseq+1]*np.log(y_probs+eps) + (1-x[:,1:Tseq+1])*np.log(1-y_probs +eps))
            
            
        else:
            logl = None

        return y, logl 

    def generate_text(self, T_seq, x = None, case = 'vector_w_conv'):
        
        if case == 'vector_w_conv':
            U = self.U.eval() 
            V = self.V.eval()
            b1 = self.b1.eval()
            b2 = self.b2.eval()
            w = self.w.eval()
            
            K = self.K 
            L = self.L 
            ntaps = self.ntaps
            #w = np.zeros(( K, ntaps ) ) 
            #w[:,0:27] = self.w.eval()[:,23:]
            #w[:,27:] = self.w.eval()[:,0:23]
            #iw[:,
            #for k in range(K):
            #    for n in range(ntaps):
            #        w[k, n] = self.w.eval()[0, n, k, 0] 

            h = np.zeros((K, T_seq))
            y = np.zeros((L, T_seq))
           
            E = np.eye(L)
            if x == None:
                burnin = 50 
                y[:,0:burnin] = E[:, np.random.choice(L, burnin )] 
            else: 
                burnin = 0 
            #pdb.set_trace()    


            for t in range(burnin, T_seq):
                for n in range(0, np.min( [ntaps, t+1])):
                    if x == None:
                        h[:,t] = h[:,t] + w[:,n]*np.dot(U, y[:, t-n-1] )
                    else:    
                        h[:,t] = h[:,t] + w[:,n]*np.dot(U, x[:,t-n] ) 
                h[:,t] = h[:,t] + np.squeeze(b1)
                y[:,t] = infinite_tap_rnn.softmax(  np.dot(V, np.tanh(h[:,t])) + np.squeeze(b2) )
            return y

def return_Klimits(model, wform, data):

    if data == 'JSB Chorales':
        min_params = 3e5; max_params = 9e5 
        if (model == 'mod_lstm') & (wform == 'diagonal'):    
            K_min = 175; K_max = 440
        elif (model == 'mod_lstm') & (wform == 'full'):    
            K_min = 115; K_max = 264 
        elif (model == 'mod_lstm') & (wform == 'constant'):
            K_min = 176; K_max = 441
        elif (model == 'mod_lstm') & (wform == 'scalar'):
            K_min = 176; K_max = 441

        elif (model == 'gated_w') & (wform == 'diagonal'):    
            K_min = 252; K_max = 635 
        elif (model == 'gated_w') & (wform == 'full'):    
            K_min = 164; K_max = 374 
        elif (model == 'gated_w') & (wform == 'constant'):
            K_min = 253; K_max = 635 
        elif (model == 'gated_w') & (wform == 'scalar'):    
            K_min = 253; K_max = 635
        
        elif (model == 'gated_wf') & (wform == 'diagonal'):    
            K_min = 204; K_max = 512
        elif (model == 'gated_wf') & (wform == 'full'):    
            K_min = 133; K_max = 305
        elif (model == 'gated_wf') & (wform == 'constant'):   #these seem to be fine also 
            K_min = 204; K_max = 512 
        elif (model == 'gated_wf') & (wform == 'scalar'):    
            K_min = 204; K_max = 512 



    elif data == 'Piano-midi.de':
        min_params = 7e5; max_params = 1.4e6 

        if (model == 'mod_lstm') & (wform == 'diagonal'):    
            K_min = 268; K_max = 540 
        elif (model == 'mod_lstm') & (wform == 'full'):    
            K_min = 175; K_max = 325
        elif (model == 'mod_lstm') & (wform == 'constant'):    
            K_min = 276; K_max = 560 
        elif (model == 'mod_lstm') & (wform == 'scalar'):    
            K_min = 276; K_max = 560 

        elif (model == 'gated_w') & (wform == 'diagonal'):    
            K_min = 385; K_max = 775 
        elif (model == 'gated_w') & (wform == 'full'):    
            K_min = 251; K_max = 462
        elif (model == 'gated_w') & (wform == 'constant'):
            K_min = 394; K_max = 791
        elif (model == 'gated_w') & (wform == 'scalar'):    
            K_min = 394; K_max = 791




        elif (model == 'gated_wf') & (wform == 'diagonal'):    
            K_min = 310; K_max = 630 
        elif (model == 'gated_wf') & (wform == 'full'):    
            K_min = 204; K_max = 380
        elif (model == 'gated_wf') & (wform == 'constant'):   #these seem to be fine also 
            K_min = 313; K_max = 630 
        elif (model == 'gated_wf') & (wform == 'scalar'):    
            K_min = 313; K_max = 630 

        

    elif data == 'Nottingham':
        min_params = 4e5; max_params = 1.2e6 

        if (model == 'mod_lstm') & (wform == 'diagonal'):    
            K_min = 203; K_max = 510#268,540 
        elif (model == 'mod_lstm') & (wform == 'full'):    
            K_min = 133; K_max = 305
        elif (model == 'mod_lstm') & (wform == 'constant'):    
            K_min = 203; K_max = 510#268,540 
        elif (model == 'mod_lstm') & (wform == 'scalar'):    
            K_min = 203; K_max = 510#268,540 


        elif (model == 'gated_w') & (wform == 'diagonal'):  #seems ok 
            K_min = 290; K_max = 730
        elif (model == 'gated_w') & (wform == 'full'):    
            K_min = 190; K_max = 432
        elif (model == 'gated_w') & (wform == 'constant'):  #seems ok 
            K_min = 290; K_max = 730
        elif (model == 'gated_w') & (wform == 'scalar'):  #seems ok 
            K_min = 290; K_max = 730


        elif (model == 'gated_wf') & (wform == 'diagonal'):    
            K_min = 236; K_max = 590 
        elif (model == 'gated_wf') & (wform == 'full'):    
            K_min = 154; K_max = 352
        elif (model == 'gated_wf') & (wform == 'constant'):  #seems ok 
            K_min = 236; K_max = 592
        elif (model == 'gated_wf') & (wform == 'scalar'):  #seems ok 
            K_min = 236; K_max = 592



    elif data == 'MuseData':
        min_params = 7e5; max_params = 1.4e6 

        if (model == 'mod_lstm') & (wform == 'diagonal'):    
            K_min = 268; K_max = 540
        elif (model == 'mod_lstm') & (wform == 'full'):    
            K_min = 175; K_max = 325
        elif (model == 'mod_lstm') & (wform == 'constant'):    
            K_min = 268; K_max = 540
        elif (model == 'mod_lstm') & (wform == 'scalar'):    
            K_min = 268; K_max = 540


        elif (model == 'gated_w') & (wform == 'diagonal'):    
            K_min = 385; K_max = 775 
        elif (model =='gated_w') & (wform == 'full'):    
            K_min = 251; K_max = 462
        elif (model == 'gated_w') & (wform == 'constant'):    
            K_min = 385; K_max = 775 
        elif (model == 'gated_w') & (wform == 'scalar'):    
            K_min = 385; K_max = 775 


        elif (model == 'gated_wf') & (wform == 'diagonal'):    
            K_min = 312; K_max = 627
        elif (model == 'gated_wf') & (wform == 'full'):    
            K_min = 204; K_max = 377
        elif (model == 'gated_wf') & (wform == 'constant'):    
            K_min = 312; K_max = 629
        elif (model == 'gated_wf') & (wform == 'scalar'):    
            K_min = 312; K_max = 629

    elif data == 'mnist':
        min_params = 1e5; max_params = 1e6 

        if (model == 'bi_mod_lstm') & (wform == 'diagonal'):
            K_min = 50; K_max = 350
        elif (model in ['bi_mod_lstm','bi_lstm']) & (wform == 'full'):    
            K_min = 40; K_max = 200
        
        elif (model == 'bi_gated_w') & (wform == 'diagonal'):
            K_min = 50; K_max = 500
        elif (model == 'bi_gated_w') & (wform == 'full'):    
            K_min = 40; K_max = 300

        elif (model == 'bi_gated_wf') & (wform == 'diagonal'):
            K_min = 50; K_max = 410
        elif (model == 'bi_gated_wf') & (wform == 'full'):    
            K_min = 40; K_max = 250

    elif data == 'timit':
        min_params = 1e4; max_params = 1e6 

        if (model == 'bi_mod_lstm') & (wform == 'diagonal'):
            K_min = 50; K_max = 350
        elif (model in ['bi_mod_lstm','bi_lstm']) & (wform == 'full'):    
            K_min = 40; K_max = 200
        
        elif (model == 'bi_gated_w') & (wform == 'diagonal'):
            K_min = 50; K_max = 500
        elif (model == 'bi_gated_w') & (wform == 'full'):    
            K_min = 40; K_max = 300

        elif (model == 'bi_gated_wf') & (wform == 'diagonal'):
            K_min = 50; K_max = 410
        elif (model == 'bi_gated_wf') & (wform == 'full'):    
            K_min = 40; K_max = 250

        elif model == 'multi_layer_ff':
            min_params = 1e4; max_params = 3e6
            K_min = 100; K_max = 1000

    return K_min, K_max, min_params, max_params 

def generate_random_hyperparams(lr_min, lr_max, K_min, K_max, num_layers_min, num_layers_max):
    # random search for learning rate
    random_lr_exp = np.random.uniform(lr_min, lr_max)
    random_lr = 10**(random_lr_exp)
    # Random search for num_layers
    random_num_layers = np.random.choice(np.arange(num_layers_min, num_layers_max + 1),1)[0]
    # Random search for K
    random_K = np.random.choice(np.arange(K_min, K_max+1),1)[0]

    return random_lr, random_K, random_num_layers

def load_data(task, data):

    if task == 'text':
        print('Task is text')

        if data == 'ptb':
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
        filename = data +'.pickle'
        dataset = load_musicdata( filename ) 
        Trainseq = np.concatenate(dataset[0],axis=1 )  # Training data
        Validseq = dataset[1] # Validation data
        Testseq = dataset[2] # Test data

        if data in ['Piano-midi.de', 'JSB Chorales'] :
            mbatchsize = Trainseq.shape[1]-1 #this is the batch size. 
        elif data == 'Nottingham':
            mbatchsize = (Trainseq.shape[1] - 1)/2  
        elif data == 'MuseData':
            mbatchsize = (Trainseq.shape[1] - 1)/2  

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

    elif task == 'speech':
        filehandle = open('timit.pickle','rb')
        sets = cPickle.load(filehandle)

        trlen = sets[0][0].shape[1]
        Trainseq = [sets[0][0][:,0:trlen],sets[0][1][:,0:trlen]]
        
        tstlen = sets[1][0].shape[1] #chop out the remainder 
        tstlen = tstlen - tstlen%200
        Testseq = [sets[1][i][:,:tstlen] for i in range(2)]
        
        validlen = sets[2][0].shape[1]
        validlen = validlen - validlen%200
        Validseq = [sets[2][i][:,:validlen] for i in range(2)]

        batchsize = np.min([trlen,140000])
        
    parameters = {'batchsize':batchsize,
                  'L1':L1,
                  'L2':L2,
                  'outstage':outstage,
                  'mapping_mode':mapping_mode,
                  'num_steps':num_steps}

    return {'Train':df_train, 'Test':df_test, 'Validation':df_valid}, parameters


class SimpleDataIterator():
    def __init__(self, df):
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
            temp = list(part['data'].values)
            data = np.transpose(np.asarray(temp),[2,0,1])  
            labels = part['labels'].values
            lengths = part['lengths'].values
        
        if self.cursor+n >= self.size:
            self.epochs += 1
            self.shuffle()
        else:
            self.cursor += n

        return data, labels, lengths


def load_multiple_textdata(filenames):
    'This file reads multiple text files given in filenames, and it returns a list containing the one hot coded representation of these files'
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


def load_musicdata(fl):

    filename = open(fl,'rb')
    dataset = cPickle.load(filename)
    
    dataset_list = [dataset['train'],dataset['valid'],dataset['test']]
    
    #this part extracts the max and min from the data
    lst = []
    for i in range(3):
        for j in range(len(dataset_list[i])):
            lst.extend( list(itertools.chain.from_iterable( dataset_list[i][j] )) )                     
    max_val = max(lst) 
    min_val = min(lst) 
    ################### 

    #this part puts the data in binary matrix format 
    mat_data= [[],[],[]]
    L = max(lst) + 1 
    for i in range(3):
        for j in range(len(dataset_list[i])):
            T = len(dataset_list[i][j] ) #length of the sequence  
            mat = np.zeros((L, T))
            for t in range(len( dataset_list[i][j] ) ):
                mat[dataset_list[i][j][t],t] = 1    
           
            mat = mat[min_val-1:max_val,:] #eliminate the empty parts of the data
            mat_data[i].append(mat)

    return mat_data


def reconstruct_text(mat,fmapping):
    reclist = []
    for j in range(mat.shape[1]):
        arg = np.argmax(mat[:,j])
        reclist.append(fmapping[arg])
    return ''.join(reclist)    

class Error(Exception):
    pass

class num_paramsError(Error):
    pass

