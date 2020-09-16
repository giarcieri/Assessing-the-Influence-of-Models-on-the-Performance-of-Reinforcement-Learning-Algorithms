import tensorflow as tf
'''
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
'''
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense

import numpy as np

class PNN:
    """
    Builds basic BNN (Anchored Ensembling) model with uncertainty 
    """
    
    def __init__(self, env, reg = 'anc', n_hidden = 40, activation_in = 'relu', data_noise = 0.001, n_data = 32, epochs = 30, 
                 l_rate = 0.001, ens_num = 0, env_name = 'Pendulum-v0'):
        """
        :env: environment
        :param reg: type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
        :param n_hidden: units per layer
        :param activation_in: activation hidden layers - relu tanh sigmoid
        :param data_noise: estimated data_noise
        :param n_data: number of data 
        :param epochs: epochs
        :param l_rate: lr
        """
        self.env = env
        self.reg = reg
        self.n_hidden = n_hidden
        self.activation_in = activation_in
        self.data_noise = data_noise
        self.n_data = n_data
        self.epochs = epochs
        self.l_rate = l_rate
        self.ens_num = ens_num 
        self.env_name = env_name
        
        D = self.env.observation_space.shape[0] 
        A = self.env.action_space.shape[0]
        Q = D + A # input shape
        
        W1_var = 20/Q                 # 1st layer weights and biases
        W_mid_var = 1/self.n_hidden   # 2st layer weights and biases
        W_last_var = 1/self.n_hidden  # 3st layer weights and biases
        
        # get initialisations, and regularisation values
        W1_lambda = self.data_noise/(D*W1_var) 
        W1_anc = np.random.normal(loc=0,scale=np.sqrt(W1_var),size=[Q,self.n_hidden])
        W1_init = np.random.normal(loc=0,scale=np.sqrt(W1_var),size=[Q,self.n_hidden])
        
        b1_var = W1_var
        b1_lambda =  self.data_noise/(D*b1_var)
        b1_anc = np.random.normal(loc=0,scale=np.sqrt(b1_var),size=[self.n_hidden])
        b1_init = np.random.normal(loc=0,scale=np.sqrt(b1_var),size=[self.n_hidden])
        
        W_mid_lambda = self.data_noise/(D*W_mid_var)
        W_mid_anc = np.random.normal(loc=0,scale=np.sqrt(W_mid_var),size=[self.n_hidden,self.n_hidden])
        W_mid_init = np.random.normal(loc=0,scale=np.sqrt(W_mid_var),size=[self.n_hidden,self.n_hidden])
        
        b_mid_var = W_mid_var
        b_mid_lambda =  self.data_noise/(D*b_mid_var)
        b_mid_anc = np.random.normal(loc=0,scale=np.sqrt(b_mid_var),size=[self.n_hidden])
        b_mid_init = np.random.normal(loc=0,scale=np.sqrt(b_mid_var),size=[self.n_hidden])
        
        W_last_lambda = self.data_noise/(D*W_last_var)
        W_last_anc = np.random.normal(loc=0,scale=np.sqrt(W_last_var),size=[self.n_hidden, D])
        W_last_init = np.random.normal(loc=0,scale=np.sqrt(W_last_var),size=[self.n_hidden, D])
        
        # create custom regularised
        def custom_reg_W1(weight_matrix):
            if self.reg == 'reg':
                return K.sum(K.square(weight_matrix)) * W1_lambda/self.n_data
            elif self.reg == 'free':
                return 0.
            elif self.reg == 'anc':
                return K.sum(K.square(weight_matrix - W1_anc)) * W1_lambda/self.n_data

        def custom_reg_b1(weight_matrix):
            if self.reg == 'reg':
                return K.sum(K.square(weight_matrix)) * b1_lambda/self.n_data
            elif self.reg == 'free':
                return 0.
            elif self.reg == 'anc':
                return K.sum(K.square(weight_matrix - b1_anc)) * b1_lambda/self.n_data

        def custom_reg_W_mid(weight_matrix):
            if self.reg == 'reg':
                return K.sum(K.square(weight_matrix)) * W_mid_lambda/self.n_data
            elif self.reg == 'free':
                return 0.
            elif self.reg == 'anc':
                return K.sum(K.square(weight_matrix - W_mid_anc)) * W_mid_lambda/self.n_data

        def custom_reg_b_mid(weight_matrix):
            if self.reg == 'reg':
                return K.sum(K.square(weight_matrix)) * b_mid_lambda/self.n_data
            elif self.reg == 'free':
                return 0.
            elif self.reg == 'anc':
                return K.sum(K.square(weight_matrix - b_mid_anc)) * b_mid_lambda/self.n_data

        def custom_reg_W_last(weight_matrix):
            if self.reg == 'reg':
                return K.sum(K.square(weight_matrix)) * W_last_lambda/self.n_data
            elif self.reg == 'free':
                return 0.
            elif self.reg == 'anc':
                return K.sum(K.square(weight_matrix - W_last_anc)) * W_last_lambda/self.n_data
            
        model = Sequential()
        model.add(Dense(self.n_hidden, activation=self.activation_in, input_shape=(Q,),
            kernel_initializer=keras.initializers.Constant(value=W1_init),
            bias_initializer=keras.initializers.Constant(value=b1_init),
            kernel_regularizer=custom_reg_W1,
            bias_regularizer=custom_reg_b1))
        
        model.add(Dense(self.n_hidden, activation=self.activation_in,
            kernel_initializer=keras.initializers.Constant(value=W_mid_init),
            bias_initializer=keras.initializers.Constant(value=b_mid_init),
            kernel_regularizer=custom_reg_W_mid,
            bias_regularizer=custom_reg_b_mid))
        
        model.add(Dense(D, activation='linear', use_bias=False,
            kernel_initializer=keras.initializers.Constant(value=W_last_init),
            kernel_regularizer=custom_reg_W_last))
        
        model.compile(loss='mean_squared_error', 
            optimizer=keras.optimizers.Adam(lr=l_rate))
        
        self.model = model
        return 
    
    def fit(self, X, Y, batch_size = 32, epochs = 30, verbose = 2, validation_split=0.1):
        """
        Trains model
        :param epochs: defines how many times each training point is revisited during training time
        """
        # save checkpoints
        weights_file_std = './folder_models/PNN_'+str(self.ens_num)+'_'+str(self.env_name)+'_check_point_weights.h5'
        model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_std, monitor='val_loss', save_best_only=True,
                                   save_weights_only=True, mode='auto',verbose=0)
        
        Early_Stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.model.fit(X, Y, epochs=epochs,
                                         batch_size=batch_size, verbose=verbose,
                                         validation_split = validation_split, callbacks=[Early_Stop, 
                                                                                         #model_checkpoint
                                                                                        ])
        
    def predict(self, x_test, verbose=0):
        y_pred = self.model.predict(x_test, verbose=verbose)
        return y_pred
    
    def summary(self):
        self.model.summary()
        
    def get_weights(self):
        weights = self.model.get_weights()
        self.weights = weights
        return weights
    
    def set_weights(self, weights): 
        self.model.set_weights(weights)


class ens_PNNs:
    """
    Build an ensemble of BNNs (Anchored Ensembling) with uncertainty
    """
    def __init__(self, env, reg = 'anc', n_hidden = 40, activation_in = 'relu', data_noise = 0.001, n_data = 32, epochs = 30, 
                 l_rate = 0.001, n_ensemble = 5, ens_num = 0, env_name = 'Pendulum-v0'):
        """
        :env: environment
        :param reg: type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
        :param n_hidden: units per layer
        :param activation_in: activation hidden layers - relu tanh sigmoid
        :param data_noise: estimated data_noise
        :param n_data: number of data 
        :param epochs: epochs
        :param l_rate: lr
        :param n_ensemble: number of NNs
        """
        
        self.env = env
        self.reg = reg
        self.n_hidden = n_hidden
        self.activation_in = activation_in
        self.data_noise = data_noise
        self.n_data = n_data
        self.epochs = epochs
        self.l_rate = l_rate
        self.n_ensemble = n_ensemble
        self.ens_num = ens_num 
        self.env_name = env_name
        
        NNs=[]
        for m in range(self.n_ensemble):
            NNs.append(PNN(env = self.env, reg = self.reg, n_hidden = self.n_hidden, activation_in = self.activation_in, 
                           data_noise = self.data_noise, n_data = self.n_data, epochs = self.epochs, l_rate = self.l_rate, ens_num = m,
                           env_name = self.env_name))
        print(NNs[-1].summary())
        self.NNs = NNs
        return #NNs 
    
    def train(self, X_train, y_train, batch_size=32, validation_split=0.1):
        
        NNs_hist_train=[];
        for m in range(len(self.NNs)):  
            print('-- training: ' + str(m+1) + ' of ' + str(self.n_ensemble) + ' NNs --')
            hist = self.NNs[m].fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=self.epochs,
                      verbose=0,
                      validation_split = validation_split)
            #NNs_hist_train.append(hist.history['loss'])
        return NNs_hist_train 
            
    def predict_ensemble(self, x_test):
        ''' fn to predict given a list of NNs (an ensemble)''' 
        y_preds = []
        for m in range(len(self.NNs)):
            y_preds.append(self.NNs[m].predict(x_test, verbose=0))
        y_preds = np.array(y_preds) # predictions for all the NNs
        y_preds_mu = np.mean(y_preds,axis=0) # mean prediction
        y_preds_std = np.std(y_preds,axis=0, ddof = 1) # epistemic uncertainty 
        
        # add on data noise
        y_preds_std = np.sqrt(np.square(y_preds_std) + self.data_noise)

        self.y_preds_mu = y_preds_mu
        self.y_preds_std = y_preds_std
        self.y_preds = y_preds

        return y_preds_mu, y_preds, y_preds_std
    
    
    def uncertainty(self):
        y_preds_mu, y_preds, y_preds_std = self.y_preds_mu, self.y_preds, self.y_preds_std
        epistemic_uncertainty = np.var(y_preds, 0)#.mean(0)
        return epistemic_uncertainty 
    
    def get_weights(self):
        weights = []
        for n in range(len(self.NNs)):
            weights.append(self.NNs[n].get_weights())
        self.weights = weights
        return weights
    
    def set_weights(self, weights):
        for n in range(len(self.NNs)):
            self.NNs[n].set_weights(weights[n])

        

        
        
        
        
        
        
        
        
        