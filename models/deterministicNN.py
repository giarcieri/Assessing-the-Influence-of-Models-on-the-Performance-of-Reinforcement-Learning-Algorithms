#from tensorflow import keras
import keras
import tensorflow as tf
import numpy as np


class NN:
    """
    Builds basic NN model 
    """
    
    def __init__(self, env, nb_layers = 3, n_hidden = 32, activation_in = 'relu', 
                 kernel_initializer="he_normal", epochs = 30, l_rate = 0.001, ens_num = 0, env_name = 'Pendulum-v0'):
        """
        :env: environment
        :param n_hidden: units per layer
        :param nb_layers: number of hidden layers
        :param activation_in: activation hidden layers
        :param kernel_initializer: kernel initializer
        :param data_noise: estimated data_noise
        :param epochs: epochs
        :param l_rate: lr
        """
        self.env = env
        self.nb_layers = nb_layers
        self.n_hidden = n_hidden
        self.activation_in = activation_in
        self.kernel_initializer = kernel_initializer
        self.epochs = epochs
        self.l_rate = l_rate
        self.ens_num = ens_num 
        self.env_name = env_name
        
        D = self.env.observation_space.shape[0] #output shape
        A = self.env.action_space.shape[0]
        Q = D + A # input shape
        
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[Q]))
        for layer in range(self.nb_layers):
            model.add(keras.layers.Dense(self.n_hidden, activation=self.activation_in, kernel_initializer=self.kernel_initializer))
        model.add(keras.layers.Dense(D))
        optimizer = keras.optimizers.Adam(lr=self.l_rate)
        model.compile(loss="mse", optimizer=optimizer)
        self.model = model
        #return #model
    
    def fit(self, X, Y, batch_size = 32, epochs = 30, verbose = 2, validation_split=0.1):
        """
        Trains model
        :param epochs: defines how many times each training point is revisited during training time
        """
        # save checkpoints
        weights_file_std = './folder_models/NN_'+str(self.ens_num)+'_'+str(self.env_name)+'_check_point_weights.h5'
        model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_std, monitor='val_loss', save_best_only=True,
                                   save_weights_only=True, mode='auto',verbose=0)
       
        Early_Stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.model.fit(X, Y, epochs=epochs,
                                         batch_size=batch_size, verbose=verbose,
                                         validation_split = validation_split, callbacks=[Early_Stop#, model_checkpoint
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
    
class ens_NNs: 
    """
    Build an ensemble of NN 
    """
    def __init__(self, env, nb_layers = 3, n_hidden = 32, activation_in = 'relu', kernel_initializer="he_normal", 
                 epochs = 30, l_rate = 0.001, n_ensemble = 5, ens_num = 0, env_name = 'Pendulum-v0'):
        """
        :env: environment
        :param n_hidden: units per layer
        :param nb_layers: number of hidden layers
        :param activation_in: activation hidden layers
        :param kernel_initializer: kernel initializer
        :param data_noise: estimated data_noise
        :param epochs: epochs
        :param l_rate: lr
        :param n_ensemble: number of NNs
        """
        
        self.env = env
        self.nb_layers = nb_layers
        self.n_hidden = n_hidden
        self.activation_in = activation_in
        self.kernel_initializer = kernel_initializer
        self.epochs = epochs
        self.l_rate = l_rate
        self.n_ensemble = n_ensemble
        self.ens_num = ens_num 
        self.env_name = env_name
        
        NNs=[]
        for m in range(self.n_ensemble):
            NNs.append(NN(env = self.env, nb_layers = self.nb_layers, n_hidden = self.n_hidden, activation_in = self.activation_in, 
                          kernel_initializer=self.kernel_initializer, epochs = self.epochs, l_rate = self.l_rate, ens_num = m,
                          env_name = self.env_name))
        print(NNs[-1].summary())
        self.NNs = NNs
        #return NNs 
    
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
        y_preds = np.array(y_preds)
        y_preds_mu = np.mean(y_preds,axis=0)
        y_preds_std = np.std(y_preds,axis=0, ddof = 1)
        
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
            
                           

        
        
        
        
        
        
        
        

