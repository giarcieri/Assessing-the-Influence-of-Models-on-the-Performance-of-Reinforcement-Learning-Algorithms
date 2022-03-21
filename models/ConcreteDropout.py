import sys
import numpy as np
#np.random.seed(0)


import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
'''
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, Wrapper, concatenate, InputSpec
'''
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, merge, concatenate
from keras.layers.wrappers import Wrapper
from keras.engine import InputSpec
from keras import initializers


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)



class BNN:
    """
    Builds Concrete Dropout NN 
    """
    
    def __init__(self, env, nb_units = 32, nb_layers = 3, activation = 'relu', n_data = 32, 
                 
                 dropout = 0.1, T = 10, tau = 1.0, lengthscale = 1e-4, ens_num = 0, train_flag = True, env_name = 'Pendulum-v0'):
        """
        :env: environment
        :param nb_units: units per layer
        :param nb_layers: number of hidden layers
        :param activation: layers activation
        :param n_data: number of data
        :param dropout: probability of perceptron being dropped out
        :param T: number of samples during test time
        :param tau: precision of prior
        :param lengthscale: lengthscale
        """
        self.env = env
        D = self.env.observation_space.shape[0]
        
        self.ens_num = ens_num
        self.env_name = env_name
        self.dropout = dropout
        self.T = T
        self.tau = tau
        self.lengthscale = lengthscale
        self.n_data = n_data

        self.weight_decay = ((1-self.dropout)*self.lengthscale**2)/(self.n_data*self.tau) 
        self.nb_units = nb_units
        self.nb_layers = nb_layers
        self.activation = activation
        self.train_flag = train_flag

        if K.backend() == 'tensorflow':
            K.clear_session()
        N = self.n_data 
        wd = self.lengthscale**2. / N
        dd = 2. / N
        A = self.env.action_space.shape[0]
        Q = D + A
        inp = Input(shape=(Q,))
        x = inp
        for _ in range(nb_layers):
            x = ConcreteDropout(Dense(self.nb_units, activation=self.activation), weight_regularizer=wd, dropout_regularizer=dd)(x)
        mean = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
        log_var = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x) 
        out = concatenate([mean, log_var])
        self.model = Model(inp, out)
    
        def heteroscedastic_loss_Gal(true, pred):
            mean = pred[:, :D]
            log_var = pred[:, D:]
            precision = K.exp(-log_var)
            return K.sum(precision * (true - mean)**2. + log_var, -1)



        self.model.compile(optimizer='adam', loss=heteroscedastic_loss_Gal)

    
    def fit(self, X, Y, batch_size = 32, epochs = 30, validation_split=0.1, verbose=2):
        """
        Trains model
        :param epochs: defines how many times each training point is revisited during training time
        """
        # save checkpoints
        weights_file_std = './folder_models/BNN_'+str(self.ens_num)+'_'+str(self.env_name)+'_check_point_weights.h5'
        model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_std, monitor='val_loss', save_best_only=True,
                                   save_weights_only=True, mode='auto',verbose=0)

        Early_Stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        #if self.train_flag:
        self.historyBNN = self.model.fit(X, Y, epochs=epochs,
                                     batch_size=batch_size, verbose=verbose,
                                     validation_split = validation_split, callbacks=[Early_Stop#, model_checkpoint
                                                                                    ])
        #self.model.load_weights(weights_file_std)
        #        tl,vl = historyBNN.history['loss'], historyBNN.history['val_loss'] 
        
    def predict(self, X_test):
        D = self.env.observation_space.shape[0] 
        Yt_hat = np.array([self.model.predict(X_test, verbose=0) for _ in range(self.T)])
        mean = Yt_hat[:, :, :D]  
        logvar = Yt_hat[:, :, D:]
        MC_pred = np.mean(mean, 0)
        self.MC_pred = MC_pred
        self.mean = mean
        self.logvar = logvar
        return MC_pred, mean, logvar
    
    def uncertainty(self):
        MC_pred, means, logvars = self.MC_pred, self.mean, self.logvar
        epistemic_uncertainty = np.var(means, 0)#.mean(0)
        logvar = np.mean(logvars, 0)  
        aleatoric_uncertainty = np.exp(logvar)#.mean(0)
        return epistemic_uncertainty#, aleatoric_uncertainty
    
    def summary(self):
        self.model.summary()
        
    def get_weights(self):
        weights = self.model.get_weights()
        self.weights = weights
        return weights
    
    def set_weights(self, weights): 
        self.model.set_weights(weights)
    

    
class ens_BNN:
    """
    Build an ensemble of Concrete Dropout NNs
    """
    
    def __init__(self, env, nb_units = 32, nb_layers = 3, activation = 'relu', n_data = 32, dropout = 0.1, T = 10, 
                 tau = 1.0, lengthscale = 1e-4, ens_num = 0, n_ensemble = 5, train_flag = True, env_name = 'Pendulum-v0'):
        """
        :env: environment
        :param nb_units: units per layer
        :param nb_layers: number of hidden layers
        :param activation: layers activation
        :param n_data: number of data
        :param dropout: probability of perceptron being dropped out
        :param T: number of samples during test time
        :param tau: precision of prior
        :param lengthscale: lengthscale
        :param n_ensemble: number of BNNs
        """
        self.env = env
        #D = self.env.observation_space.shape[0]
        self.ens_num = ens_num
        self.n_ensemble = n_ensemble
        self.dropout = dropout
        self.T = T
        self.tau = tau
        self.lengthscale = lengthscale
        self.n_data = n_data

        self.weight_decay = ((1-self.dropout)*self.lengthscale**2)/(self.n_data*self.tau) 
        self.nb_units = nb_units
        self.nb_layers = nb_layers
        self.activation = activation
        self.train_flag = train_flag
        self.env_name = env_name
        
        NNs=[]
        for m in range(self.n_ensemble):
            #np.random.seed(seed=m)
            #random.seed(m)
            NNs.append(BNN(env = self.env, nb_units = self.nb_units, nb_layers = self.nb_layers, activation = self.activation, 
                           n_data = self.n_data, dropout = self.dropout, T = self.T, tau = self.tau, lengthscale = self.lengthscale, 
                           ens_num = m, train_flag = self.train_flag, env_name = self.env_name))
        print(NNs[-1].summary())
        self.NNs = NNs
        return
    
    
    def train(self, X_train, y_train, batch_size=32, epochs=30, validation_split=0.1):
        
        NNs_hist_train=[]
        for m in range(len(self.NNs)):
            #np.random.seed(seed=m)
            #random.seed(m)
            print('-- training: ' + str(m+1) + ' of ' + str(self.n_ensemble) + ' NNs --')
            hist = self.NNs[m].fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_split = validation_split)
            #NNs_hist_train.append(hist.history['loss'])
        return NNs_hist_train
    
    def predict_ensemble(self, x_test):
        ''' fn to predict given a list of NNs (an ensemble)'''
        
        MC_preds = []
        means_preds = []
        logvar_preds = []
        for m in range(len(self.NNs)):
            #np.random.seed(seed=m)
            #random.seed(m)
            MC_pred, mean, logvar = self.NNs[m].predict(x_test)
            MC_preds.append(MC_pred)
            means_preds.append(mean)
            logvar_preds.append(logvar)
        MC_preds = np.array(MC_preds) # mean predictions for all the NNs
        means_preds = np.array(means_preds) # mean predictions for all the NNs and all the samples
        logvar_preds = np.array(logvar_preds) # var predictions for all the NNs and all the samples
        MC_pred = np.mean(MC_preds,axis=0) # mean prediction averaged over the NNs
        means = np.mean(means_preds, 0) # mean predictions for all the samples (averaged over the NNs)
        logvars = np.mean(logvar_preds, 0) # var predictions for all the samples (averaged over the NNs)

        self.MC_pred = MC_pred
        self.means = means
        self.logvars = logvars

        return MC_pred, means, logvars
    
    
    def uncertainty(self):
        MC_pred, means, logvars = self.MC_pred, self.means, self.logvars
        epistemic_uncertainty = np.var(means, 0)#.mean(0)
        logvar = np.mean(logvars, 0)  
        aleatoric_uncertainty = np.exp(logvar)#.mean(0)
        return epistemic_uncertainty#, aleatoric_uncertainty
    
    def get_weights(self):
        weights = []
        for n in range(len(self.NNs)):
            weights.append(self.NNs[n].get_weights())
        self.weights = weights
        return weights
    
    def set_weights(self, weights):
        for n in range(len(self.NNs)):
            self.NNs[n].set_weights(weights[n])
    
        
        
    
    
    
        
        