import gpflow
import numpy as np
from gpflow.ci_utils import ci_niter
gpflow.config.set_default_float(np.float64)

class GP:
    """Build Gaussian Process"""
    def __init__(self, env, gp_model, kernel = 'Matern52', mean_function=None, batch_size=32):
        """
        :env: environment
        :gp_model: two GPs are defined, namely 'GPR' and 'SVGP'. The former is the one applied in the evaluation.
        :param kernel: kernel function for the GP
        """
        self.env = env
        self.gp_model = gp_model
        self.kernel = kernel
        self.mean_function = mean_function
        self.batch_size = batch_size
        
        D = self.env.observation_space.shape[0] 
        A = self.env.action_space.shape[0]
        Q = D + A # input shape
        lengthscales = np.random.uniform(0, 1, Q)
        if self.gp_model == 'GPR' and self.kernel == 'Matern52': 
            k = gpflow.kernels.Matern52(lengthscales=lengthscales)
        elif self.gp_model == 'SVGP':
            k = gpflow.kernels.SharedIndependent(gpflow.kernels.SquaredExponential() + gpflow.kernels.Linear(), output_dim=D)
            state = self.env.reset().reshape(1, D)
            action = self.env.action_space.sample().reshape(1, A)
            Z = np.concatenate((state, action), axis=1)
            iv = gpflow.inducing_variables.SharedIndependentInducingVariables(gpflow.inducing_variables.InducingPoints(Z))
            model = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=D)
            self.model = model
            
        self.D = D
        self.Q = Q
        self.lengthscales = lengthscales
        self.k = k
        
        return
    
    def fit(self, X, Y, variance = 0.001, optimize = True, maxiter=100):
        
        print('-- fitting gaussian process on ' + str(X.shape[0]) + ' data --')
        
        opt = gpflow.optimizers.Scipy()
        mean_X = X.mean()
        std_X = X.std()
        mean_Y = Y.mean()
        std_Y = Y.std()
        X = (X - mean_X) / std_X
        Y = (Y - mean_Y) / std_Y
        self.mean_X = mean_X
        self.std_X = std_X
        self.mean_Y = mean_Y
        self.std_Y = std_Y
        if self.gp_model == 'GPR':
            model = gpflow.models.GPR(data=(np.array(X, dtype=float), np.array(Y, dtype=float)), kernel=self.k,
                                      mean_function=self.mean_function, noise_variance = variance)
            #model.likelihood.variance.assign(variance)
            #model.likelihood.variance.fixed = True
            #if optimize:
            #    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=maxiter))
            self.model = model
        elif self.gp_model == 'SVGP':
            data = X, Y
            MAXITER = ci_niter(2000)
            #self.model.likelihood.variance.assign(variance)
            if optimize:
                opt.minimize(self.model.training_loss_closure(data), variables=self.model.trainable_variables, 
                             method="l-bfgs-b", options={"maxiter": MAXITER},)
                
        return
    
    def predict(self, x_test):
        
        x_test = (x_test - self.mean_X) / self.std_X
        mean, var = self.model.predict_f(np.array(x_test, dtype=float))
        mean = mean * self.std_Y + self.mean_Y
        var = var * self.std_Y 
        self.mean = mean
        self.var = var
        means_tot = np.nan # just for compatibility with other models
        return mean, means_tot, var
    
    def uncertainty(self):
        epistemic_uncertainty = self.var
        return epistemic_uncertainty
        
        
        
        
        
        
        
        
        
        