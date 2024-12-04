#Bolstering error calculation
import jax
from jax import numpy as jnp
from jax import random
from jax import jit

#Qudratic loss
@jit
def quad_loss(obs,pred):
    """
    Quadratic loss
    -------
    Parameters
    ----------
    obs : jax.numpy.array

        Observed values

    x : jax.numpy.array

        Predicted values

    Returns
    -------
    jax.numpy.array
    """
    return jnp.pow(obs - pred,2)

#Bolstering for each point
def bolstering_loss(psi,x,y,k,mc_sample = 100,key = 0,loss = quad_loss):
    """
    Gaussian Bolstering loss for a point (x,y)
    -------
    Parameters
    ----------
    psi : functions

        Estimated function

    x : jax.numpy.array

        x-point

    y : jax.numpy.array

        y-point

    k : jax.numpy.array

        Kernel matrix

    mc_sample : int

        Number of points for Monte Carlo integration

    key : int

        Seed for sampling

    loss : function

        Loss function

    Returns
    -------
    float
    """
    #Number of variables
    d = x.shape[0]

    #Sample point
    if k.shape[1] == d: #Bolstering on X-direction
      input = x
    else: #Bolstering on XY-direction
      input = jnp.column_stack((x.reshape((1,d)),y.reshape((1,1))))

    #Monte Carlo integration
    test_points = jax.random.multivariate_normal(key,input,k,shape = (mc_sample,))

    #Calculate error
    if k.shape[1] == d:
      error_test_points = loss(psi(test_points),y)
    else:
      error_test_points = loss(psi(test_points[:,:-1]),test_points[:,-1].reshape((mc_sample,1)))

    return jnp.mean(error_test_points)

#Bolstering error estimator
def bolstering(psi,x,y,k,mc_sample = 100,key = 0,loss = quad_loss):
    """
    Gaussian Bolstering error estimator
    -------
    Parameters
    ----------
    psi : function

        Estimated function

    x : jax.numpy.array

        x-data

    y : jax.numpy.array

        y-data

    k : jax.numpy.array

        Kernel matrix

    mc_sample : int

        Number of points for Monte Carlo integration

    key : int

        Seed for sampling

    loss : function

        Loss function

    Returns
    -------
    float
    """
    #Sample size and number of variables
    n = x.shape[0]
    d = x.shape[1]

    #Repeat kernel if it is equal for all points
    if len(k.shape) < 3:
        k = jnp.tile(k,(n,1,1))

    #Generate keys
    keys = jax.random.split(jax.random.PRNGKey(key),mc_sample)

    #Calculate the bolstered loss for each sample point
    bi = lambda i: bolstering_loss(psi,k[i,:,:],x[i,:],y[i,:],mc_sample,random.PRNGKey(keys[i,0]),loss)
    bolst = jax.vmap(bi)(jnp.array(range(n)).reshape((n,)))

    return jnp.mean(bolst)
