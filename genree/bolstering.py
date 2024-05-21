#Bolstering error calculation
import jax
from jax import numpy as jnp
from jax import random
from jax import jit

#Qudratic loss
@jit
def quad_loss(obs,pred):
    return jnp.pow(obs - pred,2)

#Bolstering for each point
def bolstering_index(psi,k,x,y,mc_sample,key,loss = quad_loss):
    #Number of variables
    d = x.shape[0]

    #Sample point
    if k.shape[1] == d:
      input = x
    else:
      input = jnp.column_stack((x.reshape((1,d)),y.reshape((1,1))))

    #Monte Carlo integration
    test_points = jax.random.multivariate_normal(key, input, k,shape = (mc_sample,))

    #Calculate error
    if k.shape[1] == d:
      error_test_points = loss(psi(test_points),y)
    else:
      error_test_points = loss(psi(test_points[:,:-1]),test_points[:,-1].reshape((mc_sample,1)))

    return jnp.mean(error_test_points)

#Bolstering error estimator
def bolstering(psi,x,y,k,key,mc_sample = 100,loss = quad_loss):
    #Assure data is matrix
    x = jnp.array(x)
    y = jnp.array(y)

    #Sample size and number of variables
    n = x.shape[0]
    d = x.shape[1]

    #Repeat kernel
    if len(k.shape) < 3:
        k = jnp.tile(k,(n,1,1))

    #Generate keys
    keys = jax.random.randint(key,(mc_sample,),0,1e9)

    #Calculate the bolstered loss for each sample point
    bi = lambda i: bolstering_index(psi,k[i,:,:],x[i,:],y[i,:],mc_sample,random.PRNGKey(keys[i]),loss)
    bolst = jax.vmap(bi)(jnp.array(range(n)).reshape((n,)))

    return jnp.mean(bolst)
