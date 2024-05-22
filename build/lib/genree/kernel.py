#Kernel estimation module
import jax
from jax import numpy as jnp
from jax import random
from jax import jit

#Mahalanobis distance matrix
@jit
def dist_mahalanobis(x,S = None):
    #Assure data is matrix
    x = jnp.array(x)

    #Sample size and number of variables
    n = x.shape[0]
    d = x.shape[1]

    #Compute the kernel
    if S is None:
        S = jnp.diag(jnp.array([1] * d),0)
    Sinvert = jnp.linalg.inv(S)
    D = jax.vmap(jax.vmap(lambda x,y: jnp.dot(x,y)))(jnp.matmul(x[None,:] - x[:,None],Sinvert),x[None,:] - x[:,None])

    return jnp.sqrt(D)

#Mahalanobis distance to a center
@jit
def mahalanobis(x,center,S = None):
    #Assure data is matrix
    x = jnp.array(x)

    #Sample size and number of variables
    n = x.shape[0]
    d = x.shape[1]

    #Compute the kernel
    if S is None:
        S = jnp.diag(jnp.array([1] * d),0)
    Sinvert = jnp.linalg.inv(S)
    dist = lambda x: jnp.sqrt(jnp.matmul(jnp.matmul((x - center).reshape((1,d)),Sinvert),jnp.transpose((x - center).reshape((1,d)))))[0,0]
    D = jax.vmap(dist)(x)

    return D

#Sample from distance (delta(X)) distribution
def sample_dist(x,sigma,S,mc_sample,key):
    #Sample size and number of variables
    n = x.shape[0]
    d = x.shape[1]

    #Generate keys
    keys = jax.random.randint(key,(mc_sample + 1,),0,1e9)

    #Sample means (distribution of mixture that generated the point)
    mean_index = jnp.array(jax.random.randint(random.PRNGKey(keys[0]),(mc_sample,),0,n),dtype = jnp.uint32)

    #Sample points
    sample = lambda i: jax.random.multivariate_normal(random.PRNGKey(keys[i + 1]), x[mean_index[i],:], sigma*S, shape = (1,)).reshape((1,d))
    sample = jax.vmap(sample)(jnp.array(range(mc_sample)).reshape((mc_sample,))).reshape((mc_sample,d))

    #Compute distances
    distance = lambda s: jnp.min(mahalanobis(x = x,center = s,S = S))
    distance = jax.vmap(distance)(sample)

    return jnp.mean(distance)

#E-Step
@jit
def e_step(i,x,n,S,lamb):
    W = jax.scipy.stats.multivariate_normal.pdf(x,x[i,:],S[i,:,:]).at[i].set(-lamb)
    return W

#M step
@jit
def m_step(i,x,W,n):
    Snext = jnp.matmul(W[i,:]*jnp.transpose(x[i,:] - x),x[i,:] - x)
    return Snext

#Estimate the kernel
def kernel_estimator(x,key,method = "chi",S = None,S0 = None,mc_sample = 100,ec = 1e-6,grid_delta = 0.001,lamb = 1,trace = False):
    #Assure data is jax.Array
    x = jnp.array(x)

    #Sample size and number of variables
    n = x.shape[0]
    d = x.shape[1]

    #Compute the kernel via chi approximation
    if method == "chi":
        #Initialize S
        if S is None:
            S = jnp.diag(jnp.array([1] * d),0)

        #Compute delta_bar
        delta_bar = jnp.mean(jnp.min(dist_mahalanobis(x,S) + jnp.diag(jnp.array([jnp.inf]*n),0),1))

        #Compute expectation of chi random variable
        dom = jnp.linspace(0,10*d,round(10*d/0.001))[1:]
        int_chi = jax.scipy.integrate.trapezoid(jnp.sqrt(dom)*jax.scipy.stats.chi2.pdf(dom,d),dom)

        #Compute sigma
        sigma = delta_bar/int_chi

        return jnp.tile(sigma*S,(n,1,1))
    #Compute the kernel via exact Method of moments
    elif method == "mm":
        #Initialize S
        if S is None:
            S = jnp.diag(jnp.array([1] * d),0)

        #Compute delta_bar
        delta_bar = jnp.mean(jnp.min(dist_mahalanobis(x,S) + jnp.diag(jnp.array([jnp.inf]*n),0),1))

        #Generate keys
        keys = jax.random.randint(key,(100000 + 1,),0,1e9)

        #Compute mean distance for each sigma in a grid
        sd = jax.jit(lambda sigma,key: sample_dist(x,sigma,S,mc_sample,key))
        r = 0
        mean_dist = []
        while (max(mean_dist + [0]) - delta_bar) < 0 and r < 1e5:
          r = r + 1
          sigma = r*grid_delta
          mean_dist = mean_dist + [sd(sigma,random.PRNGKey(keys[r]))]
        sigma = (r - 1)*grid_delta

        if r == 1e5:
            print("Grid for method of moments is too small!")
            return 0

    return jnp.tile(sigma*S,(n,1,1))
    #Compute the kernel via maximum pseudolikelihood
    elif(method == "mpe"):
        #Initialize S
        if S0 is None:
            S0 = jnp.tile(jnp.diag(jnp.array([1] * d),0),(n,1,1))
        S = S0

        #While criteria is not met
        if trace:
          print("-----------------------------------------------------\n Initializing EM algorithm\n-----------------------------------------------------\n")
        criteria = 1
        t = 0
        while criteria:
            #E step
            W = jax.vmap(lambda i: e_step(i,x,n,S,lamb))(jnp.array(range(n))) + lamb
            W = W/jnp.sum(W,axis = 0)

            #M step
            Snext = jax.vmap(lambda i: m_step(i,x,W,n))(jnp.array(range(n)))/(n-1)

            #Dif
            dif = jnp.max(S - Snext)
            if dif < ec:
                criteria = 0
            S = Snext
            del Snext
            t = t + 1
            if trace:
                print("t = " + str(t) + " step size = " + str(round(dif,8)) + "\n")

    return S
