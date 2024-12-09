#Kernel estimation module
import jax
from jax import numpy as jnp
from jax import random
from jax import jit
from jax.scipy.linalg import eigh
from genree import bolstering as gb

#Approximate by a posiitve definite matrix
def nearest_pd(A,e = 1e-4):
    """
    Approximate by a positive definite matrix
    -------
    Parameters
    ----------
    A : jax.numpy.array

        Matrix

    e : float

        Value to substitute for negative eigenvalues

    Returns
    -------
    jax.numpy.array
    """
    #Compute the eigenvalues and eigenvectors of A
    eigvals, eigvecs = eigh(A)

    #Set the negative eigenvalues to small positive values
    eigvals = jax.nn.relu(eigvals) + e

    #Reconstruct the matrix from the eigenvalues and eigenvectors
    return jnp.dot(eigvecs, jnp.dot(jnp.diag(eigvals), eigvecs.T))


#Mahalanobis distance matrix
@jit
def dist_mahalanobis(data,S = None):
    """
    Mahalanobis distance
    -------
    Parameters
    ----------
    data : jax.numpy.array

        Data

    S : jax.numpy.array

        Covariance matrix

    Returns
    -------
    jax.numpy.array
    """
    #Sample size and number of variables
    n = data.shape[0]
    d = data.shape[1]

    #Compute the kernel
    if S is None:
        S = jnp.diag(jnp.array([1] * d),0)

    #Compute the distance
    Sinvert = jnp.linalg.inv(S)
    D = jax.vmap(jax.vmap(lambda x,y: jnp.dot(x,y)))(jnp.matmul(data[None,:] - data[:,None],Sinvert),data[None,:] - data[:,None])

    return jnp.sqrt(D)

#Mahalanobis distance to a point
@jit
def mahalanobis(data,point,S = None):
    """
    Mahalanobis distance to a point
    -------
    Parameters
    ----------
    data : jax.numpy.array

        Data

    point : jax.numpy.array

        Point

    S : jax.numpy.array

        Covariance matrix

    Returns
    -------
    jax.numpy.array
    """
    #Sample size and number of variables
    n = data.shape[0]
    d = data.shape[1]

    #Compute the kernel
    if S is None:
        S = jnp.diag(jnp.array([1] * d),0)
    Sinvert = jnp.linalg.inv(S)

    #Compute distance
    dist = lambda x: jnp.sqrt(jnp.matmul(jnp.matmul((x - point).reshape((1,d)),Sinvert),jnp.transpose((x - point).reshape((1,d)))))[0,0]
    D = jax.vmap(dist)(data)

    return D

#Sample from distance (delta(X)) distribution
def sample_mean_dist(data,sigma,S,mc_sample = 100,key = 0):
    """
    Sample from theoretical distance to sample distribution and compute mean minimum distance to data
    -------
    Parameters
    ----------
    data : jax.numpy.array

        Data

    sigma : float

        Theoretical standard deviation

    S : jax.numpy.array

        Theoretical form of covariance matrix

    point : jax.numpy.array

        Point

    mc_sample : int

        Number of points for Monte Carlo integration

    key : int

        Seed for sampling

    Returns
    -------
    jax.numpy.array
    """
    #Sample size and number of variables
    n = data.shape[0]
    d = data.shape[1]

    #Generate keys
    keys = jax.random.split(jax.random.PRNGKey(key),mc_sample + 1)

    #Sample means (distribution of mixture that generated the point)
    mean_index = jnp.array(jax.random.randint(random.PRNGKey(keys[0,0]),(mc_sample,),0,n),dtype = jnp.uint32)

    #Sample points
    sample = lambda i: jax.random.multivariate_normal(random.PRNGKey(keys[i + 1,0]), data[mean_index[i],:], sigma*S, shape = (1,)).reshape((1,d))
    sample = jax.vmap(sample)(jnp.arange(mc_sample).reshape((mc_sample,))).reshape((mc_sample,d))

    #Compute distances
    distance = lambda point: jnp.min(mahalanobis(data = data,point = point,S = S))
    distance = jax.vmap(distance)(sample)

    return jnp.mean(distance)

#E-Step
@jit
def e_step(i,data,S,lamb):
    """
    E-step of EM algorithm for kernel estimation via the maximum pseudo-likelihood method
    -------
    Parameters
    ----------
    i : int

        Data index

    data : jax.numpy.array

        Data

    S : jax.numpy.array

        Current covariance matrix

    lamb : float

        Lambda parameter

    Returns
    -------
    jax.numpy.array
    """
    W = jax.scipy.stats.multivariate_normal.pdf(data,data[i,:],S[i,:,:]).at[i].set(-lamb)
    return W

#M step
@jit
def m_step(i,data,W):
    """
    M-step of EM algorithm for kernel estimation via the maximum pseudo-likelihood method
    -------
    Parameters
    ----------
    i : int

        Data index

    data : jax.numpy.array

        Data

    W : jax.numpy.array

        Current weights

    Returns
    -------
    jax.numpy.array
    """
    Snext = jnp.matmul(W[i,:]*jnp.transpose(data[i,:] - data),data[i,:] - data)
    return Snext

#Estimate the kernel
def kernel_estimator(data,method = "chi",S = None,S0 = None,bias = None,psi = None,mc_sample = 10000,ec = 1e-6,grid_delta = 0.01,lamb = 1,trace = False,loss = gb.quad_loss,key = 0):
    """
    Estimate the kernel for Bolstering
    -------
    Parameters
    ----------
    data : jax.numpy.array

        Data

    method : str

        Should be 'chi', 'mm', 'mpe' or 'hessian'

    S : jax.numpy.array

        Form of covariance matrix for estimation by the chi approximation method

    S0 : jax.numpy.array

        Initial covariance matrix for estimation by the maximum pseudo-likelihood method

    bias : float

        Expected bias of resubstitution estimator

    psi : function

        Estimated function for estimation via the Hessian method

    mc_sample : int

        Number of points for Monte Carlo integration

    ec : float

        Error criteria to stop EM algorithm

    grid_delta : float

        Size of grid for grid-search in the chi approximation method

    lamb : float

        Lambda parameter for estimation by the maximum pseudo-likelihood method

    trace : logical

        Whether to trace the algorithm

    loss : function

        Loss function

    key : int

        Seed for sampling

    Returns
    -------
    jax.numpy.array
    """
    #Sample size and number of variables
    n = data.shape[0]
    d = data.shape[1]

    #Compute the kernel via chi approximation
    if method == "chi":
        #Initialize S
        if S is None:
            S = jnp.diag(jnp.array([1] * d),0)

        #Compute delta_bar
        delta_bar = jnp.mean(jnp.min(dist_mahalanobis(data,S) + jnp.diag(jnp.array([jnp.inf]*n),0),1))

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
        delta_bar = jnp.mean(jnp.min(dist_mahalanobis(data,S) + jnp.diag(jnp.array([jnp.inf]*n),0),1))

        #Generate keys
        keys = jax.random.split(jax.random.PRNGKey(key),100010)

        #Compute mean distance for each sigma in a grid
        sd = jax.jit(lambda sigma,key: sample_mean_dist(data,sigma,S,mc_sample,key))
        r = 0
        mean_dist = []
        while (max(mean_dist + [0]) - delta_bar) < 0 and r < 1e5:
          r = r + 1
          sigma = r*grid_delta
          mean_dist = mean_dist + [sd(sigma,keys[r,0])]
          if trace:
              print('Sigma: ' + str(sigma) + ' Mean distance: ' + str(round(mean_dist[-1],2)) + ' Observed distance: ' + str(round(delta_bar,2)))
        sigma = (r - 1)*grid_delta

        if r == 1e5:
            print("Grid for method of moments is too small!")
            return 0

        return jnp.tile(sigma*S,(n,1,1))
    #Compute the kernel via maximum pseudolikelihood
    elif method == "mpe":
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
            W = jax.vmap(lambda i: e_step(i,data,S,lamb))(jnp.arange(n)) + lamb
            W = W/jnp.sum(W,axis = 0)

            #M step
            Snext = jax.vmap(lambda i: m_step(i,data,W))(jnp.arange(n))/(n-1)

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
    elif method == 'hessian':
        #Loss function
        def lf(xy):
            return loss(psi(xy[:,0:-1]),xy[:,-1])

        #Hessian
        H = jax.vmap(lambda x: jax.hessian(lf)(x.reshape((1,x.shape[0]))))(data).reshape((data.shape[0],data.shape[1],data.shape[1]))

        #Compute kernel
        S = 2 * bias * 1/H * (1/(data.shape[1] ** 2))

        #Nearest pd
        S = jax.vmap(nearest_pd)(S)

        return S
