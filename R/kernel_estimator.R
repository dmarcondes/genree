#' @export
#' @import MASS
#' @import mvtnorm
#' @title Estimate the Kernel in Gaussian Bolstering
#'
#' @description Estimate the bolster kernel via the maximum pseudolikelihood estimator (MPE) or via the Method of Moments, through Monte Carlo integration (exact) or
#' the chi approximation
#'
#' @param X X data
#' @param method Estimation method. Should be "mm" for method of moments, "chi" for chi approximation and "mpe"for maximum pseudolikelihood estimation
#' @param S Symmetric and positive definite matrix $\Sigma$ for the method of moments estimator. If NULL the identity matrix is considered
#' @param mc_sample Number of samples for Monte Carlo integration
#' @param S0 List of initial kernels for the EM algorithm of the MPE. If NULL the identity matrix is considered
#' @param ec Stop criteria of the EM algorithm
#' @param grid_delta Step of grid search for method of moments estimation
#' @param lambda Parameter of EM algorithm
#' @param trace Trace EM algorithm
#' @return A list of kernels
#' @references D. Marcondes, U. Braga-Neto. Generalized Resubstitution for Error Estimation in Regression. 2024.
#' @examples
#' X <- matrix(runif(2*100),nrow = 100,ncol = 2)
#' K <- kernel_estimator(X,method = "mpe")
kernel_estimator <- function(X,method = "chi",S = NULL,S0 = NULL,mc_sample = 100,ec = 1e-8,grid_delta = 0.001,lambda = 1,trace = T){
  #Assure data is matrix
  X <- as.matrix(X)

  #Sample size and number of variables
  n <- nrow(X)
  d <- ncol(X)

  #Compute the kernel
  if(method == "chi"){
    if(is.null(S))
      S <- diag(rep(1,d))
    sigma <- mean(apply(dist_mahalanobis(X,S) + diag(Inf,n,n),1,min))/integrate(f = function(x) sqrt(x)*dchisq(x = x,df = d),lower = 0,upper = Inf)$value

    return(list(sigma*S))
  }
  else if(method == "mm"){
    if(is.null(S))
      S <- diag(rep(1,d))
    delta_bar <- mean(apply(dist_mahalanobis(X,S) + diag(Inf,n,n),1,min))
    r <- -1
    mean_dist <- c()
    while(max(c(0,mean_dist) - delta_bar) < 0){
      r <- r + 1
      sigma <- r*grid_delta
      sample_dist <- unlist(lapply(as.list(1:mc_sample),function(i) min(sqrt(mahalanobis(x = X,center = mvrnorm(n = 1,mu = X[sample(1:n,1),],Sigma = sigma*S),cov = S)))))
      mean_dist <- c(mean_dist,mean(sample_dist))
    }
    w <- which(abs(mean_dist - delta_bar) == min(abs(mean_dist - delta_bar)))

    return(list(w*grid_delta*S))
  }
  else if(method == "mpe"){
    #Initialize
    if(is.null(S0))
      S0 <- lapply(as.list(1:n),function(i) diag(1,d))
    S <- S0

    #While criteria is not met
    if(trace)
      cat("-----------------------------------------------------\n Initializing EM algorithm\n-----------------------------------------------------\n")
    criteria <- 1
    t <- 0
    while(criteria){
      #E step
      W <- matrix(nrow = n,ncol = n)
      for(i in 1:n)
        W[i,] <- ifelse(c(1:n) != i,dmvnorm(x = X,mean = X[i,],sigma = S[[i]]),-lambda)
      W <- apply(W + lambda,2,function(x) x/sum(x))

      #M step
      Snext <- list()
      for(i in 1:n){
        Snext[[i]] <- matrix(0,d,d)
        for(j in 1:n)
          if(j != i)
            Snext[[i]] <- Snext[[i]] + W[i,j] * (cbind(X[j,] - X[i,]) %*% rbind(X[j,] - X[i,]))
        Snext[[i]] <- Snext[[i]]/(n-1)
      }
      dif <- max(unlist(lapply(1:n,function(i) max(abs(Snext[[i]] - S[[i]])))))
      if(dif < ec)
        criteria <- 0
      S <- Snext
      rm(Snext)
      t <- t + 1
      if(trace)
        cat(paste("t = ",t,"step size =",dif,"\n"))
    }
    return(S)
  }
}
