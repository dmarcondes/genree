#' @export
#' @import MASS
#' @title Gaussian Bolstering error estimation
#'
#' @description Compute the Gaussian bolstering error estimator in regression. If the kernel dimension is d + 1 the smoothing is on both directions.
#'
#' @param psi Estimated function
#' @param X X data
#' @param Y Y data
#' @param S List of kernels. If list size is 1, then use the same kernel for all points
#' @param mc_sample Number of samples for Monte Carlo integration
#' @param loss A loss function
#' @return The Gaussian bolstering error estimator considering the Monte Carlo integration
#' @references D. Marcondes, U. Braga-Neto. Generalized Resubstitution for Error Estimation in Regression. 2024.
#' @examples
#' psi_star <- function(x){(1 + sum(x))^2}
#' X <- matrix(runif(2*20),nrow = 20,ncol = 2)
#' Y <- apply(X,1,psi_star) + rnorm(20,sd = 0.25)
#' mod <- lm(Y ~ .,data.frame(poly(X,2,raw = TRUE),Y))
#' psi <- function(x){predict(mod,data.frame(poly(rbind(x),2,raw = TRUE)))}
#' S <- kernel_estimator(cbind(X,Y),method = "mpe")
#' bolstering(psi,X,Y,S)
bolstering <- function(psi,X,Y,S,mc_sample = 100,loss = quad_loss){
  #Assure data is matrix
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  #Sample size and number of variables
  n <- nrow(X)
  d <- ncol(X)

  #Repeat kernel
  if(length(S) == 1)
    S <- lapply(as.list(1:n),function(i) S[[1]])

  #Calculate the bolstered loss for each sample point
  bolstered <- vector()
  for(i in 1:n){
    #Sample point
    if(ncol(S[[i]]) == d)
      input <- X[i,]
    else
      input <- c(X[i,],Y[i])

    #Monte Carlo integration
    if(max(S[[i]]) == 0)
      test_points <- rbind(input)
    else
      test_points <- mvrnorm(n = mc_sample,mu = input,Sigma = S[[i]])

    #Calculate error
    if(ncol(S[[i]]) == d)
      error_test_points <- loss(psi(test_points),Y[i])
    else
      error_test_points <- loss(psi(cbind(test_points[,-ncol(test_points)])),test_points[,ncol(test_points)])

    #Store mean and median error
    bolstered[i] <- mean(error_test_points)
  }

  return(mean(bolstered))
}
