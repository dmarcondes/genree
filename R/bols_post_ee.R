#' @import rstanarm
#' @export
#' @title Posterior distribution-Gaussian Bolstering error estimation
#'
#' @description Compute the posterior distribution-Gaussian bolstering error estimator in regression. The posterior distribution is given by Bayesian regression.
#'
#' @param psi Regression function
#' @param X X data
#' @param Y Y data
#' @param S List of kernels. If list size is 1, then use the same kernel for all points
#' @param mc_sample Number of samples for Monte Carlo integration
#' @param degree Degree of the polynomial for Bayesian regression
#' @param loss A loss function
#' @param mod Object with the fitted Bayesian Regression. If not provided, the respective model will be fitted
#' @return The posterior distribution-Gaussian Bolstering error estimation considering Monte Carlo integration
#' @references D. Marcondes, U. Braga-Neto. Generalized Resubstitution for Error Estimation in Regression. 2024.
#' @examples
#' psi_star <- function(x){(1 + sum(x))^2}
#' X <- matrix(runif(2*20),nrow = 20,ncol = 2)
#' Y <- apply(X,1,psi_star) + rnorm(20,sd = 0.25)
#' mod <- lm(Y ~ .,data.frame(poly(X,2,raw = TRUE),Y))
#' psi <- function(x){predict(mod,data.frame(poly(rbind(x),2,raw = TRUE)))}
#' S <- kernel_estimator(X)
#' bols_post_ee(psi,X,Y,mc_sample = 100,degree = 1,mod = NULL,S = S)
bols_post_ee <- function(psi,X,Y,S,mc_sample = 100,degree = 1,mod = NULL,loss = quad_loss){
  #Assure data is matrix
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  #Data train
  if(degree > 1)
    data_train <- data.frame(poly(X,degree,raw = TRUE))
  else
    data_train <- data.frame(X)
  data_train$y <- Y

  #Sample size and number of variables
  n <- nrow(X)
  d <- ncol(X)

  #Repeat kernel
  if(length(S) == 1)
    S <- lapply(as.list(1:n),function(i) S[[1]])

  #If posterior model is not given, then fit
  if(is.null(mod)){
    #Fit Bayesian model
    mod <- stan_glm(y ~ .,data = data_train,algorithm = "sampling")
  }

  #Calculate the posterior bolstered loss for each sample point
  Pbolstered <- vector()
  for(i in 1:n){
    #Sample point
    if(ncol(S[[i]]) == d)
      input <- X[i,]
    else
      input <- c(X[i,],Y[i])

    #Monte Carlo integration
    if(max(S[[i]]) == 0)
      Xtest_points <- rbind(input)
    else
      Xtest_points <- mvrnorm(n = mc_sample,mu = input,Sigma = S[[i]])

    Ytest_points <- posterior_predict(mod,data_train[i,], draws = mc_sample)

    error_test_points <- loss(psi(Xtest_points),Ytest_points)
    Pbolstered[i] <- mean(error_test_points)
  }

  return(mean(Pbolstered))
}
