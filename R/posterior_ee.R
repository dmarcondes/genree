#' @import rstanarm
#' @export
#' @title Posterior distribution generalized resubstitution error
#'
#' @description Compute the posterior distribution generalized resubstitution error in regression with posterior distribution given by Bayesian regression.
#'
#' @param psi Regression function
#' @param X X data
#' @param Y Y data
#' @param mc_sample Number of samples for Monte Carlo integration
#' @param degree Degree of the polynomial for Bayesian regression
#' @param loss A loss function
#' @param mod Object with the fitted Bayesian Regression. If not provided, the model will be fitted
#' @return The posterior probability error estimation considering the Monte Carlo mean loss under the posterior distribution
#' @references D. Marcondes, U. Braga-Neto. Generalized Resubstitution for Error Estimation in Regression. 2024.
#' @examples
#' psi_star <- function(x){(1 + sum(x))^2}
#' X <- matrix(runif(2*20),nrow = 20,ncol = 2)
#' Y <- apply(X,1,psi_star) + rnorm(20,sd = 0.25)
#' mod_pol <- lm(Y ~ .,data.frame(poly(X,2,raw = TRUE),Y))
#' psi <- function(x){predict(mod_pol,data.frame(poly(rbind(x),2,raw = TRUE)))}
#' posterior_ee(psi,X,Y,mc_sample = 100,degree = 2,mod = NULL)
posterior_ee <- function(psi,X,Y,mc_sample = 100,degree = 1,mod = NULL,loss = quad_loss){
  #Make sure they are matrices
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  #Train data
  if(degree > 1)
    data_train <- data.frame(poly(X,degree,raw = TRUE))
  else
    data_train <- data.frame(X)
  data_train$y <- Y

  #Sample size
  n <- nrow(X)

  #Fit model
  if(is.null(mod)){
    #Fit Bayesian model
    mod <- stan_glm(y ~ .,data = data_train,algorithm = "sampling")
  }

  #Predict
  error_posterior <- vector()
  for(i in 1:n)
    error_posterior[i] <- mean(loss(as.numeric(psi(matrix(X[i,],nrow = 1))),posterior_predict(mod,data_train[i,], draws = mc_sample)))

  return(mean(error_posterior))
}
