#' @export
#' @title Cross-validation error estimator
#'
#' @description Compute the cross-validation error estimator
#'
#' @param X X data
#' @param Y Y data
#' @param k Number of folds
#' @param model A function that fit a model and return its prediction function psi
#' @param loss A loss function
#' @return The cross-validation error estimation
#' @examples
#' psi_star <- function(x){(1 + sum(x))^2}
#' X <- matrix(runif(2*20),nrow = 20,ncol = 2)
#' Y <- apply(X,1,psi_star) + rnorm(20,sd = 0.25)
#' model <- function(X,Y){
#'   mod_pol <- lm(Y ~ .,data.frame(poly(X,2,raw = TRUE),Y))
#'   psi <- function(x){predict(mod_pol,data.frame(poly(rbind(x),2,raw = TRUE)))}
#'   return(psi)
#' }
#' cv_ee(X,Y,k = 10,model)
cv_ee <- function(X,Y,k = 10,model,loss = quad_loss){
  #Assure data is matrix
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  #Sample size and number of variables
  n <- nrow(X)
  d <- ncol(X)

  #Shuffle sample
  s <- sample(1:n,n,replace = F)
  X <- matrix(X[s,],nrow = n,ncol = d)
  Y <- Y[s]

  #Error in each fold
  error_cv <- vector()
  size <- ceiling(n/k)
  if((k-1)*size >= n)
    size <- floor(n/k)
  for(f in 1:k){
    if(f < k)
      fold <- 1 + ((f-1)*size):(f*size - 1)
    else
      fold <- 1 + ((f-1)*size):(n - 1)
    X_train <- matrix(X[-fold,],nrow = n - length(fold),ncol = d)
    Y_train <- Y[-fold]
    X_fold <- matrix(X[fold,],nrow = length(fold),ncol = d)
    Y_fold <- Y[fold]
    psi_f <- model(X_train,Y_train)
    error_cv[f] <- mean(loss(psi_f(X_fold),Y_fold))
  }

  return(mean(error_cv))
}
