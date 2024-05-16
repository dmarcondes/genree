#' @export
#' @title Bootstrap error estimator
#'
#' @description Compute the Bootstrap error estimator
#'
#' @param X X data
#' @param Y Y data
#' @param B Number of Bootstrap samples
#' @param model A function that fit a model and return its prediction function psi
#' @param loss A loss function
#' @return The Bootstrap error estimation
#' @examples
#' psi_star <- function(x){(1 + sum(x))^2}
#' X <- matrix(runif(2*20),nrow = 20,ncol = 2)
#' Y <- apply(X,1,psi_star) + rnorm(20,sd = 0.25)
#' model <- function(X,Y){
#'   mod_pol <- lm(Y ~ .,data.frame(poly(X,2,raw = TRUE),Y))
#'   psi <- function(x){predict(mod_pol,data.frame(poly(rbind(x),2,raw = TRUE)))}
#'   return(psi)
#' }
#' bootstrap_ee(X,Y,B = 10,model)
bootstrap_ee <- function(X,Y,B,model,loss = quad_loss){
  #Assure data is matrix
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  #Sample size and number of variables
  n <- nrow(X)
  d <- ncol(X)

  #Bootstrap samples
  error_boot <- vector()
  for(b in 1:B){
    s <- sample(x = 1:n,size = n,replace = T)
    X_sample <- as.matrix(X[s,],nrow = n,ncol = d)
    Y_sample <- Y[s]
    if(length(unique(Y_sample)) > 1){
      psi_b <- model(X_sample,Y_sample)
      if(length(unique(s)) < n)
        error_boot[b] <- mean(loss(psi_b(matrix(X[-unique(s),],nrow = n - length(unique(s)),ncol = d)),Y[-unique(s)]))
      else
        error_boot[b] <- NA
    }
  }

  return(mean(error_boot,na.rm = T))
}
