#' @export
#' @title Mahalanobis distance
#'
#' @description Mahalanobis distance matrix between sample points
#'
#' @param X X data
#' @param S Symmetric and positive definite matrix. If NULL the identity matrix is considered
#' @return A Mahalanobis distance matrix
#' @examples
#' X <- matrix(runif(2*5),nrow = 5,ncol = 2)
#' dist_mahalanobis(X,rbind(c(1,0.5),c(0.5,1)))
dist_mahalanobis <- function(X,S = NULL){
  #Assure data is matrix
  X <- as.matrix(X)

  #Sample size and number of variables
  n <- nrow(X)
  d <- ncol(X)

  #Compute the kernel
  if(is.null(S))
    S <- diag(rep(1,d))
  Sinvert <- solve(S)
  D <- matrix(nrow = n,ncol = n)
  for(i in 1:n)
    for(j in 1:n)
      D[i,j] <- sqrt(rbind(X[i,] - X[j,]) %*% Sinvert %*% cbind(X[i,] - X[j,]))

  return(D)
}
