#' @export
#' @title Loss functions
#'
#' @description Common loss functions for learning problems. \emph{quad_loss} for quadratic loss,
#' \emph{simple_loss} for the simple loss function and \emp{mae_loss} for the mean absolute error (MAE) loss.
#'
#' @param obs Vector with observed data
#' @param pred Vector with predicted data
#' @return A vector of losses.
quad_loss <- function(obs,pred){
  return((obs - pred)^2)
}

#' @export
simple_loss <- function(obs,pred){
  return(as.numeric(obs != pred))
}

#' @export
mae_loss <- function(obs,pred){
  return(abs(obs - pred))
}

