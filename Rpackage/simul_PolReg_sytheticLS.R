#Simulation Polynomial Regression with synthetic data (least square)
#Marcondes, D. and Braga-Neto, U.; Generalized Resubstitution for Regression Error Estimation (2024)

#libraries
library(genree)
library(foreach)
#library(tidyverse)
library(doParallel)

#Seeding and paralell
set.seed(878)
cl <- makeCluster(4)
registerDoParallel(cl)

#Simulation function
#d = dimension of inputs
#sigma = standard deviation of error distribution
#n = sample size
#rep = repetitions of simulation
#B = number of bootstrap repetitions
#degree = degree of the polynomial for Bayesian regression
#docv = whether to perform cross validation
#doboot = whether to perform bootstrap
#mc_sample = sample size for Monte Carlo integration
#pgen = degree of polynomial to generate data
#pfit = degree of polynomial to fit the data
simulate_ee_polynomial <- function(d,sigma,n,rep,B = 100,degree = degree,docv = T,doboot = T,mc_sample = 1000,pgen,pfit){
  #Polynomial that generates the data
  psi_star <- function(x){(1 + rowSums(x))^pgen}

  #Fit a polynomial to data
  model <- function(X,Y,dg = pfit){
    if(dg > 1){
      dat <- data.frame(poly(X,dg,raw = TRUE),"y" = Y)
      m <- lm(y ~.,dat)
      psi <- function(x){
        predict(m,data.frame(poly(rbind(x),dg,raw = TRUE)))
      }
    }
    else{
      dat <- data.frame(X,"y" = Y)
      colnames(dat) <- c(paste("X",1:ncol(X),sep = ""),"y")
      m <- lm(y ~.,dat)
      psi <- function(x){
        x <- cbind(x)
        colnames(x) <- paste("X",1:ncol(x),sep = "")
        predict(m,data.frame(x))
      }
    }
    return(psi)
  }

  #For each repetition
  store <- foreach(r = 1:rep,.combine = "rbind",.packages = c("genree")) %do% {
    #Initialize result
    res_tab <- na.omit(data.frame("estimator" = NA,"estimate" = NA))

    #Generate data
    X <- matrix(runif(d*n),nrow = n,ncol = d)
    Y <- psi_star(X) + rnorm(n,sd = sigma)

    #Fit model
    psi <- model(X,Y)

    #Resubstitution
    sample_loss <- quad_loss(psi(X),Y)
    er <- mean(sample_loss)
    res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = "Resubstitution","estimate" = er))

    #Real error (Monte Carlo 1000 samples)
    data_test <- matrix(runif(d*1000),nrow = 1000,ncol = d)
    en <- mean((psi_star(data_test) - psi(data_test))^2) + sigma^2

    #Gaussian Bolstered on X direction
    for(method in c("chi","mm","mpe")){
      eb <- bolstering(psi = psi,X = X,Y = Y,S = kernel_estimator(X = X,method = method),mc_sample = mc_sample)
      res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = paste("Bolstering X -",method),
                                                     "estimate" = eb))
    }

    #Gaussian Bolstered on XY direction
    for(method in c("chi","mm","mpe")){
      eb <- bolstering(psi = psi,X = X,Y = Y,S = kernel_estimator(X = cbind(X,Y),method = method),mc_sample = mc_sample)
      res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = paste("Bolstering XY -",method),
                                                     "estimate" = eb))
    }

    #Bootstrap
    if(doboot){
      eboot <- bootstrap_ee(X = X,Y = Y,B = B,model = model)
      res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = "Bootstrap","estimate" = eboot))
    }

    #CV
    if(docv){
      cv <- cv_ee(X = X,Y = Y,k = 10,model = model)
      res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = "CV","estimate" = cv))
    }

    #Posterior probability gee
    for(deg in degree){
      post_baye <- posterior_ee(X = X,Y = Y,psi = psi,degree = deg,mc_sample = mc_sample)
      res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = paste("Posterior Bayes - Degree",deg),
                                                     "estimate" = post_baye))
    }


    #Bolstering + Posterior
    for(method in c("chi","mm","mpe")){
      S <- kernel_estimator(X = X,method = method)
      for(deg in degree){
        eb_post_baye <- bols_post_ee(psi = psi,degree = deg,X = X,Y = Y,S = S,
                                     mc_sample = mc_sample)
        res_tab <- rbind.data.frame(res_tab,data.frame("estimator" = paste("Bolstering-Posterior Bayes -",method,"- Degree",deg),
                                                       "estimate" = eb_post_baye))
      }
    }


    #Store
    res_tab$rep <- r
    res_tab$en <- en
    as.matrix(res_tab)
  }
  store <- data.frame(store)
  store$d <- d
  store$sigma <- sigma
  store$n <- n
  store$pgen <- pgen
  store$pfit <- pfit
  store$key <- paste(d,sigma,n,pgen,pfit,rep)
  return(store)
}

####Simulations####
#d = number of variables
#degree = polynomial degree
#sigma = standard deviation of the error term
#n = sample size
res <- NULL
t1 <- Sys.time()
t2 <- Sys.time()
for(d in c(1,2,3)){
  for(pgen in c(1,2,3)){
    for(pfit in c(1,2,3)){
      #Set data to get number of variables in model
      dat <- matrix(runif(d*2),nrow = 2,ncol = d)
      dat <- poly(dat,pfit,raw = TRUE)
      for(sigma in c(0.25,0.5)){
        for(n in c(10,20,50,100)){
          #Trace
          print(paste(d,pgen,pfit,sigma,n,t2-t1))

          #Fit if there are more sample the variables
          if(n > ncol(dat)){
            t1 <- Sys.time()
            #Simulate
            sink("tmp")
            sim <- simulate_ee_polynomial(d = d,sigma = sigma,n = n,rep = 100,degree = c(1,2,3),
                                          docv = T,doboot = T,mc_sample = 1000,pgen = pgen,pfit = pfit)
            sink()

            #Store
            if(is.null(res))
              res <- sim
            else
              res <- rbind.data.frame(res,sim)

            #Save
            saveRDS(object = res,file = "results_PolReg_syntheticLS.rds")
            t2 <- Sys.time()
          }
        }
      }
    }
  }
}
stopCluster(cl)

# res <- readRDS(file = "results_PolReg_syntheticLS.rds")
# r <- res %>% group_by(d,sigma,n,pgen,pfit,estimator) %>% summarise(mse = mean((as.numeric(estimate) - as.numeric(en))^2),
#                                                                    bias = abs(mean(as.numeric(estimate) - as.numeric(en))))
# View(r)
# View(r %>% spread(type,mse) %>% round(digits = 6))
# View(r[order(r$n,r$mse),])
# a <- r %>% group_by(d,degree,sigma,n) %>% summarise(type = type[which(mse == min(mse))])
# prop.table(table(a$n,a$type),1)
# prop.table(table(a$type))
#
# res$estimate <- as.numeric(res$estimate) - as.numeric(res$en)
# res$estimator <- factor(res$estimator,unique(r$estimator[order(r$mse,decreasing = F)]))
# ggplot(res,aes(x = estimator,y = estimate)) + theme_linedraw() + geom_boxplot() +
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + facet_wrap(~ n) +
#   geom_hline(yintercept = 0)
