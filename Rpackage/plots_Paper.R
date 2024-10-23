#Plots paper: Marcondes, D. and Braga-Neto, U.; Generalized Resubstitution for Regression Error Estimation (2024)
library(ggplot2)
library(latex2exp)
library(ggpubr)
library(rstanarm)
library(genree)
library(MASS)

#####Gaussian Bolstering####
#Polinomyal
set.seed(11)
x <- seq(-0.5,0.3,0.05)
y <- x^2 + rnorm(17,0,0.025)
dat <- data.frame(x,y)
dat$x2 <- dat$x^2
m <- lm(y ~ x + x2,dat)
f <- function(x){
  x^2
}
dat$y <- dat$y - f(0)
dat_ex <- data.frame(x = c(0,0),y = c(0.05,-0.05))

p1 <- ggplot(dat,aes(x = x,y = y)) + theme_minimal() + geom_point(alpha = 0.5) + stat_function(fun = function(x){f(x) - f(0)}) +
  geom_point(data = dat_ex,color = c("red","blue")) +
  annotate(geom = "segment",x = -0.5,xend = 0.5,y = 0.05,yend = 0.05,linetype = "dashed") +
  geom_hline(yintercept = dat_ex$y,linetype = "dashed",alpha = 0.5) +
  geom_ribbon(data = data.frame(x = seq(-0.3,0.3,0.01),
                                f = f(seq(-0.3,0.3,0.01)) - f(0),
                                y = - 0.05),
              aes(ymax = f,ymin = y), fill="blue", alpha=.3) +
  geom_ribbon(data = data.frame(x = seq(-0.3,0.3,0.01),
                                f = f(seq(-0.3,0.3,0.01)) - f(0),
                                y = 0.05),
              aes(ymax = f,ymin = y), fill="red", alpha=.3) +
  scale_y_continuous(breaks = c(-0.05,0,0.05),label = c(-0.05,0,0.05),limits = c(-0.075,0.1)) +
  scale_x_continuous(limits = c(-0.3,0.3))+ xlab("x") + ylab("y")
p1

fred <- function(x){ (f(x) - 0.05)^2 * dnorm(x,sd = 0.1)}
fblue <- function(x){ (f(x) + 0.05)^2 * dnorm(x,sd = 0.1)}

integrate(f = fred,lower = -Inf,upper = Inf)
integrate(f = fblue,lower = -Inf,upper = Inf)

p2 <- ggplot(data.frame(x = c(-0.3,0.3)),aes(x = x)) + theme_minimal() + stat_function(fun = fred,color = "red") +
  stat_function(fun = fblue,color = "blue") + xlab("x") + ylab(unname(TeX("$(\\psi_{n}(x) - Y_{i})^{2} \\times p_{0,0.1}(x)$"))) +
  scale_y_continuous(labels = NULL)
p2

#Ulisses
set.seed(242)
x <- seq(-1.4,1.4,0.4)
y <- cos(7*x) + rnorm(length(x),0,0.25)
ggplot(data.frame(x,y),aes(x=x,y=y)) + stat_function(fun = function(x) cos(7*x)) + geom_point()

dat <- data.frame(x,y)
dat$x2 <- dat$x^2
dat$x3 <- dat$x^3
dat$x4 <- dat$x^4
dat$x5 <- dat$x^5
dat$x6 <- dat$x^6
dat$x7 <- dat$x^7
m <- lm(y ~ x + x2 + x3 + x4 + x5 + x6 + x7,dat)
fpred <- function(x){ apply(cbind(x),1,function(x) sum(c(m$coefficients[1], m$coefficients[-1]*x^(1:7))))}

p3 <- ggplot(dat,aes(x = x,y = y)) + theme_minimal() + geom_point() + stat_function(fun = function(x) cos(7*x),color = "blue",linetype = "dashed") +
  stat_function(fun = fpred) +  xlab("x") + ylab("y") +
  annotate(geom = "segment",x = dat$x - 0.1,xend = dat$x + 0.1,y = dat$y,yend = dat$y) +
  coord_cartesian(ylim = c(-1.1,1.1),xlim = c(-1.075,1.075))
for(i in 1:nrow(dat))
  p3 <- p3 + geom_ribbon(data = data.frame("x" = seq(dat$x[i] - 0.1,dat$x[i] + 0.1,0.01),"f" = fpred(seq(dat$x[i] - 0.1,dat$x[i] + 0.1,0.01)),"y" = dat$y[i]),
              aes(ymax = f,ymin = y), fill="blue", alpha=.3)
p3

p <- ggarrange(ggarrange(plotlist = list(p1,p2),labels = c("(A)","(B)"),nrow = 1),p3,labels = c("","(C)"),ncol = 1)
pdf("example_poly_gaussian.pdf",width = 15,height = 10)
p
dev.off()

#####Bolstering Both directions####
#Polinomyal
set.seed(11)
x <- seq(-0.5,0.3,0.05)
y <- x^2 + rnorm(17,0,0.025)
dat <- data.frame(x,y)
dat$x2 <- dat$x^2
m <- lm(y ~ x + x2,dat)
f <- function(x){
  x^2#m$coefficients[1] + m$coefficients[2]*x + m$coefficients[3]*x^2
}
dat$y <- dat$y - f(0)
dat <- dat[dat$x <= 0.3 & dat$x >= -0.3 & dat$y <= 0.1 & dat$y >= -0.075,]
dat <- rbind.data.frame(dat,data.frame(x = c(0,0),y = c(0.05,-0.05),x2 = c(0,0)))
S <- kernel_estimator(X = as.matrix(dat[,1:2]),method = "mpe")


p1 <- ggplot(dat,aes(x = x,y = y)) + theme_minimal() + geom_point(alpha = 0.5) + stat_function(fun = function(x){f(x) - f(0)}) +
  geom_point(data = dat_ex,color = c("red","blue")) +
  scale_y_continuous(breaks = c(-0.05,0,0.05),label = c(-0.05,0,0.05)) +
  coord_cartesian(ylim = c(-0.075,0.1),xlim = c(-0.3,0.35)) + xlab("x") + ylab("y")
for(i in 1:(nrow(dat) - 2))
  p1 <- p1 + stat_ellipse(data = data.frame(mvrnorm(n = 1e3,mu = unlist(c(dat[i,1:2])),Sigma = S[[i]])),level = c(0.5),
                          alpha = 0.5,color = "black")
p1 <- p1 + stat_ellipse(data = data.frame(mvrnorm(n = 1e3,mu = unlist(c(dat[i+1,1:2])),Sigma = S[[i+1]])),level = c(0.5),
                        geom = "polygon",fill = "red",alpha = 0.5,color = "red")
p1 <- p1 + stat_ellipse(data = data.frame(mvrnorm(n = 1e3,mu = unlist(c(dat[i+2,1:2])),Sigma = S[[i+2]])),level = c(0.5),
                        geom = "polygon",fill = "blue",alpha = 0.5,color = "blue")
p1

#Ulisses
set.seed(242)
x <- seq(-1.4,1.4,0.4)
y <- cos(7*x) + rnorm(length(x),0,0.25)
dat <- data.frame(x,y)
dat$x2 <- dat$x^2
dat$x3 <- dat$x^3
dat$x4 <- dat$x^4
dat$x5 <- dat$x^5
dat$x6 <- dat$x^6
dat$x7 <- dat$x^7
m <- lm(y ~ x + x2 + x3 + x4 + x5 + x6 + x7,dat)
dat <- dat[-c(1,nrow(dat)),]
fpred <- function(x){ apply(cbind(x),1,function(x) sum(c(m$coefficients[1], m$coefficients[-1]*x^(1:7))))}
S <- kernel_estimator(X = dat[,1:2],method = "mpe")

p3 <- ggplot(dat,aes(x = x,y = y)) + theme_minimal() + geom_point() + stat_function(fun = function(x) cos(7*x),color = "blue",linetype = "dashed",n = 1e5) +
  stat_function(fun = fpred,n = 1e5) + xlab("x") + ylab("y") + coord_cartesian(ylim = c(-1.1,1.1),xlim = c(-1.15,1.15))
for(i in 1:nrow(dat))
  p3 <- p3 + stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = unlist(c(dat[i,1:2])),Sigma = S[[i]])),level = c(0.1),
                          alpha = 0.5,color = "black")
p3
p <- ggarrange(plotlist = list(p1,p3),labels = c("(A)","(B)"),nrow = 1)
pdf("example_poly_bothdir.pdf",width = 10,height = 5)
p
dev.off()

#####Posterior error estimator####
#Bayesian
set.seed(11)
x <- seq(-0.5,0.3,0.05)
y <- x^2 + rnorm(17,0,0.025)
dat <- data.frame(x,y)
dat$x2 <- dat$x^2
m <- lm(y ~ x + x2,dat)
model_bayes <- stan_glm(y ~ x + x2,data = dat)
summary(model_bayes)
f <- function(x){
  x^2#m$coefficients[1] + m$coefficients[2]*x + m$coefficients[3]*x^2
}
dat_ex <- data.frame(x = c(0,0.12),y = c(0.05,0.12^2 - 0.05))

p1 <- ggplot(dat,aes(x = x,y = y)) + theme_minimal() + geom_point(alpha = 0.5) + stat_function(fun = f) +
  geom_point(data = dat_ex,color = c("red","blue")) +
  annotate(geom = "segment",x = 0,xend = 0,y = 0,yend = 0.05,linetype = "dashed",color = "red") +
  annotate(geom = "segment",x = 0.12,xend = 0.12,y = 0.12^2,yend = 0.12^2 - 0.05,linetype = "dashed",color = "blue") +
  scale_y_continuous(breaks = c(-0.05,0,0.05),label = c(-0.05,0,0.05),limits = c(-0.075,0.1)) +
  scale_x_continuous(limits = c(-0.3,0.3))+ xlab("X") + ylab("Y")
p1

ytilde <- na.omit(data.frame(X1 = NA,X2 = NA))
for(i in 1:100)
  ytilde <- rbind.data.frame(ytilde,data.frame(posterior_predict(model_bayes, data.frame(x = c(0,0.12),x2 = c(0,0.12^2)), draws = 4000)))

p2 <- ggplot(ytilde) + theme_minimal() + geom_density(aes(x = X1),color = "red") + geom_density(aes(x = X2),color = "blue") +
  geom_vline(xintercept = c(0,0.12^2),linetype = "dashed",color = c("red","blue")) +
  ylab(unname(TeX("$\\hat{P}_{n}(Y = y|X)$"))) + scale_y_continuous(labels = NULL) + scale_x_continuous(limits = c(-0.1,0.1)) + xlab("Y")

mean((f(0) - ytilde$X1)^2)
mean((f(0.12) - ytilde$X2)^2)

p <- ggarrange(plotlist = list(p1,p2),labels = c("(A)","(B)"))
pdf("example_poly_posterior.pdf",width = 10,height = 5)
p
dev.off()

#####Kernel Estimation####
#X direction
set.seed(32842)
n <- 10
X <- matrix(runif(2*n),nrow = n,ncol = 2)
Schi <- kernel_estimator(X = X,method = "chi",mc_sample = 1e4)
Smm <- kernel_estimator(X = X,method = "mm",mc_sample = 1e4)
Smpe <- kernel_estimator(X = X,method = "mpe",mc_sample = 1e4)

p1 <- ggplot(data.frame(X),aes(x = X1,y = X2)) + theme_minimal() + geom_point(alpha = 0.5,size = 0.5) +
  xlab(unname(TeX("$x_{1}$"))) + ylab(unname(TeX("$x_{2}$"))) +
  scale_x_continuous(breaks = scales::pretty_breaks(5)) +
  scale_y_continuous(breaks = scales::pretty_breaks(5))
for(i in 1:nrow(X)){
  p1 <- p1 +  stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = X[i,],Sigma = Smpe[[i]])),level = c(0.05),
                         alpha = 1,color = "black")
  p1 <- p1 +  stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = X[i,],Sigma = Schi[[1]])),level = c(0.05),
                         alpha = 1,color = "black",linetype = "dotted")
  p1 <- p1 +  stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = X[i,],Sigma = Smm[[1]])),level = c(0.05),
                         alpha = 1,color = "black",linetype = "dashed")
}
p1

#Both directions
set.seed(2842)
n <- 10
X <- matrix(runif(n),nrow = n,ncol = 1)
X <- cbind(X,X + rnorm(n,sd = 0.1))
Schi <- kernel_estimator(X = X,method = "chi",mc_sample = 1e4)
Smm <- kernel_estimator(X = X,method = "mm",mc_sample = 1e4)
Smpe <- kernel_estimator(X = X,method = "mpe",mc_sample = 1e4)
m <- lm(X[,2] ~ X[,1])
fpred <- function(x){m$coefficients[0] + m$coefficients[1]*x}

p2 <- ggplot(data.frame(X),aes(x = X1,y = X2)) + theme_minimal() + geom_point(alpha = 0.5,size = 0.5) +
  xlab(unname(TeX("$x$"))) + ylab(unname(TeX("$y$"))) +
  scale_x_continuous(breaks = scales::pretty_breaks(5)) +
  scale_y_continuous(breaks = scales::pretty_breaks(5)) +
  stat_function(fun = fpred,alpha = 0.5)
for(i in 1:nrow(X)){
  p2 <- p2 +  stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = X[i,],Sigma = Smpe[[i]])),level = c(0.05),
                           alpha = 1,color = "black")
  p2 <- p2 +  stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = X[i,],Sigma = Schi[[1]])),level = c(0.05),
                           alpha = 1,color = "black",linetype = "dotted")
  p2 <- p2 +  stat_ellipse(data = data.frame(mvrnorm(n = 1e5,mu = X[i,],Sigma = Smm[[1]])),level = c(0.05),
                           alpha = 1,color = "black",linetype = "dashed")
}
p2

p <- ggarrange(plotlist = list(p1,p2),labels = c("(A)","(B)"))
pdf("example_kernel_estimation.pdf",width = 10,height = 5)
p
dev.off()

#####Grid Search mm####
set.seed(32842)
n <- 10
X <- matrix(runif(2*n),nrow = n,ncol = 2)
Smm <- kernel_estimator(X = X,method = "mm",mc_sample = 1e3)
kernel_estimator(X = X,method = "chi",mc_sample = 1e3)

p1 <- ggplot(data.frame(X),aes(x = X1,y = X2)) + theme_minimal() + geom_point(alpha = 0.5) +
  xlab(unname(TeX("$x_{1}$"))) + ylab(unname(TeX("$x_{2}$"))) +
  scale_x_continuous(breaks = scales::pretty_breaks(5)) +
  scale_y_continuous(breaks = scales::pretty_breaks(5))

p1

n <- nrow(X)
d <- ncol(X)
mc_sample <- 1e3
grid_delta <- 0.005

#Compute the kernel
S <- diag(rep(1,d))
delta_bar <- mean(apply(dist_mahalanobis(X,S) + diag(Inf,n,n),1,min))
r <- -1
mean_dist <- c()
while(r < 20){
  print(r)
  r <- r + 1
  sigma <- r*grid_delta
  sample_dist <- unlist(lapply(as.list(1:mc_sample),function(i) min(sqrt(mahalanobis(x = X,center = mvrnorm(n = 1,mu = X[sample(1:n,1),],Sigma = sigma*S),cov = S)))))
  mean_dist <- c(mean_dist,mean(sample_dist))
}
w <- which(abs(mean_dist - delta_bar) == min(abs(mean_dist - delta_bar)))
sigma <- w*grid_delta



p2 <- ggplot(data.frame(sigma = sqrt((1:(r+1))*grid_delta),exp = mean_dist),aes(x = sigma,y = exp)) + theme_minimal() + geom_point() +
  geom_hline(yintercept = delta_bar,linetype = "dashed") + geom_vline(xintercept = sqrt(sigma)) +
  scale_x_continuous(breaks = scales::pretty_breaks(5)) + scale_y_continuous(breaks = scales::pretty_breaks(5)) +
  xlab(unname(TeX("$\\sigma_{S_{n}}$"))) + ylab(unname(TeX("$E[\\delta(\\hat{X})]$")))
p2

p <- ggarrange(plotlist = list(p1,p2),labels = c("(A)","(B)"))
pdf("example_grid_search.pdf",width = 10,height = 5)
p
dev.off()
