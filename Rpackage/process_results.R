#Process results of simulation
#Marcondes, D. and Braga-Neto, U.; Generalized Resubstitution for Regression Error Estimation (2024)

library(tidyverse)
library(ggplot2)
library(xtable)
library(ggpubr)
library(latex2exp)

titles <- theme(strip.text = element_text(size = 12), axis.text = element_text(size = 8,
                                                                               color = "gray30"),
                axis.title = element_text(size = 14), legend.text = element_text(size = 14),
                legend.title = element_text(size = 14), panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(), panel.border = element_blank(),
                panel.background = element_rect(fill="white",size=0.5, linetype="solid",color = "black"),
                legend.background = element_rect(fill="white",size=0.5, linetype="solid",color = "black"),plot.margin = unit(c(0,0,0,0), 'lines'),
                legend.position="bottom",legend.spacing.x = unit(0.5, 'cm'),plot.title = element_text(size = 12,hjust = 0.5,face = "bold"))

####Simulation Polynomial Regression with synthetic data (least square)####

#Process data
res <- readRDS(file = "~/Dropbox/Diego/Profissional/Code/Experiments/error_estimation/Results/results_PolReg_syntheticLS.rds")
res <- res %>% filter(!(estimator %in% c("CV","Bootstrap")))
res$estimate <- as.numeric(res$estimate)
res$en <- as.numeric(res$en)
res$degree <- factor(res$estimator,c("Posterior Bayes - Degree 1","Posterior Bayes - Degree 2","Posterior Bayes - Degree 3",
                                     "Bolstering-Posterior Bayes - chi - Degree 1","Bolstering-Posterior Bayes - chi - Degree 2","Bolstering-Posterior Bayes - chi - Degree 3",
                                     "Bolstering-Posterior Bayes - mm - Degree 1","Bolstering-Posterior Bayes - mm - Degree 2","Bolstering-Posterior Bayes - mm - Degree 3",
                                     "Bolstering-Posterior Bayes - mpe - Degree 1","Bolstering-Posterior Bayes - mpe - Degree 2","Bolstering-Posterior Bayes - mpe - Degree 3"))
res$degree <- as.numeric(as.character(plyr::mapvalues(res$degree,
                                                      c("Posterior Bayes - Degree 1","Posterior Bayes - Degree 2","Posterior Bayes - Degree 3",
                                                        "Bolstering-Posterior Bayes - chi - Degree 1","Bolstering-Posterior Bayes - chi - Degree 2","Bolstering-Posterior Bayes - chi - Degree 3",
                                                        "Bolstering-Posterior Bayes - mm - Degree 1","Bolstering-Posterior Bayes - mm - Degree 2","Bolstering-Posterior Bayes - mm - Degree 3",
                                                        "Bolstering-Posterior Bayes - mpe - Degree 1","Bolstering-Posterior Bayes - mpe - Degree 2","Bolstering-Posterior Bayes - mpe - Degree 3"),
                                    c(rep(c(1,2,3),4)))))
res <- res %>% filter(n != 10 & (is.na(degree) | pfit == degree))
res$estimator <- plyr::mapvalues(res$estimator,c("Bolstering X - chi","Bolstering X - mm","Bolstering X - mpe",
                                                 "Bolstering XY - chi","Bolstering XY - mpe","Bolstering XY - mm",
                                                 "Bolstering-Posterior Bayes - chi - Degree 1","Bolstering-Posterior Bayes - chi - Degree 2","Bolstering-Posterior Bayes - chi - Degree 3",
                                                 "Bolstering-Posterior Bayes - mm - Degree 1","Bolstering-Posterior Bayes - mm - Degree 2","Bolstering-Posterior Bayes - mm - Degree 3",
                                                 "Bolstering-Posterior Bayes - mpe - Degree 1","Bolstering-Posterior Bayes - mpe - Degree 2","Bolstering-Posterior Bayes - mpe - Degree 3",
                                                 "Posterior Bayes - Degree 1","Posterior Bayes - Degree 2","Posterior Bayes - Degree 3","Resubstitution"),
                                         c("Bolst X chi","Bolst X MM","Bolst X MPE",
                                           "Bolst XY chi","Bolst XY MPE","Bolst XY MM",
                                           "Bolst-Post chi","Bolst-Post chi","Bolst-Post chi",
                                           "Bolst-Post MM","Bolst-Post MM","Bolst-Post MM",
                                           "Bolst-Post MPE","Bolst-Post MPE","Bolst-Post MPE",
                                           "Posterior","Posterior","Posterior","Resubstitution"))
res$estimator <- factor(res$estimator,c("Resubstitution","Posterior","Bolst X MPE","Bolst XY MPE","Bolst X chi","Bolst XY chi","Bolst X MM","Bolst XY MM","Bolst-Post MPE",
                                        "Bolst-Post chi","Bolst-Post MM"))
summary(res)

#Build table with bias and rmse
res <- res %>% filter(!(estimator %in% c("Bolst XY chi","Bolst XY MM","Bolst-Post chi","Bolst-Post MM")))
res_summary <- res %>% group_by(d,sigma,n,pgen,pfit,estimator) %>% summarise(rmse = sqrt(mean((estimate - en)^2)),bias = mean((estimate - en)))
res_summary$lab <- paste(ifelse(abs(res_summary$bias) < 1,signif(res_summary$bias,2),round(res_summary$bias,2))," (",
                       ifelse(res_summary$rmse < 1,signif(res_summary$rmse,2),round(res_summary$rmse,2)),")",sep = "")
res_summary <- res_summary %>% filter(n != 10 & estimator != "Bolst X chi" & pfit <= 2) %>% droplevels()
tab_bias <- res_summary %>% select(-rmse,-lab) %>% spread(estimator,bias) %>% data.frame()
tab_rmse <- res_summary %>% select(-bias,-lab) %>% spread(estimator,rmse) %>% data.frame()
tab_lab <- res_summary %>% select(-rmse,-bias) %>% spread(estimator,lab) %>% data.frame()

tab_bias2 <- tab_bias
tab_rmse2 <- tab_rmse

for(i in 1:nrow(tab_bias))
  tab_bias2[i,-c(1:5)] <- ifelse(abs(tab_bias2[i,-c(1:5)]) < 1,signif(tab_bias2[i,-c(1:5)],2),round(tab_bias2[i,-c(1:5)],2))

for(i in 1:nrow(tab_rmse))
  tab_rmse2[i,-c(1:5)] <- ifelse(abs(tab_rmse2[i,-c(1:5)]) < 1,signif(tab_rmse2[i,-c(1:5)],2),round(tab_rmse2[i,-c(1:5)],2))

tab_best <- tab_lab[,1:5]
tab_best$bias <- NA
tab_best$rmse <- NA
for(i in 1:nrow(tab_lab)){
  tab_lab[i,abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1:5)]))] <- paste("\\textbf{",tab_lab[i,abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1:5)]))],"}",sep = "")
  tab_bias2[i,abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1:5)]))] <- paste("\\textbf{",tab_bias2[i,abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1:5)]))],"}",sep = "")
  tab_best$bias[i] <- c(1:5,levels(res_summary$estimator))[abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1:5)]))]
  tab_lab[i,tab_rmse[i,] == min(tab_rmse[i,-c(1:5)])] <- paste("\\textbf{",tab_lab[i,tab_rmse[i,] == min(tab_rmse[i,-c(1:5)])],"}",sep = "")
  tab_rmse2[i,tab_rmse[i,] == min(tab_rmse[i,-c(1:5)])] <- paste("\\textbf{",tab_rmse2[i,tab_rmse[i,] == min(tab_rmse[i,-c(1:5)])],"}",sep = "")
  tab_best$rmse[i] <- c(1:5,levels(res_summary$estimator))[tab_rmse[i,] == min(tab_rmse[i,-c(1:5)])]
}

#Compare estimators
sink("tables.tex")
cat("Bias")
for(i in 1:3)
  print(xtable(tab_bias2 %>% filter(d == i)),include.rownames = F)

cat("RMSE")
for(i in 1:3)
  print(xtable(tab_rmse2 %>% filter(d == i)),include.rownames = F)
sink()

table(apply(tab_rmse[,6:11],1,function(x) names(tab_rmse)[6:11][x == min(x)]))
nrow(tab_rmse)
table(apply(tab_bias[,6:11],1,function(x) names(tab_bias)[6:11][abs(x) == min(abs(x))]))
nrow(tab_bias)

#Plot
res_summary <- res_summary %>% filter(!(d >= 2 & estimator == "Bolst X MM"))
sink("plots.tex")
for(pd in 1:3){
  plist <- list()
  k <- 1
  for(ppgen in 1:3){
    for(psigma in c(0.25,0.5)){
      dat_plot <- res_summary %>% filter(d == pd & pgen == ppgen & n != 10 & estimator != "Bolst X chi" & pfit <= 2 & sigma == psigma) %>% droplevels()
      if(nrow(dat_plot) > 1){
        plist[[k]] <- ggplot(dat_plot,aes(x = estimator,y = bias,color = factor(pfit),group = factor(pfit))) + theme_linedraw() + titles +
          geom_hline(yintercept = 0,linetype = "dashed") +
              geom_point(position=position_dodge(width=0.9)) +
              geom_errorbar(mapping = aes(ymin = bias - rmse,ymax = bias + rmse,color = factor(pfit)),position = position_dodge()) +
              theme(axis.text.x = element_text(angle = 15, vjust = 1, hjust = 1)) +
              xlab("") + ylab("") + facet_wrap(~ n,scales = "free_y") +
          ggtitle(bquote(p[g] ~ "=" ~ .(ppgen) ~ and ~ sigma ~ "=" ~ .(psigma))) +
          scale_color_discrete(expression(p[f]))
        k <- k + 1
      }
    }
  }
  cat("\n")
  autoAnalise::salvar_plot(p = ggplot(),arquivo = paste("d",pd,".pdf",sep = ""),
                           legenda = paste("Bias \\pm RMSE for each estimator and sample size for d = ",pd,
                                           ". Each plot represents a value of $p_{g}$ and $\\sigma$ and the color refer to $p_{f}$.",sep = ""))
  cat("\n")
  pdf(paste("d",pd,".pdf",sep = ""),width = 15,height = 10)
  print(ggarrange(plotlist = plist,ncol = 2,nrow = 3,common.legend = T,legend = "bottom"))
  dev.off()
  }
sink()



# ####Simulation Polynomial Regression with synthetic data (Gaussian Process)####
# res <- readRDS(file = "~/Dropbox/Diego/Pós-doutorado/Code/error_estimation/Results/results_PolReg_syntheticGP.rds")
# res_summary <- res %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP"))) %>% group_by(d,degree,sigma,n,type) %>%
#   summarise(rmse = sqrt(mean(error^2)),bias = mean(error))
# res_summary$lab <- paste(ifelse(res_summary$bias < 1,round(res_summary$bias,6),round(res_summary$bias,3))," (",
#                          ifelse(res_summary$rmse < 1,round(res_summary$rmse,6),round(res_summary$rmse,3)),")",sep = "")
# tab_bias <- res_summary %>% select(-rmse,-lab) %>% spread(type,bias) %>% data.frame()
# tab_rmse <- res_summary %>% select(-bias,-lab) %>% spread(type,rmse) %>% data.frame()
# tab_lab <- res_summary %>% select(-rmse,-bias) %>% spread(type,lab) %>% data.frame()
# for(i in 1:nrow(tab_lab)){
#   tab_lab[i,abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1,2,3,4,6)]))] <- paste("\\textbf{",tab_lab[i,abs(tab_bias[i,]) == min(abs(tab_bias[i,-c(1,2,3,4,6)]))],"}",sep = "")
#   tab_lab[i,tab_rmse[i,] == min(tab_rmse[i,-c(1,2,3,4,6)])] <- paste("\\textit{",tab_lab[i,tab_rmse[i,] == min(tab_rmse[i,-c(1,2,3,4,6)])],"}",sep = "")
# }
# tab_lab <- tab_lab[,match(c("d", "degree", "sigma","n","er","ebMean","posterior_baye","posterior_GP","PebMean_baye","PebMean_GP",
#                             "bootstrap","cv"),colnames(tab_lab))]
# View(tab_lab)
# print(xtable(tab_lab),include.rownames = F)
#
# a25 <- res_summary %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP","cv")) & sigma == 0.25) %>% group_by(d,degree,sigma,n) %>%
#   summarise(type = type[which(rmse == min(rmse))])
# a50 <- res_summary %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP","cv")) & sigma == 0.5) %>% group_by(d,degree,sigma,n) %>%
#   summarise(type = type[which(rmse == min(rmse))])
# a <- res_summary %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP","cv"))) %>% group_by(d,degree,sigma,n) %>%
#   summarise(type_mse = type[which(rmse == min(rmse))],
#             type_bias = type[which(abs(bias) == min(abs(bias)))])
# table(a$type_mse)
# table(a$type_bias)
# prop.table(table(a25$n,a25$type),1)
# prop.table(table(a25$type))
# prop.table(table(a50$n,a50$type),1)
# prop.table(table(a50$type))
#
# #Plot
# res$type <- factor(res$type,c("er","ebMean","posterior_baye","posterior_GP","PebMean_baye","PebMean_GP",
#                               "bootstrap","cv"))
# res$type <- plyr::mapvalues(res$type,c("er","ebMean","posterior_baye","posterior_GP","PebMean_baye","PebMean_GP",
#                                        "bootstrap","cv"),
#                             c("Resubstitution","GB","Posterior BR",
#                               "Posterior GP","GB-Posterior BR",
#                               "GB-Posterior GP",
#                               "Bootstrap","CV"))
# p25 <- list()
# i25 <- 1
# p50 <- list()
# i50 <- 1
# for(i in 1:nrow(tab_bias)){
#   dat_plot <- res %>% filter(d == tab_bias$d[i] & degree == tab_bias$degree[i] & sigma == tab_bias$sigma[i] & n == tab_bias$n[i] & !is.na(type))
#   p <- ggplot(dat_plot,aes(x = factor(type),y = error,fill = factor(type))) + theme_linedraw() + titles + geom_boxplot() + #theme(legend.position = "none") +
#     scale_x_discrete(breaks = NULL) + xlab("") + ylab("") +
#     ggtitle(paste("d = ",tab_bias$d[i],", p = ",tab_bias$degree[i],", n = ",tab_bias$n[i],sep = "")) +
#     scale_y_continuous(limits = c(-quantile(abs(dat_plot$error[dat_plot$type == "Resubstitution"]),0.99),
#                                   2*quantile(abs(dat_plot$error[dat_plot$type == "Resubstitution"]),0.99))) + geom_hline(yintercept = 0,linetype = "dashed") +
#     scale_fill_discrete("")
#   if(tab_bias$sigma[i] == 0.25){
#     if(i25 == 25){
#       p25[[i25]] <- NULL
#       i25 <- i25 + 1
#     }
#     p25[[i25]] <- p
#     i25 <- i25 + 1
#   }
#   else{
#     if(i50 == 25){
#       p50[[i50]] <- NULL
#       i50 <- i50 + 1
#     }
#     p50[[i50]] <- p
#     i50 <- i50 + 1
#   }
# }
# pdf("bp_GP_sigma25.pdf",width = 10,height = 20)
# ggarrange(plotlist = p25,ncol = 3,nrow = 9,common.legend = TRUE, legend="bottom")
# dev.off()
#
# pdf("bp_GP_sigma50.pdf",width = 10,height = 15)
# ggarrange(plotlist = p25,ncol = 3,nrow = 9,common.legend = TRUE, legend="bottom")
# dev.off()
#
# ####Simulation linear regression PMBL####
# res <- readRDS(file = "~/Dropbox/Diego/Pós-doutorado/Code/error_estimation/Workspace/results_pmblr_lin.rds")
# res_summary <- res %>% filter(!(type %in% c("en","ebMedian","PebMedian","PebMedian_GP"))) %>% group_by(dataset,n,d,type) %>%
#   summarise(rmse = sqrt(mean(error^2)),bias = mean(error))
# res_summary$lab <- paste(ifelse(res_summary$bias < 1,round(res_summary$bias,6),round(res_summary$bias,3))," (",
#                          ifelse(res_summary$rmse < 1,round(res_summary$rmse,6),round(res_summary$rmse,3)),")",sep = "")
# tab_bias <- res_summary %>% select(-rmse,-lab) %>% spread(type,bias) %>% data.frame()
# tab_rmse <- res_summary %>% select(-bias,-lab) %>% spread(type,rmse) %>% data.frame()
# tab_lab <- res_summary %>% select(-rmse,-bias) %>% spread(type,lab) %>% data.frame()
# for(i in 1:nrow(tab_lab)){
#   tab_lab[i,2+which(abs(tab_bias[i,-c(1,2)]) == min(abs(tab_bias[i,-c(1,2)]),na.rm = T))] <- paste("\\textbf{",tab_lab[i,2+which(abs(tab_bias[i,-c(1,2)]) == min(abs(tab_bias[i,-c(1,2)]),na.rm = T))],"}",sep = "")
#   tab_lab[i,2+which(tab_rmse[i,-c(1,2)] == min(tab_rmse[i,-c(1,2)],na.rm = T))] <- paste("\\textit{",tab_lab[i,2+which(tab_rmse[i,-c(1,2)] == min(tab_rmse[i,-c(1,2)],na.rm = T))],"}",sep = "")
# }
# tab_lab <- tab_lab[,match(c("dataset","n","er","ebMean","PebMean","posterior"),colnames(tab_lab))]
# View(tab_lab)
# print(xtable(tab_lab),include.rownames = F)
#
# a25 <- res_summary %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP","cv")) & sigma == 0.25) %>% group_by(d,degree,sigma,n) %>%
#   summarise(type = type[which(rmse == min(rmse))])
# a50 <- res_summary %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP","cv")) & sigma == 0.5) %>% group_by(d,degree,sigma,n) %>%
#   summarise(type = type[which(rmse == min(rmse))])
# a <- res_summary %>% filter(!(type %in% c("en","ebMedian","PebMedian_baye","PebMedian_GP","cv"))) %>% group_by(d,degree,sigma,n) %>%
#   summarise(type_mse = type[which(rmse == min(rmse))],
#             type_bias = type[which(abs(bias) == min(abs(bias)))])
# table(a$type_mse)
# table(a$type_bias)
# prop.table(table(a25$n,a25$type),1)
# prop.table(table(a25$type))
# prop.table(table(a50$n,a50$type),1)
# prop.table(table(a50$type))
#
# #Plot
# res$type <- factor(res$type,c("er","ebMean","posterior_baye","posterior_GP","PebMean_baye","PebMean_GP",
#                               "bootstrap","cv"))
# res$type <- plyr::mapvalues(res$type,c("er","ebMean","posterior_baye","posterior_GP","PebMean_baye","PebMean_GP",
#                                        "bootstrap","cv"),
#                             c("Resubstitution","GB","Posterior BR",
#                               "Posterior GP","GB-Posterior BR",
#                               "GB-Posterior GP",
#                               "Bootstrap","CV"))
# p25 <- list()
# i25 <- 1
# p50 <- list()
# i50 <- 1
# for(i in 1:nrow(tab_bias)){
#   dat_plot <- res %>% filter(d == tab_bias$d[i] & degree == tab_bias$degree[i] & sigma == tab_bias$sigma[i] & n == tab_bias$n[i] & !is.na(type))
#   p <- ggplot(dat_plot,aes(x = factor(type),y = error,fill = factor(type))) + theme_linedraw() + titles + geom_boxplot() + #theme(legend.position = "none") +
#     scale_x_discrete(breaks = NULL) + xlab("") + ylab("") +
#     ggtitle(paste("d = ",tab_bias$d[i],", p = ",tab_bias$degree[i],", n = ",tab_bias$n[i],sep = "")) +
#     scale_y_continuous(limits = c(-quantile(abs(dat_plot$error[dat_plot$type == "Resubstitution"]),0.99),
#                                   2*quantile(abs(dat_plot$error[dat_plot$type == "Resubstitution"]),0.99))) + geom_hline(yintercept = 0,linetype = "dashed") +
#     scale_fill_discrete("")
#   if(tab_bias$sigma[i] == 0.25){
#     if(i25 == 25){
#       p25[[i25]] <- NULL
#       i25 <- i25 + 1
#     }
#     p25[[i25]] <- p
#     i25 <- i25 + 1
#   }
#   else{
#     if(i50 == 25){
#       p50[[i50]] <- NULL
#       i50 <- i50 + 1
#     }
#     p50[[i50]] <- p
#     i50 <- i50 + 1
#   }
# }
# pdf("bp_GP_sigma25.pdf",width = 10,height = 20)
# ggarrange(plotlist = p25,ncol = 3,nrow = 9,common.legend = TRUE, legend="bottom")
# dev.off()
#
# pdf("bp_GP_sigma50.pdf",width = 10,height = 15)
# ggarrange(plotlist = p25,ncol = 3,nrow = 9,common.legend = TRUE, legend="bottom")
# dev.off()
#
