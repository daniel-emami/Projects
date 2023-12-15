# Loading libraries
library(MASS)
library(ROCR)
library(nnet) 
library(stats)
options(scipen=999)

# Loading data
huse <- read.table("realestate.txt", sep="", header=F, stringsAsFactors=F)
colnames(huse) <- c('id', 'sales price in $', 'square feet', 'number of bedrooms', 'number of bathrooms', 'air conditioning', 'garage size (number of cars)', 'pool', 'year built', 'quality (1-3)', 'style', 'lot size in square feet', 'adjacent to highway')
huse <- subset(huse,select = -1)




########## FUNCTIONS ########## 

qq_function <- function(x, name_variable){
  qq_plot <- qqnorm(x, plot.it = F)
  my_cor <- cor(qq_plot$x, qq_plot$y) # step 4
  plot(qq_plot, 
       main = paste0("QQ-plot for ", name_variable),
       ylab = "Observed quantiles",
       xlab = "Theoretical quantiles") # plot the data
  legend('topleft', paste0("r = ", round(my_cor,4))) # add the correlation value to the chart 
  return(my_cor)
}


box_cox <-function(x, name){
  boxcoxTransc <- boxcox(x~1, plotit = F)
  flagidx <- which(boxcoxTransc$y==max(boxcoxTransc$y)) 
  optlam <- boxcoxTransc$x[flagidx]
  transvec <- (x^optlam-1)/optlam
  return(transvec)
}


FindcritChi <- function(n=n0, p=p0, alpha=alpha1, N=N0){
  cricvec <- rep(0, N)  #vector for the rQ result collection#
  for(i in 1:N){
    #iteration to estimate rQ#
    numvec <- rchisq(n, p)  #generate a data set of size n, degree of freedom=p#
    d <- sort(numvec)
    q <- qchisq((1:n-0.5)/n, p)
    cricvec[i] <- cor(d,q)		
  }
  scricvec <- sort(cricvec)
  cN <- ceiling(N* alpha) #to be on the safe side I use ceiling instead of floor(), take the 'worst' alpha*N cor as rQ, everything lower than that is deemed as rejection#
  cricvalue <- scricvec[cN]
  result <- list(cN, cricvalue, scricvec)
  return(result)
}


bivar_norm <- function(x1, x2, alpha, name, remove_outlier = FALSE) {
  df <- data.frame(x1,x2) # create dataframe
  n <- nrow(df) # obersvations
  p <- ncol(df) # number of variables
  D2 <- mahalanobis(df,
                    center  = colMeans(df),
                    cov = cov(df)) # generalized squared distance
  if(remove_outlier == TRUE){
    D2 <- D2[-which.max(D2)]
  }
  chi_plot <- qqplot(qchisq(ppoints(n, a = .5), df = p), D2,
                     plot.it = F) # chi square plot values.
  # ppoints: j-1/2/n = 1:length(x)-1/2/length(x)
  my_cor <- cor(chi_plot$x, chi_plot$y) # correlation value
  critical_value <- qchisq(p = alpha,
                           df = p,
                           lower.tail = F) # calculate critical value
  prop_within_contour <- round(length(D2[D2 <= critical_value]) / length(D2),4)
  plot(chi_plot,
       ylab = 'Mahalanobis distances',
       xlab = 'Chi-square quantiles',
       main = paste0(name, ' alpha = ',alpha)) # plot chi square plot
  legend("topleft",
         paste0("r = ", round(my_cor,4), "\n",
                "% D2 <= c^2: ", prop_within_contour, "\n",
                "Expected if normal: ", 1-alpha),
         cex = 0.75,
         bty = "n") # add legend to plot
  return(my_cor)
}


chi_square_all <- function(x1,x2,x3,alpha,name, remove_outlier = FALSE){
  df <- data.frame(x1,x2,x3) # create dataframe
  n <- nrow(df) # observations
  p <- ncol(df) # number of variables
  D2 <- mahalanobis(df,
                    center  = colMeans(df),
                    cov = cov(df)) # generalized squared distance
  if(remove_outlier == TRUE){
    D2 <- D2[-which.max(D2)]
  }
  chi_plot <- qqplot(qchisq(ppoints(n, a = .5), df = p), D2,
                     plot.it = F) # chi square plot values
  my_cor <- cor(chi_plot$x, chi_plot$y) # correlation value
  critical_value <- qchisq(p = alpha,
                           df = p,
                           lower.tail = F) # calculate critical value
  prop_within_contour <- round(length(D2[D2 <= critical_value]) / length(D2),4)
  plot(chi_plot,
       ylab = 'Mahalanobis distances',
       xlab = 'Chi-square quantiles',
       main = paste0('chi square plot of D2 vs. chi_2^2(',alpha,") for ",
                     name)) # plot chi square plot
  legend("topleft",
         paste0("r = ", round(my_cor,4), "\n",
                "% D2 <= c^2: ", prop_within_contour, "\n",
                "Expected if normal: ", 1-alpha),
         cex = 1,
         bty = "n") # add legend to plot
  return(my_cor)
}




########## DATA WRANGLING ########## 

head(huse)
summary(huse)
colMeans(huse)
# pairs(huse)
cov(huse)
cor(huse)


# Searching for the houses with 0 bed rooms and 0 bathrooms
no_beds <- huse[(huse$`number of bedrooms`)==0, ] 
no_bath <- huse[(huse$`number of bathrooms`)==0, ]
outlier1_index <- as.numeric(row.names(no_beds)) # Saving the row index of the house without bath and bedrooms


# Finding multivariate outliers using Mahalanobis distance 
MD <- mahalanobis(huse,center  = colMeans(huse),cov = cov(huse))
plot(MD)
outlier2_index <- as.numeric(which.max(MD)) # Finding the row index of the house with highest MD


# Removing the two outliers
huse <- huse[c(-outlier1_index,-outlier2_index),]




########## TESTTING FOR NORMALITY ########## 



# Calculating critical value to test for marginal univariate normality 
set.seed(1) # Setting seed to ensure the same critical value
critical_value_uni <- as.numeric(FindcritChi(520,1,0.01,10000)[2])


# Isolating the continous variables
price <- huse[,1]
floor_area <- huse[,2]
lot_size <- huse[,11]

# QQ-plots without transformed variables
par(mfrow=c(2,2))
price_qq <- qq_function(price,"price")
floor_area_qq <- qq_function(floor_area,"floor_area")
lot_size_qq <- qq_function(lot_size,"lot_size")

# Testing if the correlation coefficient is higher than the critical value. If false we reject normality.
price_qq > critical_value_uni
floor_area_qq > critical_value_uni
lot_size_qq > critical_value_uni


# Box Cox-transformation
price_transformed <- box_cox(price,"price")
floor_area_transformed <- box_cox(floor_area,"floor_area")
lot_size_transformed <- box_cox(lot_size,"lot_size")

# QQ-plots with Box Cox-transformation
par(mfrow=c(2,2))
price_transformed_qq <- qq_function(price_transformed,"price")
floor_area_transformed_qq <- qq_function(floor_area_transformed,"floor_area")
lot_size_transformed_qq <- qq_function(lot_size_transformed,"lot_size")

# Testing if the correlation coefficient is higher than the critical value. If false we reject normality.
price_transformed_qq > critical_value_uni
floor_area_transformed_qq > critical_value_uni
lot_size_transformed_qq > critical_value_uni


# Calculating critical value to test for marginal univariate normality 
critical_value_bi <- as.numeric(FindcritChi(520,2,0.01,10000)[2])

# QQ-plots for bivariate
par(mfrow=c(2,2))
pr_fa_qq <- bivar_norm(price_transformed,floor_area_transformed,0.01,"price and floor area \n")
pr_ls_qq <- bivar_norm(price_transformed,lot_size_transformed,0.01,"price and lot size \n")
fa_ls_qq <- bivar_norm(floor_area_transformed,lot_size_transformed,0.01,"floor area and lot size \n")

# Testing if the correlation coefficient is higher than the critical value. If false we reject normality.
pr_fa_qq > critical_value_bi
pr_ls_qq > critical_value_bi
fa_ls_qq > critical_value_bi


# Calculating critical value to test for marginal univariate normality 
critical_value_mul <- as.numeric(FindcritChi(520,3,0.01,10000)[2])

# QQ-plot for multivariate
multi_qq <- chi_square_all(price_transformed,floor_area_transformed,lot_size_transformed,0.01,"price, floor area and lot size",remove_outlier = FALSE)

# Testing if the correlation coefficient is higher than the critical value. If false we reject normality.
multi_qq > critical_value_mul


########## TEST HOMOGENEOUS COVARIANCE MATRICES - BOX'S M-TEST ########## 

# Determining constants for M-test and correction factor
g <- 2 # Two groups. Adjacent // Not adjacent
p <- 7 # Seven variables. Price, Floor area, bedrooms, bathrooms, garage size, year built, lot size. Boolean variables are excluded.


# Splitting data into adjacent and not adjacent to highway. Boolean variables are excluded.

huse[,1] <- box_cox(price,"price")
huse[,2] <- box_cox(floor_area,"floor_area")
huse[,11] <- box_cox(lot_size,"lot_size")

not_adjacent <- huse[huse$`adjacent to highway`== 0,c(1,2,3,4,6,8,11)]
adjacent <- huse[huse$`adjacent to highway`== 1,c(1,2,3,4,6,8,11)]

s1 <- cov(not_adjacent)
s2 <- cov(adjacent)

n1 <- nrow(not_adjacent)
n2 <- nrow(adjacent)
n <- n1+n2

# Calculating S_pooled (6-49)
w <- (n1-1)*s1+(n2-1)*s2  #Within matrix#
spooled <- w*1/(n-g)

# Compute M (6-50)
M <- (n-g)*log(det(spooled))-(n1-1)*log(det(s1))-(n2-1)*log(det(s2))

# Compute correction factor (6-51)
u <- (1/(n1-1)+1/(n2-1)-1/(n-g))*(2*p^2+3*p-1)/(6*(p+1)*(g-1))

# Test statistic (6-52)
C <- (1-u)*M

# Critical value (6-53)
critvalue <- qchisq(.05,p*(p+1)*(g-1)/2, lower.tail = FALSE)   #v=p*(p+1)*(g-1)/2#


# Final decision
decisionflag <- (C > critvalue) 



########## LDA CLASSIFICATION ########## 

# Box Cox-transformation., multiplied to increase value, since lda() thinks the values are the same if they are too small
huse[,1] <- huse[,1]*100000
huse[,2] <- huse[,2]*100000
huse[,11] <- huse[,11]*100000

# LDA modelling
lda <- lda(`adjacent to highway`~.,data = huse, prior=c(0.28,0.72),cv=TRUE)
summary(lda)
huse_x <- huse[,1:11]

lda_fit <- predict(lda,huse_x) # LDA Prediction  

# Construction of confusion matrix
y <- huse$`adjacent to highway` # Actual values
yhat <- lda_fit$class # Predicted values
ConM <- table(y,yhat)
ConM

# Finding performance of the LDA fit 
pred_lda <- prediction(lda_fit[["posterior"]][,2],y)
perf_lda <- performance(pred_lda,"tpr","fpr")  # True positive rate & false positive rate

# Evaluation of performance using ROC method
par(mfrow=c(1,2))
plot(perf_lda,colorize=F,lwd=3, main="LDA") # ROC curve with TPR & FPR
AUC_lda <- performance(pred_lda, measure="auc") # Area Under Curve

auc_value_lda <- AUC_lda@y.values # Area Under Curve Value. Used to compare with the following generalized linear model.





####### ADDITIONAL CLASSIFIER FOR COMPARISON - GENERALIZED LINEAR MODEL ########

# Converting categorical variables to factors for use in glm()
huse$`air conditioning` <- as.factor(huse$`air conditioning`)
huse$pool <- as.factor(huse$pool)
huse$`adjacent to highway` <- as.factor(huse$`adjacent to highway`)
huse$`quality (1-3)` <- as.factor(huse$`quality (1-3)`)
huse$style <- as.factor(huse$style)

# Fitting the model using GLM
result <- glm(`adjacent to highway`~.,data = huse,family=binomial(link="logit")) # GLM fit using logistic regression
summary(result)


# Finding performance of the GLM fit 
p <- result$fitted.values
y <- huse$`adjacent to highway`
pred <- prediction(predictions = p, labels = y)
perf <- performance(pred,"tpr","fpr")

# Evaluation of performance using ROC method
plot(perf,colorize=F,lwd=3, main="Logistic Regression") # ROC curve with TPR & FPR


# Calculating Area Under Curve
AUC <- performance(pred, measure="auc")
auc_value_glm <- AUC@y.values 

# Cross Validation, using Leave One Out method to optimize the model
prcv <- rep(0,n) 
for (i in 1:n){
  example1 <- huse[-i,]
  res1 <- glm(example1$`adjacent to highway`~.,example1,family=binomial(link="logit")) # GLM fit using logistic regression
  xc <- huse[i,]
  lp <- predict(res1,xc)
  prcv[i] <- exp(lp)/(1+exp(lp))
}

# Confusion matrix for the model after cross validation
pr <- as.numeric( prcv >= .50) # 0.5 is the probability threshold of a real estate being adjacent or not adjacent 
table(y,pr)

# Finding performance of the cross validation 
pred1 <- prediction(prcv,y)
perf1 <- performance(pred1,"tpr","fpr")


# Calculating Area Under Curve. To be compared with the AUC values from the other models
AUCCV <- performance(pred1, measure="auc")
auc_glm_cv <- AUCCV@y.values  

# Comparing the AUC values to determine the best model
auc_value_lda # LDA
auc_value_glm # GLM, without 
auc_glm_cv # GLM Cross Validated 

# Plot of ROC curves for the different models
par(mfrow=c(1,2))
plot(perf_lda,colorize=F,lwd=3,main="LDA CV") # ROC curve for LDA
plot(perf,colorize=T,lwd=3,main="Logistic (colorized) and Logistic CV (black)") # ROC curve for initial GLM model
plot(perf1,colorize=F,lwd=3,add=T) # ROC curve for cross validated GLM