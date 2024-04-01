#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))#error in here so I use the button on the Files tab
library(tidyverse)
library(haven)
library(caret)

#Data Import and Cleaning
gss<-read_sav("./data/GSS2016.sav")
gss_tbl<-gss %>% 
  drop_na(MOSTHRS) %>% 
  rename(`work hours`=MOSTHRS) %>% 
  select(-c(HRS1, HRS2)) %>% 
  select(where(~ mean(is.na(.)) < 0.75))
gss_tbl$`work hours`<-as.numeric(gss_tbl$`work hours`)

#Visualization
ggplot(gss_tbl, aes(x = `work hours`)) +
  geom_histogram()+
  labs(title = "Distribution of Work Hours",
                         x = "Work Hours",
                         y = "Frequency")

#Analysis
set.seed(0220)
train_spilit <- createDataPartition(gss_tbl$`work hours`, 
                                  p = 0.75, 
                                  list = FALSE) #spilt the data into 75%/25%
train <- gss_tbl[train_spilit, ] #save 75% of df for ML
test <- gss_tbl[-train_spilit, ] #save 25% of df for test 

ctrl<-trainControl(method = "cv",
                  number = 10)   # get estimates of both 10-fold CV and holdout CV

#OLS regression model
ols_model <- train(`work hours` ~ ., 
                   data = train, 
                   method = "lm",
                   trControl = ctrl,
                   preProcess="medianImpute", #median imputation
                   na.action=na.pass)

# Elastic Net model
elastic_net_model <- train(`work hours` ~ ., 
                           data = train, 
                           method = "glmnet",
                           trControl = ctrl,
                           preProcess = "medianImpute",
                           na.action = na.pass)
# Random Forest model
rf_model <- train(`work hours` ~ ., 
                  data = train, 
                  method = "ranger",
                  trControl = ctrl,
                  preProcess = "medianImpute",
                  na.action = na.pass)

#eXtreme Gradient Boosting model
xgb_model <- train(`work hours` ~ ., 
                   data = train, 
                   method = "xgbLinear",
                   trControl = ctrl,
                   preProcess = "medianImpute",
                   na.action = na.pass)

#CV prediction
ols_pred <- predict(ols_model, test, na.action=na.pass)
net_pred <- predict(elastic_net_model, test, na.action=na.pass)
rf_pred <- predict(rf_model, test, na.action=na.pass)
xgb_pred <- predict(xgb_model, test, na.action=na.pass)

# Evaluate holdout CV
ols_rmse<-cor(test$`work hours`, ols_pred)^2
net_rmse<-cor(test$`work hours`, net_pred)^2
rf_rmse<-cor(test$`work hours`, rf_pred)^2
xgb_rmse<-cor(test$`work hours`, xgb_pred)^2

#Publication
table1_tbl<-tibble(
  algo = c( "OLS regression", "Elastic net", "Random forest", "eXtreme Gradient Boosting"),
  cv_rsq= c(
    round(ols_model$results$Rsquared[1],2),
    round(elastic_net_model$results$Rsquared[1],2),
    round(rf_model$results$Rsquared[1],2),
    round(xgb_model$results$Rsquared[1],2)
  ), #format is missing....
  ho_rsq=c(
    round(ols_rmse,2),
    round(net_rmse,2),
    round(rf_rmse,2),
    round(xgb_rmse,2)
  )
)

table1_tbl


#Answer the following questions in comments in this section, at the end of your file:
##1. How did your results change between models? Why do you think this happened, specifically?
####  OLS Regression performed the least (0.08), while the other models demonstrated better performance (0.51, 0.33, and 0.53, respectively).OLS regression assumes a linear relationship between DV and IVs. This may have limitation to explain the existing dataset. Therefore, other models produce better fit.

##2. How did you results change between k-fold CV and holdout CV? Why do you think this happened, specifically?
####Except the Elastic net model, R-squred values in other models are differed. This implies Elastic net model generalizes well across different data subsets.

##3. Among the four models, which would you choose for a real-life prediction problem, and why? Are there tradeoffs? Write up to a paragraph.
####I would choose Elastic net model in this case. Elastic Net present consistent and well performance across both k-fold CV and holdout CV. However, this model has limitation dealing with the relationship which is about more nonlinear patterns compared to Random forest or XGb model.