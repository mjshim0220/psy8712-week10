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
                                  list = FALSE)
train <- gss_tbl[train_spilit, ]
test <- gss_tbl[-train_spilit, ]

ctrl<-trainControl(method = "cv",
                  number = 10)   

#OLS regression model
ols_model <- train(`work hours` ~ ., 
                   data = train, 
                   method = "lm",
                   trControl = ctrl,
                   preProcess="medianImpute",
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

#Publication
#CV prediction
ols_pred <- predict(ols_model, test, na.action=na.pass)
net_pred <- predict(elastic_net_model, test, na.action=na.pass)
rf_pred <- predict(rf_model, test, na.action=na.pass)
xgb_pred <- predict(xgb_model, test, na.action=na.pass)
