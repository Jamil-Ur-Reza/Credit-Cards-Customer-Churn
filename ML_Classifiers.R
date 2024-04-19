
setwd("C:/Users/wg1075/Downloads/TRU/dasc5420/Assignments/PROJECT DASC5420/Data")

predictions_and_performance <- function(train.data = train.data,
                                        test.data = test.data,
                                        classifier = key,
                                        model = model,
                                        response.variable = response.variable 
){
  
  # train.data = df[train_indices,]
  # test.data = df[-train_indices,]
  
  #Data Segmentation
  data.X = train.data[,-which(colnames(train.data) %in% c(response.variable))]
  data.Y = train.data[,which(colnames(train.data) %in% c(response.variable))]
  
  test.X = test.data[,-which(colnames(test.data) %in% c(response.variable))]
  test.Y = test.data[,which(colnames(test.data) %in% c(response.variable))]
  
  #predictions
  predicted_classes <- predict(model, newdata = data.X)
  predicted_classes_test <- predict(model, newdata = test.X)
  
  # Calculate accuracy
  accuracy <- mean(predicted_classes == data.Y)
  accuracy_test <- mean(predicted_classes_test == test.Y)
  
  # Confusion matrix
  conf_mat <- as.numeric(data.frame(actual = as.matrix(data.Y), predicted = as.matrix(predicted_classes)) |> 
                           dplyr::filter(actual == 1 & predicted == 1) |> 
                           dplyr::summarize(sum(actual)))
  
  conf_mat_test <- as.numeric(data.frame(actual = as.matrix(test.Y), predicted = as.matrix(predicted_classes_test)) |> 
                                dplyr::filter(actual == 1 & predicted == 1) |> 
                                dplyr::summarize(sum(actual)))
  
  # Precision
  precision <- conf_mat / sum(as.numeric(as.matrix(predicted_classes)))
  precision_test <- conf_mat_test / sum(as.numeric(as.matrix(predicted_classes_test)))
  
  # Recall (Sensitivity)
  recall <- conf_mat / sum(data.Y)
  recall_test <- conf_mat_test / sum(test.Y)
  
  #F1 Score
  f1_score <- 2 * (precision * recall) / (precision + recall)
  f1_score_test <- 2 * (precision_test * recall_test) / (precision_test + recall_test)
  
  #ROC AUC
  if(length(unique(predicted_classes_test)) > 1){
    roc.obj <- pROC::roc(as.numeric(as.matrix(data.Y)), as.numeric(as.matrix(predicted_classes)))
    roc.obj_test <- pROC::roc(as.numeric(as.matrix(test.Y)), as.numeric(as.matrix(predicted_classes_test)))
    roc.auc <- pROC::auc(roc.obj)
    roc.auc_test <- pROC::auc(roc.obj_test)
  } else {
    roc.auc <- NA
    roc.auc_test <- NA
  }
  
  performance_matrix <- NULL
  performance_matrix <- rbind(performance_matrix,data.frame(
    classifier = classifier,
    source = "train",
    accuracy = accuracy, 
    precision = precision,
    recall =  recall,
    f1_score = f1_score,
    roc.auc =  roc.auc
  ))
  
  performance_matrix <- rbind(performance_matrix,data.frame(
    classifier = classifier,
    source = "test",
    accuracy = accuracy_test, 
    precision = precision_test,
    recall =  recall_test,
    f1_score = f1_score_test,
    roc.auc =  roc.auc_test
  ))
  
  #knitr::kable(performance_matrix)
  return(list(performance_matrix = performance_matrix))
}

print_stats <- function(df){
  cat(paste0(
  'Total Rows: ',nrow(df),'\n',
  'No Attritted ', round(prop.table(table(df$Attrition_Flag))[1] * 100, 2), '% of the dataset\n',
  'Attritted ', round(prop.table(table(df$Attrition_Flag))[2] * 100, 2), '% of the dataset\n',
  'Columns names: ',paste(names(df), collapse = ','),'\n',
  "Quantity of na values in dataset: ",max(colSums(is.na(df))),'\n'))
  
  print('DF Sample Data:\n')
  print(knitr::kable(head(df[,c(1:4,(ncol(df)-3):ncol(df))])))
  
  print('DF DataType:\n')
  dplyr::glimpse(df)
  
  print('DF Summary:\n')
  print(t(as.data.frame(summarytools::descr(df))[c('Mean','Std.Dev','Min','Median','Max','N.Valid','Pct.Valid'),]))
}
colors <- c("cyan", "violet","turquoise")

df <- readxl::read_excel('data_pca_v1.xlsx')
columns_to_exclude <- paste0("PC", 25:34)
df <- df[,-which(colnames(df) %in% columns_to_exclude)]

print_stats(df)

ggplot2::ggplot(df, ggplot2::aes(x = Attrition_Flag)) +
  ggplot2::geom_bar(fill = c("blue","red")) +
  ggplot2::labs(title = 'Attrition_Flag Distributions \n (0: No Attritted || 1: Attritted)',
       x = 'Attrition_Flag') +
  ggplot2::geom_text(stat = 'count', ggplot2::aes(label = ..count..), vjust = -0.5) +
  ggplot2::theme_minimal()

scaled_amt <- matrix(scale(df$Total_Trans_Amt_12m))
scaled_cnt <- matrix(scale(df$Total_Trans_Ct_12m))
scaled_age <- matrix(scale(df$Customer_Age_Range))

df <- cbind(scaled_age, scaled_amt, scaled_cnt, df)
df$Total_Trans_Amt_12m <- NULL
df$Total_Trans_Ct_12m <-  NULL
df$Customer_Age_Range <-  NULL

print_stats(df)

X <- df |> dplyr::select(-Attrition_Flag )
y <- df$Attrition_Flag

set.seed(728009)
train_indices <- caret::createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
y_train <- as.factor(y[train_indices])
X_test <- X[-train_indices, ]
y_test <- as.factor(y[-train_indices])
train.data = df[train_indices,]
test.data = df[-train_indices,]

dplyr::glimpse(X_train)
dplyr::glimpse(y_train)

dim(X_train)
length(y_train)

dim(X_test)
length(y_test)

# Define classifiers
classifiers <- list(
  "LogisticRegression" = list(method = "glm"),
  "KNearest" = list(method = "knn"),
  "XGBoost" = list(method = "xgbTree"),
  #"SupportVectorClassifierRadial" = list(method = "svmRadial"),
  "SupportVectorClassifierLinear" = list(method = "svmLinear"),
  #"SupportVectorClassifierPoly" = list(method = "svmPoly"),
  #"DecisionTreeClassifier" = list(method = "rpart"),
  "DecisionTreeClassifier" = list(method = "rpart"),
  "NaiveBayesClassifier" = list(method = "nb")
)

performance_matrix <- NULL
# Train and evaluate classifiers with cross-validation
for (key in names(classifiers)) {
  #key <- "NaiveBayesClassifier"
  classifier <- classifiers[[key]]
  
  model <- caret::train(
    Attrition_Flag ~ ., 
    data = data.frame(X_train, Attrition_Flag = y_train), 
    method = classifier$method,
    trControl = caret::trainControl(method = "cv", number = 5),
    tuneGrid = NULL,
    preProc = NULL,
    metric = "Accuracy"
  )
  
  response.variable <- "Attrition_Flag"
  pf <- predictions_and_performance(train.data = train.data,
                                                    test.data = test.data,
                                                    classifier = key, 
                                                    model = model,
                                                    response.variable = response.variable)
  performance_matrix <- rbind(performance_matrix,pf$performance_matrix)
  
}

knitr::kable(unique(performance_matrix))

writexl::write_xlsx(unique(performance_matrix),path = 'classifier_scores_pc24.xlsx')

