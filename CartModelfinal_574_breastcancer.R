rm(list = ls())
gc()

# Load libraries
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(reshape2)

# Load dataset
df <- read.csv("C:/Users/Vishnumenon/Documents/wdbc.data", header = FALSE)

# Adding column names
columns <- c("ID", "Diagnosis", 
             "RadiusMean", "TextureMean", "PerimeterMean", "AreaMean", "SmoothnessMean", 
             "CompactnessMean", "ConcavityMean", "ConcavePtMean", "SymmetryMean", 
             "FractalDimensionMean",
             "RadiusSE", "TextureSE", "PerimeterSE", "AreaSE", "SmoothnessSE", 
             "CompactSE", "ConcavitySE", "ConcavePtSE", "SymmetrySE", 
             "FractalDimensionSE",
             "WorstRadius", "WorstTexture", "WorstPerimeter", "WorstArea", "WorstSmoothness", 
             "WorstCompactness", "WorstConcavity", "WorstConcavePoints", "WorstSymmetry", 
             "WorstFractalDimension")
colnames(df) <- columns

# Convert Diagnosis to a factor
df$Diagnosis <- as.factor(df$Diagnosis)

# Split the data into training (80%) and testing (20%)
set.seed(574)
train_index <- createDataPartition(df$Diagnosis, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]
nrow(train_data)
nrow(test_data)
nrow(df)
# ===================== Full Tree =====================
fit_full <- rpart(Diagnosis ~ . - ID, 
                  data = train_data, 
                  method = "class", 
                  control = rpart.control(cp = 0, minsplit = 2))  # No pruning, grow full tree

# Visualizing the full tree
rpart.plot(fit_full, 
           main = "Full Decision Tree with Detailed Nodes", 
           extra = 101, 
           cex = 0.5, 
           tweak = 0.7, 
           box.palette = "RdBu", 
           fallen.leaves = FALSE)

# Predictions and evaluation for full tree on test data
predictions_full <- predict(fit_full, test_data, type = "class")
accuracy_full <- mean(predictions_full == test_data$Diagnosis)
error_full <- mean(predictions_full != test_data$Diagnosis)
print(paste("Accuracy of Full Tree:", accuracy_full))
print(paste("Classification Error of Full Tree:", error_full))

# Confusion matrix for full tree
conf_matrix_full <- confusionMatrix(predictions_full, test_data$Diagnosis)
print("Confusion Matrix for Full Tree:")
print(conf_matrix_full)

# ===================== Best Pruned Tree =====================
fit <- rpart(Diagnosis ~ . - ID, 
             data = train_data, 
             method = "class", 
             control = rpart.control(cp = 0.01, xval = 10))  # Cross-validation

# Complexity parameter table (best pruned)
print("Complexity Parameter Table:")
print(fit$cptable)

# Best complexity parameter (cp)
min_error_index <- which.min(fit$cptable[, "xerror"])
best_cp <- fit$cptable[min_error_index, "CP"]
print(paste("Best cp:", best_cp))

# Pruning the tree using the best cp
best_pruned_tree <- prune(fit, cp = best_cp)

# Visualizing the best pruned tree
rpart.plot(best_pruned_tree, main = "Best Pruned Tree")

# Predictions and evaluation for best pruned tree on test data
predictions_best <- predict(best_pruned_tree, test_data, type = "class")
accuracy_best <- mean(predictions_best == test_data$Diagnosis)
error_best <- mean(predictions_best != test_data$Diagnosis)
print(paste("Accuracy of Best Pruned Tree:", accuracy_best))
print(paste("Classification Error of Best Pruned Tree:", error_best))

# Confusion matrix for best pruned tree
conf_matrix_best <- confusionMatrix(predictions_best, test_data$Diagnosis)
print("Confusion Matrix for Best Pruned Tree:")
print(conf_matrix_best)

# ===================== Minimum Error Tree =====================
min_error_tree <- prune(fit, cp = best_cp)

# Visualizing the minimum error tree
rpart.plot(min_error_tree, main = "Minimum Error Decision Tree")

# Predictions and evaluation for the minimum error tree on test data
predictions_min_error <- predict(min_error_tree, test_data, type = "class")
accuracy_min_error <- mean(predictions_min_error == test_data$Diagnosis)
error_min_error <- mean(predictions_min_error != test_data$Diagnosis)
print(paste("Accuracy of Minimum Error Tree:", accuracy_min_error))
print(paste("Classification Error of Minimum Error Tree:", error_min_error))

# Confusion matrix for the minimum error tree
conf_matrix_min_error <- confusionMatrix(predictions_min_error, test_data$Diagnosis)
print("Confusion Matrix for Minimum Error Tree:")
print(conf_matrix_min_error)

# ===================== Cutoff Values Evaluation =====================
evaluate_cutoff <- function(cutoff, model, data, true_labels) {
  prob <- predict(model, data, type = "prob")[,2]
  pred_class <- as.numeric(prob > cutoff)
  true_labels_numeric <- as.numeric(true_labels) - 1
  error_rate <- mean(pred_class != true_labels_numeric)
  return(error_rate)
}

# Testing different cutoff values
cutoff_values <- seq(0.2, 0.8, by = 0.1)
error_rates <- sapply(cutoff_values, function(cutoff) {
  evaluate_cutoff(cutoff, best_pruned_tree, test_data, test_data$Diagnosis)
})

cutoff_results <- data.frame(Cutoff = cutoff_values, ErrorRate = error_rates)
print("Error rates for different cutoff values:")
print(cutoff_results)

# Visualizing error rates for different cutoff values
plot(cutoff_results$Cutoff, cutoff_results$ErrorRate, type = "b", 
     xlab = "Cutoff Value", ylab = "Error Rate", 
     main = "Error Rate vs. Cutoff Value")

# ===================== Metrics (Sensitivity, Specificity, Accuracy) =====================
calculate_metrics <- function(cutoff, model, data, true_labels) {
  prob <- predict(model, data, type = "prob")[, 2]
  pred_class <- as.numeric(prob > cutoff)
  true_labels_numeric <- as.numeric(true_labels) - 1
  TP <- sum(pred_class == 1 & true_labels_numeric == 1)
  TN <- sum(pred_class == 0 & true_labels_numeric == 0)
  FP <- sum(pred_class == 1 & true_labels_numeric == 0)
  FN <- sum(pred_class == 0 & true_labels_numeric == 1)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  return(c(Sensitivity = sensitivity, Specificity = specificity, Accuracy = accuracy))
}

cutoff_values <- seq(0, 0.9, by = 0.1)
metrics <- sapply(cutoff_values, function(cutoff) {
  calculate_metrics(cutoff, best_pruned_tree, test_data, test_data$Diagnosis)
})

metrics_df <- as.data.frame(t(metrics))
colnames(metrics_df) <- c("Sensitivity", "Specificity", "Accuracy")
metrics_df$Cutoff <- cutoff_values
print(metrics_df)

metrics_long <- melt(metrics_df, id.vars = "Cutoff")
ggplot(metrics_long, aes(x = Cutoff, y = value, color = variable)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "Sensitivity, Specificity, and Accuracy vs. Cutoff",
       x = "Cutoff Value",
       y = "Metric Value",
       color = "Metric") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "green", "red"))

