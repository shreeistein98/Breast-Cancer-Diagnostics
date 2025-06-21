rm( list=ls() ) # remove all existing objects in the environment
gc() # garbage collection

# Load necessary libraries
library(dplyr)
library(caret)
library(e1071)  # For logistic regression in caret
library(pROC)  # For ROC curve and AUC
library(ggplot2)  # For heatmap

data_file <- "C:/Users/khopkarprachi/Downloads/wdbc.data"

# Read the .data file as a data frame
data <- read.table(data_file, sep=",", header=FALSE)

# Save as CSV
write.csv(data, "wdbc_data.csv", row.names=FALSE)

colnames(data) <- c("ID", "Diagnosis", 
                    "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean", 
                    "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", 
                    "Fractal_Dimension_Mean", "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", 
                    "Smoothness_SE", "Compactness_SE", "Concavity_SE", "Concave_Points_SE", "Symmetry_SE", 
                    "Fractal_Dimension_SE", "Radius_Worst", "Texture_Worst", "Perimeter_Worst",
                    "Area_Worst", "Smoothness_Worst","Compactness_Worst", "Concavity_Worst", 
                    "Concave_Points_Worst", "Symmetry_Worst","Fractal_Dimension_Worst")

write.csv(data, "C:/Users/khopkarprachi/Downloads/breast_cancer_data.csv", row.names = FALSE)

# Preprocessing
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)  # Convert Diagnosis to binary
data <- data %>% select(-ID)  # Drop ID column

# Split data into training and testing sets
set.seed(574)
trainIndex <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Standardize features
scale_features <- function(df) {
  df_scaled <- as.data.frame(scale(df %>% select(-Diagnosis)))
  df_scaled$Diagnosis <- df$Diagnosis
  return(df_scaled)
}
trainData <- scale_features(trainData)
testData <- scale_features(testData)

# Full Logistic Regression Model
full_model <- glm(Diagnosis ~ ., family = binomial, data = trainData)

# Forward Selection
cat("\nPerforming Forward Selection...\n")
forward_model <- step(glm(Diagnosis ~ 1, family = binomial, data = trainData),
                      scope = list(lower = ~1, upper = full_model),
                      direction = "forward")

# Backward Elimination
cat("\nPerforming Backward Elimination...\n")
backward_model <- step(full_model, direction = "backward")

# Stepwise Selection
cat("\nPerforming Stepwise Selection...\n")
stepwise_model <- step(glm(Diagnosis ~ 1, family = binomial, data = trainData),
                       scope = list(lower = ~1, upper = full_model),
                       direction = "both")

# Calculate Validation Error at Different Cutoffs
calculate_validation_error <- function(model, data, cutoff) {
  probs <- predict(model, newdata = data, type = "response")
  predictions <- ifelse(probs >= cutoff, 1, 0)
  error <- mean(predictions != data$Diagnosis)
  cm <- confusionMatrix(factor(predictions), factor(data$Diagnosis), positive = "1")
  list(
    Cutoff = cutoff,
    Validation_Error = error,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Confusion_Matrix = cm$table
  )
}

# Evaluate all models at multiple cutoffs
cutoffs <- c(0.3, 0.4, 0.5, 0.6)
evaluate_all_cutoffs <- function(model, data, cutoffs) {
  lapply(cutoffs, function(cutoff) calculate_validation_error(model, data, cutoff))
}

forward_results <- evaluate_all_cutoffs(forward_model, testData, cutoffs)
backward_results <- evaluate_all_cutoffs(backward_model, testData, cutoffs)
stepwise_results <- evaluate_all_cutoffs(stepwise_model, testData, cutoffs)

# Print Metrics for Each Cutoff
print_metrics <- function(results, method) {
  cat("\nMetrics for", method, "Model:\n")
  for (res in results) {
    cat("\nCutoff:", res$Cutoff, "\n")
    print(res$Confusion_Matrix)
    cat("Validation Error:", res$Validation_Error, "\n")
    cat("Accuracy:", res$Accuracy, "\n")
    cat("Sensitivity:", res$Sensitivity, "\n")
    cat("Specificity:", res$Specificity, "\n")
  }
}

print_metrics(forward_results, "Forward Selection")
print_metrics(backward_results, "Backward Elimination")
print_metrics(stepwise_results, "Stepwise Selection")
# Plot Heatmap for Confusion Matrix
plot_confusion_matrix <- function(cm, title) {
  # Convert the confusion matrix into a data frame
  cm_df <- as.data.frame(as.table(cm))
  colnames(cm_df) <- c("Actual", "Predicted", "Freq")  # Rename columns for clarity
  
  # Plot the heatmap
  ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 5) +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(title = title, x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Plot heatmaps for all models
plot_heatmaps <- function(results, method) {
  for (i in seq_along(results)) {
    res <- results[[i]]
    cm <- res$Confusion_Matrix
    cutoff <- res$Cutoff
    title <- paste(method, "Model - Cutoff:", cutoff)
    print(plot_confusion_matrix(cm, title))
  }
}

# Heatmap Visualizations
cat("\nHeatmaps for Forward Selection Model:\n")
plot_heatmaps(forward_results, "Forward Selection")

cat("\nHeatmaps for Backward Elimination Model:\n")
plot_heatmaps(backward_results, "Backward Elimination")

cat("\nHeatmaps for Stepwise Selection Model:\n")
plot_heatmaps(stepwise_results, "Stepwise Selection")


# ROC Curve for Forward Model
probs <- predict(forward_model, newdata = testData, type = "response")
roc_obj <- roc(testData$Diagnosis, probs)
plot(roc_obj, main = "ROC Curve (Forward Model)", col = "blue", lwd = 2)
auc_value <- auc(roc_obj)
cat("\nAUC for Forward Model:", auc_value, "\n")

# ROC Curve for Backward Model
probs <- predict(backward_model, newdata = testData, type = "response")
roc_obj <- roc(testData$Diagnosis, probs)
plot(roc_obj, main = "ROC Curve (Backward Model)", col = "blue", lwd = 2)
auc_value <- auc(roc_obj)
cat("\nAUC for Backward Model:", auc_value, "\n")

# ROC Curve for Stepwise Model
probs <- predict(stepwise_model, newdata = testData, type = "response")
roc_obj <- roc(testData$Diagnosis, probs)
plot(roc_obj, main = "ROC Curve (Stepwise Model)", col = "blue", lwd = 2)
auc_value <- auc(roc_obj)
cat("\nAUC for Stepwise Model:", auc_value, "\n")
