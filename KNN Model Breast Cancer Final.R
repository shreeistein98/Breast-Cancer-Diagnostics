# Load necessary libraries
library(caret)
library(FNN)
library(pROC)
library(ggplot2)

# Load the dataset
data <- read.csv("~/Desktop/ISDS 574/breast_cancer_data.csv")

# Drop the ID column and convert 'Diagnosis' to a factor
data <- data[ , !(names(data) %in% c("ID"))]
data$Diagnosis <- as.factor(data$Diagnosis)

# Normalize numeric features (all except Diagnosis)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
data[ , -1] <- as.data.frame(lapply(data[ , -1], normalize))

# Split the dataset into training and testing sets
set.seed(574)
trainIndex <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Separate features and target variable
trainX <- trainData[ , -1]
trainY <- trainData$Diagnosis
testX <- testData[ , -1]
testY <- testData$Diagnosis

# Apply Weighted KNN with k = 15
trainY_numeric <- as.numeric(trainY == levels(trainY)[2])  # 1 = Malignant, 0 = Benign
weighted_knn <- knn.reg(train = trainX, test = testX, y = trainY_numeric, k = 15)
positive_probs <- weighted_knn$pred  # Predicted probabilities for the positive class

# Define cutoffs to evaluate
cutoffs <- c(0.3, 0.4, 0.5, 0.6)

# Initialize results data frame
all_results <- data.frame(Cutoff = cutoffs, Accuracy = NA, Precision = NA, Recall = NA, F1 = NA)

# Loop through cutoffs to evaluate and visualize
for (cutoff in cutoffs) {
  # Predict labels based on cutoff
  predicted_labels <- ifelse(positive_probs >= cutoff, "M", "B")
  predicted_labels <- factor(predicted_labels, levels = c("B", "M"))  # Match testY levels
  
  # Generate confusion matrix with 'M' as the positive class
  cm <- confusionMatrix(predicted_labels, testY, positive = "M")
  print(cm)  # Print the confusion matrix summary
  
  # Extract evaluation metrics
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]  # Precision
  recall <- cm$byClass["Sensitivity"]        # Recall
  f1 <- ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)  # F1-Score
  
  # Store metrics
  all_results[all_results$Cutoff == cutoff, ] <- c(cutoff, accuracy, precision, recall, f1)
  
  # Visualize confusion matrix for this cutoff
  conf_matrix_df <- as.data.frame(as.table(cm$table))
  colnames(conf_matrix_df) <- c("Actual", "Predicted", "Frequency")
  confusion_plot <- ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Frequency), color = "black", size = 5) +
    scale_fill_gradient(low = "lightblue", high = "blue") +
    labs(title = paste("Confusion Matrix Heatmap (Cutoff =", cutoff, ")"),
         x = "Actual", y = "Predicted") +
    theme_minimal()
  print(confusion_plot)  # Print the confusion matrix heatmap
  
  # Precision-Recall Visualization for this cutoff
  pr_data <- data.frame(
    Metric = c("Precision", "Recall"),
    Value = c(precision, recall)
  )
  pr_plot <- ggplot(pr_data, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity") +
    ylim(0, 1) +
    labs(title = paste("Precision and Recall (Cutoff =", cutoff, ")"),
         x = "", y = "Value") +
    theme_minimal() +
    scale_fill_manual(values = c("lightblue", "blue"))
  print(pr_plot)  # Print the precision-recall bar chart
}

# Print all results
cat("Metrics for All Cutoffs:\n")
print(all_results)

# Plot Sensitivity vs. Cutoff
ggplot(all_results, aes(x = Cutoff, y = Recall)) +
  geom_line() +
  geom_point() +
  labs(title = "Sensitivity vs. Cutoff", x = "Cutoff", y = "Sensitivity") +
  theme_minimal()

# Plot ROC Curve (Single visualization for all thresholds)
roc_curve <- roc(testY, positive_probs, levels = rev(levels(testY)))
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Weighted KNN")
abline(a = 0, b = 1, lty = 2, col = "red")
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

# Test different values of k
accuracy_values <- c()
for(k in 1:20) {
  knn_model <- knn(train = trainX, test = testX, cl = trainY, k = k)
  conf_matrix <- table(Predicted = knn_model, Actual = testY)
  accuracy_values[k] <- sum(diag(conf_matrix)) / sum(conf_matrix)
}

# Plot accuracy against k
plot(1:20, accuracy_values, type = "b", main = "KNN Accuracy vs. k", xlab = "Number of Neighbors (k)", ylab = "Accuracy")
