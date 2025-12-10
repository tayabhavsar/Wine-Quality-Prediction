# Load Required Libraries
# ==============================================================================
library(tidyverse)    # Data manipulation and visualization 
library(caret)        # Machine learning and model training
library(corrplot)     # Correlation plots
library(gridExtra)    # Arrange multiple plots
library(scales)       # Scale functions for visualization
library(ROCR)         # ROC curves and AUC

#Data Preparation
#--------------------------------------------------------------------------
# Load Data-set#
wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
wine

colnames(wine)

#Check dimensions 
dim(wine)

#Summary
summary(wine)


#Quality Distribution
#----------------------------------------------------------------------------
quality_dist <- wine %>%
  count(quality) %>%
  mutate(percentage = n / sum(n) * 100)

print(quality_dist)

#Plot Distribution of Wine Quality Scores
quality_dist_plot <- ggplot(quality_dist, aes(x = factor(quality), y = n, fill = factor(quality))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(n, "\n(", round(percentage, 1), "%)")), 
            vjust = 1, size = 2) +
  labs(title = "Distribution of Wine Quality Scores",
       x = "Quality Score",
       y = "Count") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "RdYlGn")

print(quality_dist_plot)

#Correlation Matrix Analysis
#-----------------------------------------------------------------------------
#Correlation Matrix of Wine Data 
cor_matrix <- cor(wine)

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Wine Quality Feature Correlation Matrix",
         mar = c(0,0,2,0))

#Correlation Matrix with Quality 
cor_with_quality <- cor_matrix[, "quality"] %>%
  as.data.frame() %>%
  rename(correlation = 1) %>%
  rownames_to_column("feature") %>%
  filter(feature != "quality") %>%
  arrange(desc(abs(correlation)))

print(cor_with_quality)

cor_quality_plot <- ggplot(cor_with_quality, 
                           aes(x = reorder(feature, correlation), y = correlation, 
                               fill = correlation > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Correlations with Wine Quality",
       x = "Feature",
       y = "Correlation Coefficient") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "green"), 
                    labels = c("Negative", "Positive"),
                    name = "Direction") +
  geom_hline(yintercept = 0, linetype = "dashed")

print(cor_quality_plot)


#Box Plot Out-lier Detection
#-----------------------------------------------------------------------------
#Box Plot Out-lier Detection 
par(mfrow = c(3, 4), mar = c(4, 4, 2, 1))
for (col in colnames(wine)[1:12]) {
  boxplot(wine[[col]], 
          main = col, 
          col = "lightblue",
          outcol = "red",
          outpch = 19,
          outcex = 0.5)
}

#Box Plots By Quality
top_features <- head(cor_with_quality, 4)$feature

box_plots <- lapply(top_features, function(feat) {
  ggplot(wine, aes(x = factor(quality), 
                        y = .data[[feat]],
                        fill = factor(quality))) +
    geom_boxplot() +
    labs(title = feat, x = "Quality", y = "") +
    theme_minimal() +
    theme(legend.position = "none") +
    scale_fill_brewer(palette = "RdYlGn")
})

do.call(grid.arrange, c(box_plots, ncol = 2,
                        top = "Distribution of Top Features by Quality Score"))

#Binary Classification
#-----------------------------------------------------------------------------
#Binary Classification (Good or Bad)
wine <- wine %>%
  mutate(quality_binary = factor(ifelse(quality >= 6, "Good", "Bad"),
                                 levels = c("Bad", "Good")))
binary_dist <- wine %>%
  count(quality_binary) %>%
  mutate(percentage = n / sum(n) * 100)

cat("\n=== Binary Classification Distribution ===\n")
print(binary_dist)

#Binary Classification Plot 
binary_class_plot <- ggplot(binary_dist, aes(x = quality_binary, y = n, fill = quality_binary)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(n, "\n(", round(percentage, 1), "%)")), 
            vjust = 2, size = 2) +
  labs(title = "Binary Classification: Good vs Bad Quality",
       x = "Quality Category",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("Bad" = "#ef4444", "Good" = "#10b981")) +
  theme(legend.position = "none")

print(binary_class_plot)

#Multi-Class Classification (Low, Medium, or High)
wine <- wine %>%
  mutate(quality_multiclass = factor(
    case_when(
      quality <= 4 ~ "Low",
      quality <= 6 ~ "Medium",
      TRUE ~ "High"
    ),
    levels = c("Low", "Medium", "High")
  ))
multiclass_dist <- wine %>%
  count(quality_multiclass) %>%
  mutate(percentage = n / sum(n) * 100)

cat("\n=== Multi-class Classification Distribution ===\n")
print(multiclass_dist)

#Multi-Class Distribution Plot 
mult_class_plot <- ggplot(multiclass_dist, aes(x = quality_multiclass, y = n, 
                                  fill = quality_multiclass)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(n, "\n(", round(percentage, 1), "%)")), 
            vjust = -0.5, size = 2) +
  labs(title = "Multi-class Classification: Low/Medium/High Quality",
       x = "Quality Category",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("Low" = "#ef4444", 
                               "Medium" = "#f59e0b", 
                               "High" = "#10b981")) +
  theme(legend.position = "none")

print(mult_class_plot) 
# ==============================================================================

#Train/Test Split of Data
#-----------------------------------------------------------------------------
#Random Sample (80% training, 20% test set)
wine_sample <- sample(nrow(wine), nrow(wine)* 0.80)
wine_train <- wine[wine_sample, ]
wine_test <- wine[-wine_sample, ]

summary(wine_train)
summary(wine_test)

#Standardize Data
#----------------------------------------------------------------------------
# Standardize the predictor variables for fair coefficient comparison
# Select only the 11 physico-chemical features

feature_cols <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
                  "residual.sugar", "chlorides", "free.sulfur.dioxide",
                  "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")

# Create standardized training data
train_scaled <- wine_train
train_scaled[, feature_cols] <- scale(wine_train[, feature_cols])

# Create standardized test data (using training mean and sd)
test_scaled <- wine_test
scaling_params <- lapply(wine_train[, feature_cols], function(x) {
  list(mean = mean(x), sd = sd(x))
})

for (i in seq_along(feature_cols)) {
  col <- feature_cols[i]
  test_scaled[[col]] <- (wine_test[[col]] - scaling_params[[i]]$mean) / 
    scaling_params[[i]]$sd
}

summary(test_scaled)
summary(train_scaled)

#Logistic Regression
#----------------------------------------------------------------------------
 
#Fit Logistic Regression Model 
wine_glm_scaled <- glm(quality_binary ~ . - quality - quality_multiclass,
                       data = wine_train_scaled,
                       family = binomial(link = "logit"))

summary(wine_glm0_scaled)


#Top 3 Predictors by Coefficients
#-----------------------------------------------------------------------------

#Extract and sort coefficients 
# Calculate standardized coefficients using a loop
 coef_df$feature_sd <- NA
 coef_df$std_coefficient <- NA
 
 for(i in 1:nrow(coef_df)) {
feat <- coef_df$feature[i]
coef_df$feature_sd[i] <- sd(wine_train[[feat]], na.rm = TRUE)
coef_df$std_coefficient[i] <- coef_df$coefficient[i] * coef_df$feature_sd[i]
    }

coef_df <- coef_df %>%
  +     mutate(abs_std_coefficient = abs(std_coefficient)) %>%
  +     arrange(desc(abs_std_coefficient))

print(coef_df)

# Top 3 predictors
top_three <- head(coef_df, 3)
cat("\n=== TOP 3 CHEMICAL PREDICTORS ===\n")
print(top_three)

# Visualization Plot
# Create properly scaled visualization
ggplot(coef_df, aes(x = std_coefficient, y = reorder(feature, std_coefficient))) +
  geom_col(aes(fill = std_coefficient > 0), width = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Logistic Regression Standardized Coefficients",
       subtitle = "Feature Importance for Wine Quality Prediction",
       x = "Standardized Coefficient",
       y = "Feature") +
  scale_fill_manual(values = c("red3", "green3"),
                    labels = c("Negative Impact", "Positive Impact"),
                    name = "Effect on Quality") +
  theme_minimal() +
  theme(legend.position = "right")


# Model Predictions
# ------------------------------------------------------------------------------
# Predictions on test set (using scaled data)
test_probs <- predict(wine_glm0, newdata = test_scaled, type = "response")
test_pred <- factor(ifelse(test_probs >= 0.5, "Good", "Bad"), 
                    levels = c("Bad", "Good"))

# Predictions on training set (for reference)
train_probs <- predict(wine_glm0, newdata = train_scaled, type = "response")
train_pred <- factor(ifelse(train_probs >= 0.5, "Good", "Bad"),
                     levels = c("Bad", "Good"))

## 5.4 Model Evaluation
# ------------------------------------------------------------------------------
# Confusion matrix for test set
conf_matrix <- confusionMatrix(test_pred, wine_test$quality_binary, 
                               positive = "Good")

cat("\n=== TEST SET PERFORMANCE ===\n")
print(conf_matrix)



# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]

cat("\n=== KEY PERFORMANCE METRICS ===\n")
cat(sprintf("Accuracy:  %.3f (%.1f%%)\n", accuracy, accuracy * 100))
cat(sprintf("Precision: %.3f (%.1f%%)\n", precision, precision * 100))
cat(sprintf("Recall:    %.3f (%.1f%%)\n", recall, recall * 100))
cat(sprintf("F1-Score:  %.3f (%.1f%%)\n", f1_score, f1_score * 100))

# Confusion matrix visualization
conf_matrix_df <- as.data.frame(conf_matrix$table)

conf_matrix_plot <- ggplot(conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 10, fontface = "bold") +
  labs(title = "Confusion Matrix - Test Set",
       x = "Actual Quality",
       y = "Predicted Quality") +
  theme_minimal() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(legend.position = "none")

print(conf_matrix_plot)

#ROC Curve and AUC 
#----------------------------------------------------------------------------

#ROC CURVE (Training Set)
library(ROCR)

# === Training Set ROC Curve ===
pred_train <- prediction(train_probs, wine_train$quality_binary)
perf_train <- performance(pred_train, "tpr", "fpr")

par(mfrow = c(1,1), mar = c(5, 5, 4, 2), cex.main = 1.5, cex.lab = 1.3, cex.axis = 1.2)

plot(perf_train, colorize=TRUE, 
     main = "ROC Curve - Wine Quality (Training Set)",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     lwd = 2)  # Thicker line
abline(a = 0, b = 1, lty = 4, col = "gray", lwd = 2)

#AUC (Training Set)
auc_train <- performance(pred_train, "auc")
auc_train_value <- auc_train@y.values[[1]]
text(0.6, 0.3, paste("AUC =", round(auc_train_value, 3)), cex = 1.8)
auc_train_value

# === Test Set ROC Curve ===
pred_test <- prediction(test_probs, wine_test$quality_binary)
perf_test <- performance(pred_test, "tpr", "fpr")

plot(perf_test, colorize=TRUE, 
     main = "ROC Curve - Wine Quality (Test Set)",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate")
abline(a = 0, b = 1, lty = 2, col = "gray")

# AUC for test
auc_test <- performance(pred_test, "auc")
auc_test_value <- auc_test@y.values[[1]]
text(0.6, 0.3, paste("AUC =", round(auc_test_value, 3)), cex = 1.2)
auc_test_value

#Summary Report
#-------------------------------------------------------------------------------
cat("\n")
cat("==============================================================================\n")
cat("                    WEEK 1 COMPLETION SUMMARY\n")
cat("==============================================================================\n")
cat("\n✓ Data Loading and Initial Exploration\n")
cat(sprintf("  - Dataset contains %d samples with %d features\n", 
            nrow(wine), ncol(wine) - 3))
cat(sprintf("  - Quality scores range from %d to %d\n", 
            min(wine$quality), max(wine$quality)))
cat("\n✓ Target Variable Engineering\n")
cat("  - Binary classification created: Good (≥6) vs Bad (<6)\n")
cat(sprintf("    * Bad:  %d samples (%.1f%%)\n", 
            binary_dist$n[1], binary_dist$percentage[1]))
cat(sprintf("    * Good: %d samples (%.1f%%)\n", 
            binary_dist$n[2], binary_dist$percentage[2]))
cat("  - Multi-class categories defined: Low (3-4), Medium (5-6), High (7-8)\n")
cat(sprintf("    * Low:    %d samples (%.1f%%)\n", 
            multiclass_dist$n[1], multiclass_dist$percentage[1]))
cat(sprintf("    * Medium: %d samples (%.1f%%)\n", 
            multiclass_dist$n[2], multiclass_dist$percentage[2]))
cat(sprintf("    * High:   %d samples (%.1f%%)\n", 
            multiclass_dist$n[3], multiclass_dist$percentage[3]))
cat("\n✓ Data Splitting\n")
cat(sprintf("  - Training set: %d samples (80%%)\n", nrow(wine_train)))
cat(sprintf("  - Test set:     %d samples (20%%)\n", nrow(wine_test)))
cat("\n✓ Baseline Logistic Regression Model\n")
cat(sprintf("  - Test Accuracy:  %.1f%%\n", accuracy * 100))
cat(sprintf("  - Test Precision: %.1f%%\n", precision * 100))
cat(sprintf("  - Test Recall:    %.1f%%\n", recall * 100))
cat(sprintf("  - Test F1-Score:  %.1f%%\n", f1_score * 100))
cat(sprintf("  - AUC (Train/Test): %.3f / %.3f\n", auc_train_value, auc_test_value))
cat("\n✓ Top 3 Chemical Predictors Identified:\n")
for (i in 1:3) {
  cat(sprintf("  %d. %-25s (Coefficient: %+.4f)\n", 
              i, top_three$feature[i], top_three$coefficient[i]))
}
cat("\n✓ Visualizations Created:\n")
cat("  - Quality distribution histogram\n")
cat("  - Feature correlation matrix\n")
cat("  - Feature distributions\n")
cat("  - Boxplots by quality\n")
cat("  - Binary/multi-class distributions\n")
cat("  - Feature importance plot\n")
cat("  - Confusion matrix\n")
cat("  - ROC curve\n")
cat("\n")
cat("==============================================================================\n")
cat("                      WEEK 1 DELIVERABLES COMPLETE\n")
cat("==============================================================================\n")
cat("\nNext Steps (Week 2): Classification Tree Implementation\n")
cat("\n")

# Save processed data with engineered features
write.csv(wine, "wine_processed.csv", row.names = FALSE)

# Save model
saveRDS(wine_glm_scaled, "wine_glm0_scaled_week1.rds")

# Save top 3 predictors
write.csv(top_three, "top_3_predictors.csv", row.names = FALSE)

cat("✓ Results saved:\n")
cat("  - wine_processed.csv\n")
cat("  - wine_glm0_scaled_week1.rds\n")
cat("  - top_3_predictors.csv\n")

cat("\n=== Week 1 Analysis Complete! ===\n")
