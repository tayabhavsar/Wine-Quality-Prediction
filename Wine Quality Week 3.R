# Load Required Libraries
library(randomForest)
library(gbm)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(pROC)

# Set seed for reproducibility
set.seed(123)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

setwdsetwd("~/BUS235C/Project/WineQualityProject/Week 3")
# Use existing wine_processed data and pre-split train/test sets
wine_data <- wine_processed
train_data <- wine_train
test_data <- wine_test

# Check structure
cat("\n=== Data Structure ===\n")
str(wine_data)
summary(wine_data)

# Ensure quality_binary is a factor with correct levels
wine_data$quality_binary <- factor(wine_data$quality_binary, levels = c("Bad", "Good"))
train_data$quality_binary <- factor(train_data$quality_binary, levels = c("Bad", "Good"))
test_data$quality_binary <- factor(test_data$quality_binary, levels = c("Bad", "Good"))

# Create multi-class quality variable if not exists
if(!"quality_class" %in% colnames(wine_data)) {
  wine_data$quality_class <- factor(wine_data$quality_multiclass, 
                                    levels = c("Low", "Medium", "High"))
  train_data$quality_class <- factor(train_data$quality_multiclass,
                                     levels = c("Low", "Medium", "High"))
  test_data$quality_class <- factor(test_data$quality_multiclass,
                                    levels = c("Low", "Medium", "High"))
}

# Check class distribution
cat("\n=== Class Distribution ===\n")
cat("Training set:\n")
print(table(train_data$quality_binary))
print(prop.table(table(train_data$quality_binary)))

cat("\nTest set:\n")
print(table(test_data$quality_binary))
print(prop.table(table(test_data$quality_binary)))

cat("\nTraining set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# ============================================================================
# 2. RANDOM FOREST MODEL
# ============================================================================

cat("\n=== RANDOM FOREST IMPLEMENTATION ===\n")

# Prepare features and target
features <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
              "residual.sugar", "chlorides", "free.sulfur.dioxide",
              "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")

# --- 2.1: Basic Random Forest ---
rf_model_basic <- randomForest(
  x = wine_train[, features],
  y = wine_train$quality_binary,
  ntree = 500,
  mtry = 3,  # sqrt(11) ≈ 3 for classification
  importance = TRUE,
  proximity = FALSE
)

print(rf_model_basic)

# Plot error rates
plot(rf_model_basic, main = "Random Forest Error Rate by Trees")
legend("topright", legend = colnames(rf_model_basic$err.rate),
       col = 1:3, lty = 1:3, cex = 0.8)

# --- 2.2: Hyperparameter Tuning for Random Forest ---
cat("\nTuning Random Forest hyperparameters...\n")

# Grid search for mtry and ntree
tuning_grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6),
  ntree = c(300, 500, 700),
  min_nodesize = c(1, 5, 10)
)

rf_results <- data.frame()

for(i in 1:nrow(tuning_grid)) {
  temp_rf <- randomForest(
    x = wine_train[, features],
    y = wine_train$quality_binary,
    ntree = tuning_grid$ntree[i],
    mtry = tuning_grid$mtry[i],
    nodesize = tuning_grid$min_nodesize[i],
    importance = TRUE
  )
  
  # Get OOB error
  oob_error <- temp_rf$err.rate[tuning_grid$ntree[i], "OOB"]
  
  rf_results <- rbind(rf_results, data.frame(
    mtry = tuning_grid$mtry[i],
    ntree = tuning_grid$ntree[i],
    nodesize = tuning_grid$min_nodesize[i],
    oob_error = oob_error
  ))
  
  if(i %% 5 == 0) cat("Completed", i, "of", nrow(tuning_grid), "combinations\n")
}

# Find best parameters
best_rf_params <- rf_results[which.min(rf_results$oob_error), ]
cat("\nBest Random Forest Parameters:\n")
print(best_rf_params)

# --- 2.3: Train Final Random Forest Model ---
rf_model_final <- randomForest(
  x = wine_train[, features],
  y = wine_train$quality_binary,
  ntree = best_rf_params$ntree,
  mtry = best_rf_params$mtry,
  nodesize = best_rf_params$nodesize,
  importance = TRUE
)

# Feature Importance
rf_importance <- importance(rf_model_final)
rf_importance_df <- data.frame(
  Feature = rownames(rf_importance),
  MeanDecreaseAccuracy = rf_importance[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = rf_importance[, "MeanDecreaseGini"]
)
rf_importance_df <- rf_importance_df[order(-rf_importance_df$MeanDecreaseAccuracy), ]

cat("\nRandom Forest Feature Importance (Top 5):\n")
print(head(rf_importance_df, 5))

# Plot feature importance
par(mfrow = c(1, 2))
varImpPlot(rf_model_final, main = "Random Forest Variable Importance")
par(mfrow = c(1, 1))

# --- 2.4: Random Forest Predictions ---
rf_pred_train <- predict(rf_model_final, wine_train[, features], type = "response")
rf_pred_test <- predict(rf_model_final, wine_test[, features], type = "response")

rf_pred_prob_train <- predict(rf_model_final, wine_train[, features], type = "prob")[, "Good"]
rf_pred_prob_test <- predict(rf_model_final, wine_test[, features], type = "prob")[, "Good"]

# Confusion Matrix
rf_cm_train <- confusionMatrix(rf_pred_train, wine_train$quality_binary, positive = "Good")
rf_cm_test <- confusionMatrix(rf_pred_test, wine_test$quality_binary, positive = "Good")

cat("\n--- Random Forest Performance ---\n")
cat("Training Accuracy:", round(rf_cm_train$overall["Accuracy"], 4), "\n")
cat("Test Accuracy:", round(rf_cm_test$overall["Accuracy"], 4), "\n")
cat("Test Precision:", round(rf_cm_test$byClass["Precision"], 4), "\n")
cat("Test Recall:", round(rf_cm_test$byClass["Recall"], 4), "\n")
cat("Test F1-Score:", round(rf_cm_test$byClass["F1"], 4), "\n")

# ROC Curve
rf_roc <- roc(wine_test$quality_binary, rf_pred_prob_test)
cat("Test AUC:", round(auc(rf_roc), 4), "\n")

# ============================================================================
# 3. GRADIENT BOOSTING MODEL
# ============================================================================

cat("\n=== GRADIENT BOOSTING IMPLEMENTATION ===\n")

# Convert quality_binary to numeric (0 = Bad, 1 = Good)
wine_train$quality_numeric <- as.numeric(wine_train$quality_binary) - 1
wine_test$quality_numeric <- as.numeric(wine_test$quality_binary) - 1

# --- 3.1: Basic Gradient Boosting Model ---
gbm_model_basic <- gbm(
  quality_numeric ~ fixed.acidity + volatile.acidity + citric.acid + 
    residual.sugar + chlorides + free.sulfur.dioxide + 
    total.sulfur.dioxide + density + pH + sulphates + alcohol,
  data = wine_train,
  distribution = "bernoulli",
  n.trees = 1000,
  interaction.depth = 4,
  shrinkage = 0.01,
  bag.fraction = 0.5,
  train.fraction = 0.8,
  cv.folds = 5,
  verbose = FALSE
)

# Find optimal number of trees
best_iter <- gbm.perf(gbm_model_basic, method = "cv", plot.it = TRUE)
cat("Optimal number of trees:", best_iter, "\n")

# --- 3.2: Hyperparameter Tuning for GBM ---
cat("\nTuning GBM hyperparameters...\n")

gbm_grid <- expand.grid(
  n.trees = c(500, 1000, 1500),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(5, 10, 15)
)

gbm_results <- data.frame()

for(i in 1:nrow(gbm_grid)) {
  temp_gbm <- gbm(
    quality_numeric ~ fixed.acidity + volatile.acidity + citric.acid + 
      residual.sugar + chlorides + free.sulfur.dioxide + 
      total.sulfur.dioxide + density + pH + sulphates + alcohol,
    data = wine_train,
    distribution = "bernoulli",
    n.trees = gbm_grid$n.trees[i],
    interaction.depth = gbm_grid$interaction.depth[i],
    shrinkage = gbm_grid$shrinkage[i],
    n.minobsinnode = gbm_grid$n.minobsinnode[i],
    bag.fraction = 0.5,
    cv.folds = 5,
    verbose = FALSE
  )
  
  best_trees <- gbm.perf(temp_gbm, method = "cv", plot.it = FALSE)
  
  # Predict on validation
  pred_prob <- predict(temp_gbm, newdata = wine_test, 
                       n.trees = best_trees, type = "response")
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  accuracy <- mean(pred_class == wine_test$quality_numeric)
  
  gbm_results <- rbind(gbm_results, data.frame(
    n.trees = gbm_grid$n.trees[i],
    interaction.depth = gbm_grid$interaction.depth[i],
    shrinkage = gbm_grid$shrinkage[i],
    n.minobsinnode = gbm_grid$n.minobsinnode[i],
    best_trees = best_trees,
    accuracy = accuracy
  ))
  
  if(i %% 5 == 0) cat("Completed", i, "of", nrow(gbm_grid), "combinations\n")
}

# Find best parameters
best_gbm_params <- gbm_results[which.max(gbm_results$accuracy), ]
cat("\nBest GBM Parameters:\n")
print(best_gbm_params)

# --- 3.3: Train Final GBM Model ---
gbm_model_final <- gbm(
  quality_numeric ~ fixed.acidity + volatile.acidity + citric.acid + 
    residual.sugar + chlorides + free.sulfur.dioxide + 
    total.sulfur.dioxide + density + pH + sulphates + alcohol,
  data = wine_train,
  distribution = "bernoulli",
  n.trees = best_gbm_params$n.trees,
  interaction.depth = best_gbm_params$interaction.depth,
  shrinkage = best_gbm_params$shrinkage,
  n.minobsinnode = best_gbm_params$n.minobsinnode,
  bag.fraction = 0.5,
  cv.folds = 5,
  verbose = FALSE
)

# Get optimal trees
gbm_best_iter <- gbm.perf(gbm_model_final, method = "cv", plot.it = TRUE)

# Feature Importance
gbm_importance <- summary(gbm_model_final, n.trees = gbm_best_iter, 
                          plotit = TRUE, las = 2)

cat("\nGBM Feature Importance (Top 5):\n")
print(head(gbm_importance, 5))

# --- 3.4: GBM Predictions ---
gbm_pred_prob_train <- predict(gbm_model_final, newdata = wine_train,
                               n.trees = gbm_best_iter, type = "response")
gbm_pred_prob_test <- predict(gbm_model_final, newdata = wine_test,
                              n.trees = gbm_best_iter, type = "response")

gbm_pred_train <- factor(ifelse(gbm_pred_prob_train > 0.5, "Good", "Bad"),
                         levels = c("Bad", "Good"))
gbm_pred_test <- factor(ifelse(gbm_pred_prob_test > 0.5, "Good", "Bad"),
                        levels = c("Bad", "Good"))

# Confusion Matrix
gbm_cm_train <- confusionMatrix(gbm_pred_train, wine_train$quality_binary, positive = "Good")
gbm_cm_test <- confusionMatrix(gbm_pred_test, wine_test$quality_binary, positive = "Good")

cat("\n--- GBM Performance ---\n")
cat("Training Accuracy:", round(gbm_cm_train$overall["Accuracy"], 4), "\n")
cat("Test Accuracy:", round(gbm_cm_test$overall["Accuracy"], 4), "\n")
cat("Test Precision:", round(gbm_cm_test$byClass["Precision"], 4), "\n")
cat("Test Recall:", round(gbm_cm_test$byClass["Recall"], 4), "\n")
cat("Test F1-Score:", round(gbm_cm_test$byClass["F1"], 4), "\n")

# ROC Curve
gbm_roc <- roc(wine_test$quality_binary, gbm_pred_prob_test)
cat("Test AUC:", round(auc(gbm_roc), 4), "\n")


# ============================================================================
# 4. RESEARCH QUESTION 3: Alcohol Threshold Analysis
# ============================================================================

cat("\n=== RESEARCH QUESTION 3: Alcohol Threshold ===\n")

# Get median values for all features except alcohol
feature_medians <- wine_train %>%
  select(all_of(features)) %>%
  select(-alcohol) %>%
  summarise(across(everything(), median))

# Create sequence of alcohol values
alcohol_range <- seq(min(wine_data$alcohol), max(wine_data$alcohol), by = 0.1)

# Create prediction dataset
prediction_data <- data.frame(
  fixed.acidity = rep(feature_medians$fixed.acidity, length(alcohol_range)),
  volatile.acidity = rep(feature_medians$volatile.acidity, length(alcohol_range)),
  citric.acid = rep(feature_medians$citric.acid, length(alcohol_range)),
  residual.sugar = rep(feature_medians$residual.sugar, length(alcohol_range)),
  chlorides = rep(feature_medians$chlorides, length(alcohol_range)),
  free.sulfur.dioxide = rep(feature_medians$free.sulfur.dioxide, length(alcohol_range)),
  total.sulfur.dioxide = rep(feature_medians$total.sulfur.dioxide, length(alcohol_range)),
  density = rep(feature_medians$density, length(alcohol_range)),
  pH = rep(feature_medians$pH, length(alcohol_range)),
  sulphates = rep(feature_medians$sulphates, length(alcohol_range)),
  alcohol = alcohol_range
)

# Generate predictions from both models
rf_threshold_probs <- predict(rf_model_final, prediction_data, type = "prob")[, "Good"]
gbm_threshold_probs <- predict(gbm_model_final, newdata = prediction_data,
                               n.trees = gbm_best_iter, type = "response")

# Create results dataframe
threshold_results <- data.frame(
  alcohol = alcohol_range,
  rf_probability = rf_threshold_probs,
  gbm_probability = gbm_threshold_probs
)

# Find thresholds
rf_threshold <- threshold_results %>%
  filter(rf_probability >= 0.5) %>%
  slice(1) %>%
  pull(alcohol)

gbm_threshold <- threshold_results %>%
  filter(gbm_probability >= 0.5) %>%
  slice(1) %>%
  pull(alcohol)

cat("Random Forest Alcohol Threshold (>50% prob):", round(rf_threshold, 2), "%\n")
cat("GBM Alcohol Threshold (>50% prob):", round(gbm_threshold, 2), "%\n")

# Visualization
threshold_plot <- ggplot(threshold_results, aes(x = alcohol)) +
  geom_line(aes(y = rf_probability, color = "Random Forest"), size = 1.2) +
  geom_line(aes(y = gbm_probability, color = "Gradient Boosting"), size = 1.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", size = 1) +
  geom_vline(xintercept = rf_threshold, linetype = "dotted", color = "#F8766D") +
  geom_vline(xintercept = gbm_threshold, linetype = "dotted", color = "#00BFC4") +
  annotate("text", x = rf_threshold + 0.3, y = 0.2, 
           label = paste0("RF: ", round(rf_threshold, 2), "%"),
           color = "#F8766D", size = 4) +
  annotate("text", x = gbm_threshold + 0.3, y = 0.3, 
           label = paste0("GBM: ", round(gbm_threshold, 2), "%"),
           color = "#00BFC4", size = 4) +
  labs(title = "Probability of Good Quality Wine vs. Alcohol Content",
       subtitle = "Comparison of Random Forest and Gradient Boosting",
       x = "Alcohol Content (%)",
       y = "Probability of Good Quality (≥6)",
       color = "Model") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

print(threshold_plot)

# Summary table
summary_table <- threshold_results %>%
  filter(alcohol %in% seq(8, 15, by = 0.5)) %>%
  mutate(rf_probability = round(rf_probability, 3),
         gbm_probability = round(gbm_probability, 3),
         rf_prediction = ifelse(rf_probability >= 0.5, "Good", "Bad"),
         gbm_prediction = ifelse(gbm_probability >= 0.5, "Good", "Bad"))

cat("\nProbability by Alcohol Content:\n")
print(summary_table)

# ============================================================================
# 5. MODEL COMPARISON AND VISUALIZATION
# ============================================================================

cat("\n=== OVERALL MODEL COMPARISON ===\n")

# Create comparison dataframe
model_comparison <- data.frame(
  Model = c("Random Forest", "Gradient Boosting"),
  Train_Accuracy = c(rf_cm_train$overall["Accuracy"], gbm_cm_train$overall["Accuracy"]),
  Test_Accuracy = c(rf_cm_test$overall["Accuracy"], gbm_cm_test$overall["Accuracy"]),
  Precision = c(rf_cm_test$byClass["Precision"], gbm_cm_test$byClass["Precision"]),
  Recall = c(rf_cm_test$byClass["Recall"], gbm_cm_test$byClass["Recall"]),
  F1_Score = c(rf_cm_test$byClass["F1"], gbm_cm_test$byClass["F1"]),
  AUC = c(auc(rf_roc), auc(gbm_roc))
)

print(round(model_comparison, 4))

# ROC Curve Comparison
roc_plot <- ggplot() +
  geom_line(data = data.frame(x = 1 - rf_roc$specificities, y = rf_roc$sensitivities),
            aes(x = x, y = y, color = "Random Forest"), size = 1.2) +
  geom_line(data = data.frame(x = 1 - gbm_roc$specificities, y = gbm_roc$sensitivities),
            aes(x = x, y = y, color = "Gradient Boosting"), size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve Comparison",
       x = "False Positive Rate",
       y = "True Positive Rate",
       color = "Model") +
  annotate("text", x = 0.6, y = 0.3, 
           label = paste0("RF AUC: ", round(auc(rf_roc), 3)),
           color = "#F8766D", size = 4) +
  annotate("text", x = 0.6, y = 0.2, 
           label = paste0("GBM AUC: ", round(auc(gbm_roc), 3)),
           color = "#00BFC4", size = 4) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(roc_plot)

# Feature Importance Comparison
feature_comparison <- data.frame(
  Feature = rf_importance_df$Feature,
  RF_Importance = rf_importance_df$MeanDecreaseAccuracy,
  GBM_Importance = gbm_importance$rel.inf[match(rf_importance_df$Feature, 
                                                gbm_importance$var)]
)

# Normalize to 0-100 scale
feature_comparison$RF_Importance <- (feature_comparison$RF_Importance / 
                                       max(feature_comparison$RF_Importance)) * 100
feature_comparison$GBM_Importance <- (feature_comparison$GBM_Importance / 
                                        max(feature_comparison$GBM_Importance, na.rm = TRUE)) * 100

# Reshape for plotting
library(tidyr)
feature_long <- feature_comparison %>%
  pivot_longer(cols = c(RF_Importance, GBM_Importance),
               names_to = "Model",
               values_to = "Importance") %>%
  mutate(Model = gsub("_Importance", "", Model))

importance_plot <- ggplot(feature_long, aes(x = reorder(Feature, Importance), 
                                            y = Importance, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Feature Importance Comparison",
       x = "Feature",
       y = "Relative Importance (%)") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(importance_plot)

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

# Save models
saveRDS(rf_model_final, "rf_model_final.rds")
saveRDS(gbm_model_final, "gbm_model_final.rds")

# Save results
write.csv(model_comparison, "model_comparison.csv", row.names = FALSE)
write.csv(threshold_results, "alcohol_threshold_analysis.csv", row.names = FALSE)
write.csv(feature_comparison, "feature_importance_comparison.csv", row.names = FALSE)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Models and results saved successfully!\n")
