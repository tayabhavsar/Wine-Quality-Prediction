# Load Data-set
#==============================================================================
# Load Data from Week 1 
wine_processed <- read.csv("wine_processed.csv")
wine_glm0_scaled <- readRDS("wine_glm0_scaled_week1.rds")
top_3_predictors <- read.csv("top_3_predictors.csv")


#load libraries 
#=============================================================================
# Load required libraries
library(rpart)          # For decision trees
library(rpart.plot)     # For tree visualization
library(caret)          # For model training and evaluation
library(dplyr)          # For data manipulation
library(ggplot2)        # For visualizations
library(ROCR)           # For ROC curves

# Create binary quality variable (Good >= 6, Bad < 6)
wine_processed$quality_binary <- factor(
  ifelse(wine_processed$quality >= 6, "Good", "Bad"),
  levels = c("Bad", "Good")
)

# Create multi-class quality variable (Low: 3-4, Medium: 5-6, High: 7-8)
wine_processed$quality_class <- factor(
  case_when(
    wine_processed$quality <= 4 ~ "Low",
    wine_processed$quality <= 6 ~ "Medium",
    wine_processed$quality >= 7 ~ "High"
  ),
  levels = c("Low", "Medium", "High")
)

# Create quality_multiclass (if your Week 1 model needs it)
# This is the same as quality_class but with a different name
wine_processed$quality_multiclass <- wine_processed$quality_class

print(table(wine_processed$quality_binary))

print(table(wine_processed$quality_class))


  #Random Test/train Data split 
  #==========================================================================
  #Random Sample (80% training, 20% test set)
  wine_sample <- sample(nrow(wine_processed), nrow(wine_processed)* 0.80)
  wine_train <- wine_processed[wine_sample, ]
  wine_test <- wine_processed[-wine_sample, ]
  
  summary(wine_train)
  summary(wine_test)
  
  # Rebuild the logistic regression model on current data
  #===========================================================================
  # This ensures compatibility with the classification trees
  wine_glm0_scaled <- glm(
    quality_binary ~ fixed.acidity + volatile.acidity + citric.acid + 
      residual.sugar + chlorides + free.sulfur.dioxide + 
      total.sulfur.dioxide + density + pH + sulphates + alcohol,
    data = wine_train,
    family = binomial
  )
  
  cat("Logistic regression model rebuilt successfully!\n")
  cat("Model formula:\n")
  print(formula(wine_glm0_scaled))
  
  # Quick evaluation
  pred_glm_train <- predict(wine_glm0_scaled, wine_train, type = "response")
  pred_glm_train_class <- factor(ifelse(pred_glm_train >= 0.5, "Good", "Bad"), 
                                 levels = c("Bad", "Good"))
  train_accuracy <- sum(pred_glm_train_class == wine_train$quality_binary) / nrow(wine_train)
  cat("\nLogistic Regression Training Accuracy:", round(train_accuracy, 4), "\n")
  
  # ==============================================================================
  # 3. BINARY CLASSIFICATION TREE
  # ==============================================================================
  
  cat("\n" , rep("=", 80), "\n", sep = "")
  cat("BINARY CLASSIFICATION TREE (Good vs Bad Quality)\n")
  cat(rep("=", 80), "\n", sep = "")
  
  # 3.1 Build initial full tree
  # Note: Use column names directly in formula when specifying data argument
  tree_binary_full <- rpart(
    quality_binary ~ fixed.acidity + volatile.acidity + citric.acid + 
      residual.sugar + chlorides + free.sulfur.dioxide + 
      total.sulfur.dioxide + density + pH + sulphates + alcohol,
    data = wine_train,
    method = "class",
    control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7)
  )
  
  print(tree_binary_full)
  
  cat("\nFull tree complexity parameters:\n")
  print(tree_binary_full$cptable)
  
  # 3.1.1 Visualize CP table
  plotcp(tree_binary_full, 
         col = "blue")
  # The dotted line shows the 1-SE rule threshold
  
  # 3.2 Prune the tree using cross-validation
  # Find optimal CP value
  optimal_cp <- tree_binary_full$cptable[
    which.min(tree_binary_full$cptable[, "xerror"]), "CP"
  ]
  cat("\nOptimal CP for pruning:", optimal_cp, "\n")
  
  # Prune the tree
  tree_binary_pruned <- prune(tree_binary_full, cp = optimal_cp)
  
  # 3.2.1 PLOT PRUNED BINARY TREE
  cat("\n--- Plotting Binary Classification Trees ---\n")
  
  # Plot 1: Pruned tree (main visualization)
  pdf("binary_tree_pruned.pdf", width = 16, height = 12)
  rpart.plot(tree_binary_pruned, 
             main = "Binary Classification Tree (Good vs Bad) - Pruned",
             extra = 104,
             box.palette = "RdYlGn",
             shadow.col = "gray",
             nn = TRUE,
             cex = 0.8,
             tweak = 1.2)
  dev.off()
  cat("✓ Pruned tree saved to 'binary_tree_pruned.pdf'\n")
  
  # 3.3 Variable importance
  var_importance_binary <- tree_binary_pruned$variable.importance
  var_importance_df_binary <- data.frame(
    Feature = names(var_importance_binary),
    Importance = as.numeric(var_importance_binary)
  ) %>%
    arrange(desc(Importance)) %>%
    mutate(Importance_Pct = Importance / sum(Importance) * 100)
  
  cat("\nFeature Importance (Binary Classification):\n")
  print(var_importance_df_binary, row.names = FALSE)
  
  # 3.3.1 PLOT FEATURE IMPORTANCE
  pdf("binary_feature_importance.pdf", width = 10, height = 8)
  barplot(var_importance_df_binary$Importance_Pct,
          names.arg = var_importance_df_binary$Feature,
          las = 2,
          col = "steelblue",
          main = "Feature Importance - Binary Classification Tree",
          ylab = "Importance (%)",
          cex.names = 0.8,
          ylim = c(0, max(var_importance_df_binary$Importance_Pct) * 1.1))
  dev.off()
  cat("✓ Feature importance plot saved to 'binary_feature_importance.pdf'\n\n")
  
  # 3.4 Extract decision rules and thresholds
  cat("\nDecision Rules (Binary Tree):\n")
  print(tree_binary_pruned)
  
  # Extract alcohol and volatile acidity thresholds
  tree_rules <- as.data.frame(tree_binary_pruned$splits)
  if (nrow(tree_rules) > 0) {
    cat("\nCritical Thresholds Identified:\n")
    print(tree_rules)
  }
  
  # 3.5 Make predictions on test set
  pred_binary_class <- predict(tree_binary_pruned, wine_test, type = "class")
  pred_binary_prob <- predict(tree_binary_pruned, wine_test, type = "prob")[, "Good"]
  
  # 3.6 Evaluate binary model
  conf_matrix_binary <- confusionMatrix(pred_binary_class, 
                                        wine_test$quality_binary,
                                        positive = "Good")
  
  cat("\nBinary Classification Performance:\n")
  print(conf_matrix_binary)
  
  # Calculate AUC using ROCR
  pred_obj_binary <- prediction(pred_binary_prob, wine_test$quality_binary)
  perf_auc_binary <- performance(pred_obj_binary, "auc")
  auc_value_binary <- perf_auc_binary@y.values[[1]]
  cat("\nAUC:", round(auc_value_binary, 4), "\n")
  
  # Create ROC curve object for plotting
  perf_roc_binary <- performance(pred_obj_binary, "tpr", "fpr")
  
  # 3.6.1 PLOT ROC CURVE
  pdf("binary_roc_curve.pdf", width = 8, height = 8)
  plot(perf_roc_binary, 
       main = "ROC Curve - Binary Classification Tree",
       col = "darkgreen",
       lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")  # Diagonal reference line
  text(0.6, 0.2, paste("AUC =", round(auc_value_binary, 4)), cex = 1.2)
  grid()
  dev.off()
  cat("✓ ROC curve saved to 'binary_roc_curve.pdf'\n\n")
  
  # ==============================================================================
  # 4. MULTI-CLASS CLASSIFICATION TREE
  # ==============================================================================
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("MULTI-CLASS CLASSIFICATION TREE (Low/Medium/High Quality)\n")
  cat(rep("=", 80), "\n", sep = "")
  
  # Create quality class 
  wine_train$quality_class <- factor(
    case_when(
      wine_train$quality <= 4 ~ "Low",
      wine_train$quality <= 6 ~ "Medium",
      wine_train$quality >= 7 ~ "High"
    ),
    levels = c("Low", "Medium", "High")
  )
  # 4.1 Build initial full tree
  tree_multiclass_full <- rpart(
    quality_class ~ fixed.acidity + volatile.acidity + citric.acid + 
      residual.sugar + chlorides + free.sulfur.dioxide + 
      total.sulfur.dioxide + density + pH + sulphates + alcohol,
    data = wine_train,
    method = "class",
    control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7)
  )
  
  cat("\nFull tree complexity parameters:\n")
  print(tree_multiclass_full)
  
  # 4.1.1 Visualize CP table
  plotcp(tree_multiclass_full, 
         col = "purple")
  
  # 4.2 Prune the tree
  optimal_cp_multi <- tree_multiclass_full$cptable[
    which.min(tree_multiclass_full$cptable[, "xerror"]), "CP"
  ]
  cat("\nOptimal CP for pruning:", optimal_cp_multi, "\n")
  
  tree_multiclass_pruned <- prune(tree_multiclass_full, cp = optimal_cp_multi)
  
  # 4.2.1 PLOT MULTI-CLASS TREES
  cat("\n--- Plotting Multi-class Classification Trees ---\n")
  
  # Plot 1: Pruned multi-class tree
  pdf("multiclass_tree_pruned.pdf", width = 16, height = 12)
  rpart.plot(tree_multiclass_pruned,
             main = "Multi-class Classification Tree (Low/Medium/High) - Pruned",
             extra = 104,
             box.palette = "RdYlGn",
             shadow.col = "gray",
             nn = TRUE,
             tweak = 1.2)
  dev.off()
  cat("✓ Multi-class pruned tree saved to 'multiclass_tree_pruned.pdf'\n")

  # Plot 2: Full unpruned multi-class tree
  pdf("multiclass_tree_full_unpruned.pdf", width = 20, height = 16)
  rpart.plot(tree_multiclass_full,
             main = "Full Multi-class Tree (Unpruned)",
             extra = 2,
             box.palette = "RdYlGn",
             cex = 0.6,
             compress = TRUE,
             ycompress = TRUE)
  dev.off()
  cat("✓ Full multi-class unpruned tree saved to 'multiclass_tree_full_unpruned.pdf'\n\n")
  
  # 4.3 Variable importance
  var_importance_multi <- tree_multiclass_pruned$variable.importance
  var_importance_df_multi <- data.frame(
    Feature = names(var_importance_multi),
    Importance = as.numeric(var_importance_multi)
  ) %>%
    arrange(desc(Importance)) %>%
    mutate(Importance_Pct = Importance / sum(Importance) * 100)
  
  cat("\nFeature Importance (Multi-class Classification):\n")
  print(var_importance_df_multi, row.names = FALSE)
  
  # 4.3.1 PLOT FEATURE IMPORTANCE
  pdf("multiclass_feature_importance.pdf", width = 10, height = 8)
  barplot(var_importance_df_multi$Importance_Pct,
          names.arg = var_importance_df_multi$Feature,
          las = 2,
          col = "coral",
          main = "Feature Importance - Multi-class Classification Tree",
          ylab = "Importance (%)",
          cex.names = 0.8,
          ylim = c(0, max(var_importance_df_multi$Importance_Pct) * 1.1))
  dev.off()
  cat("✓ Multi-class feature importance plot saved to 'multiclass_feature_importance.pdf'\n\n")
  
  # Create quality class for TEST set 
  wine_test$quality_class <- factor(
    case_when(
      wine_test$quality <= 4 ~ "Low",
      wine_test$quality <= 6 ~ "Medium",
      wine_test$quality >= 7 ~ "High"
    ),
    levels = c("Low", "Medium", "High")
  )
  
  # 4.4 Make predictions on test set
  pred_multiclass <- predict(tree_multiclass_pruned, wine_test, type = "class")
  
  # Ensure both prediction and reference use the same factor levels
  pred_multiclass <- factor(pred_multiclass, 
                            levels = c("Low", "Medium", "High"))
  
  wine_test$quality_class <- factor(wine_test$quality_class,
                                    levels = c("Low", "Medium", "High"))
  # 4.5 Evaluate multi-class model
  conf_matrix_multi <- confusionMatrix(pred_multiclass, 
                                       wine_test$quality_class)
  
  cat("\nMulti-class Classification Performance:\n")
  print(conf_matrix_multi)
  
  # ==============================================================================
  # 5. HYPERPARAMETER TUNING
  # ==============================================================================
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("HYPERPARAMETER TUNING\n")
  cat(rep("=", 80), "\n", sep = "")
  
  # Define parameter grid
  param_grid <- expand.grid(
    cp = c(0.001, 0.005, 0.01, 0.02, 0.05),
    minsplit = c(10, 20, 30, 40),
    maxdepth = c(3, 5, 7, 10, 15)
  )
  
  cat("\nTuning parameters across", nrow(param_grid), "combinations...\n")
  
  # Initialize results storage
  tuning_results <- data.frame()
  
  # Perform grid search (showing first 20 combinations for efficiency)
  for (i in 1:min(20, nrow(param_grid))) {
    tree_temp <- rpart(
      quality_binary ~ fixed.acidity + volatile.acidity + citric.acid + 
        residual.sugar + chlorides + free.sulfur.dioxide + 
        total.sulfur.dioxide + density + pH + sulphates + alcohol,
      data = wine_train,
      method = "class",
      control = rpart.control(
        cp = param_grid$cp[i],
        minsplit = param_grid$minsplit[i],
        maxdepth = param_grid$maxdepth[i]
      )
    )
    
    pred_temp <- predict(tree_temp, wine_test, type = "class")
    accuracy <- sum(pred_temp == wine_test$quality_binary) / nrow(wine_test)
    
    tuning_results <- rbind(tuning_results, data.frame(
      cp = param_grid$cp[i],
      minsplit = param_grid$minsplit[i],
      maxdepth = param_grid$maxdepth[i],
      accuracy = accuracy
    ))
  }
  
  # Find best parameters
  best_params <- tuning_results[which.max(tuning_results$accuracy), ]
  cat("\nBest hyperparameters found:\n")
  print(best_params)
  
  # Build final tuned model
  tree_binary_tuned <- rpart(
    quality_binary ~ fixed.acidity + volatile.acidity + citric.acid + 
      residual.sugar + chlorides + free.sulfur.dioxide + 
      total.sulfur.dioxide + density + pH + sulphates + alcohol,
    data = wine_train,
    method = "class",
    control = rpart.control(
      cp = best_params$cp,
      minsplit = best_params$minsplit,
      maxdepth = best_params$maxdepth
    )
  )
  
  # 5.1 PLOT TUNED MODEL
  cat("\n--- Plotting Hyperparameter-Tuned Binary Tree ---\n")
  
  # Plot 1: Tuned tree visualization
  pdf("binary_tree_tuned.pdf", width = 16, height = 12)
  rpart.plot(tree_binary_tuned,
             main = paste0("Hyperparameter-Tuned Binary Tree\n",
                           "CP=", round(best_params$cp, 4), 
                           ", minsplit=", best_params$minsplit,
                           ", maxdepth=", best_params$maxdepth),
             extra = 104,
             box.palette = "RdYlGn",
             shadow.col = "gray",
             nn = TRUE,
             cex = 0.8,
             tweak = 1.2)
  dev.off()
  cat("✓ Tuned tree saved to 'binary_tree_tuned.pdf'\n")
  
  # 5.2 Evaluate tuned model
  pred_tuned <- predict(tree_binary_tuned, wine_test, type = "class")
  pred_tuned_prob <- predict(tree_binary_tuned, wine_test, type = "prob")[, "Good"]
  
  conf_matrix_tuned <- confusionMatrix(pred_tuned, 
                                       wine_test$quality_binary,
                                       positive = "Good")
  
  cat("\nTuned Model Performance:\n")
  print(conf_matrix_tuned)
  
  # Calculate AUC for tuned model
  pred_obj_tuned <- prediction(pred_tuned_prob, wine_test$quality_binary)
  perf_auc_tuned <- performance(pred_obj_tuned, "auc")
  auc_value_tuned <- perf_auc_tuned@y.values[[1]]
  cat("\nTuned Model AUC:", round(auc_value_tuned, 4), "\n")
  
  # 5.3 Compare tuned vs pruned model
  cat("\n--- Model Comparison: Pruned vs Tuned ---\n")
  comparison_tuned <- data.frame(
    Model = c("Pruned Tree", "Tuned Tree"),
    Accuracy = c(
      conf_matrix_binary$overall["Accuracy"],
      conf_matrix_tuned$overall["Accuracy"]
    ),
    Precision = c(
      conf_matrix_binary$byClass["Precision"],
      conf_matrix_tuned$byClass["Precision"]
    ),
    Recall = c(
      conf_matrix_binary$byClass["Recall"],
      conf_matrix_tuned$byClass["Recall"]
    ),
    F1_Score = c(
      conf_matrix_binary$byClass["F1"],
      conf_matrix_tuned$byClass["F1"]
    ),
    AUC = c(auc_value_binary, auc_value_tuned)
  )
  print(comparison_tuned, row.names = FALSE, digits = 4)
  
  # 5.4 Plot comparison of tuning results
  pdf("hyperparameter_tuning_results.pdf", width = 12, height = 8)
  par(mfrow = c(1, 2))
  
  # Plot 1: Accuracy across combinations
  plot(1:nrow(tuning_results), tuning_results$accuracy,
       type = "b",
       col = "steelblue",
       pch = 19,
       xlab = "Combination Number",
       ylab = "Test Accuracy",
       main = "Accuracy Across Hyperparameter Combinations",
       ylim = c(min(tuning_results$accuracy) - 0.01, 
                max(tuning_results$accuracy) + 0.01))
  abline(h = max(tuning_results$accuracy), col = "red", lty = 2)
  text(nrow(tuning_results)/2, max(tuning_results$accuracy) + 0.005,
       paste("Best:", round(max(tuning_results$accuracy), 4)),
       col = "red")
  grid()
  
  # Plot 2: Hyper-parameter effects
  boxplot(accuracy ~ maxdepth, data = tuning_results,
          col = "lightgreen",
          main = "Effect of Max Depth on Accuracy",
          xlab = "Max Depth",
          ylab = "Test Accuracy")
  grid()
  
  dev.off()
  cat("✓ Tuning results plots saved to 'hyperparameter_tuning_results.pdf'\n\n")
  
  
  # ==============================================================================
  # 6. COMPARISON VISUALIZATIONS
  # ==============================================================================
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("CREATING COMPARISON VISUALIZATIONS\n")
  cat(rep("=", 80), "\n", sep = "")
  
  # 6.1 Feature importance comparison plot (Binary vs Multi-class)
  pdf("feature_importance_comparison.pdf", width = 12, height = 8)
  par(mfrow = c(1, 2))
  
  # Binary importance
  barplot(var_importance_df_binary$Importance_Pct[1:min(10, nrow(var_importance_df_binary))],
          names.arg = var_importance_df_binary$Feature[1:min(10, nrow(var_importance_df_binary))],
          las = 2,
          col = "steelblue",
          main = "Feature Importance\n(Binary Classification)",
          ylab = "Importance (%)",
          cex.names = 0.8)
  
  # Multi-class importance
  barplot(var_importance_df_multi$Importance_Pct[1:min(10, nrow(var_importance_df_multi))],
          names.arg = var_importance_df_multi$Feature[1:min(10, nrow(var_importance_df_multi))],
          las = 2,
          col = "coral",
          main = "Feature Importance\n(Multi-class Classification)",
          ylab = "Importance (%)",
          cex.names = 0.8)
  dev.off()
  cat("✓ Feature importance comparison saved to 'feature_importance_comparison.pdf'\n")
  
  # ==============================================================================
  # 7. COMPARISON WITH LOGISTIC REGRESSION
  # ==============================================================================
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("COMPARISON: CLASSIFICATION TREE vs LOGISTIC REGRESSION\n")
  cat(rep("=", 80), "\n", sep = "")
  
  # Get logistic regression predictions from Week 1
  pred_glm <- predict(wine_glm0_scaled, wine_test, type = "response")
  pred_glm_class <- factor(ifelse(pred_glm >= 0.5, "Good", "Bad"), 
                           levels = c("Bad", "Good"))
  
  conf_matrix_glm <- confusionMatrix(pred_glm_class, 
                                     wine_test$quality_binary,
                                     positive = "Good")
  
  # Create comparison table
  comparison_df <- data.frame(
    Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "AUC"),
    Logistic_Regression = c(
      conf_matrix_glm$overall["Accuracy"],
      conf_matrix_glm$byClass["Precision"],
      conf_matrix_glm$byClass["Recall"],
      conf_matrix_glm$byClass["F1"],
      NA  # Will calculate AUC separately
    ),
    Classification_Tree = c(
      conf_matrix_binary$overall["Accuracy"],
      conf_matrix_binary$byClass["Precision"],
      conf_matrix_binary$byClass["Recall"],
      conf_matrix_binary$byClass["F1"],
      auc_value_binary
    )
  )
  
  # Calculate GLM AUC using ROCR
  pred_obj_glm <- prediction(pred_glm, wine_test$quality_binary)
  perf_auc_glm <- performance(pred_obj_glm, "auc")
  comparison_df$Logistic_Regression[5] <- perf_auc_glm@y.values[[1]]
  
  cat("\nPerformance Comparison:\n")
  print(comparison_df, row.names = FALSE, digits = 4)
  
  # Save comparison
  comparison_success <- TRUE
  
  }, error = function(e) {
    cat("\n⚠ Warning: Could not compare with Week 1 logistic regression model\n")
    cat("Error message:", e$message, "\n")
    cat("This may be because:\n")
    cat("  1. The GLM model expects different variable names\n")
    cat("  2. The GLM model was trained on different data structure\n")
    cat("  3. Variables in the model don't exist in wine_test\n\n")
    
    cat("Skipping logistic regression comparison...\n")
    cat("You can still compare models manually using saved Week 1 results.\n\n")
    
    # Create comparison with just tree results
    comparison_df <- data.frame(
      Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "AUC"),
      Classification_Tree = c(
        conf_matrix_binary$overall["Accuracy"],
        conf_matrix_binary$byClass["Precision"],
        conf_matrix_binary$byClass["Recall"],
        conf_matrix_binary$byClass["F1"],
        auc(roc_binary)
      )
    )
    
    cat("\nClassification Tree Performance:\n")
    print(comparison_df, row.names = FALSE, digits = 4)
    
    comparison_success <<- FALSE
  })

# ==============================================================================
# 8. ANSWER RESEARCH QUESTIONS
# ==============================================================================

cat("\n", rep("=", 80), "\n", sep = "")
cat("RESEARCH QUESTIONS - PRELIMINARY ANSWERS\n")
cat(rep("=", 80), "\n", sep = "")

# Q1: Can wine quality be accurately predicted using only three measurements?
top_3_features <- var_importance_df_binary$Feature[1:3]
cat("\nQ1: Top 3 chemical measurements for prediction:\n")
cat("   1.", top_3_features[1], "\n")
cat("   2.", top_3_features[2], "\n")
cat("   3.", top_3_features[3], "\n")

# Build model with top 3 features
formula_top3 <- as.formula(paste("quality_binary ~", 
                                 paste(top_3_features, collapse = " + ")))
tree_top3 <- rpart(formula_top3, data = wine_train, method = "class")
pred_top3 <- predict(tree_top3, wine_test, type = "class")
accuracy_top3 <- sum(pred_top3 == wine_test$quality_binary) / nrow(wine_test)

cat("   Accuracy with top 3 features:", round(accuracy_top3, 4), "\n")
cat("   Full model accuracy:", round(conf_matrix_binary$overall["Accuracy"], 4), "\n")
cat("   Performance difference:", 
    round((conf_matrix_binary$overall["Accuracy"] - accuracy_top3) * 100, 2), "%\n")

# Q2: Extract critical thresholds from tree
cat("\nQ2: Critical thresholds from classification tree:\n")
if ("alcohol" %in% rownames(tree_binary_pruned$splits)) {
  alcohol_split <- tree_binary_pruned$splits["alcohol", "index"]
  cat("   Alcohol threshold:", round(alcohol_split, 2), "% ABV\n")
}
if ("volatile.acidity" %in% rownames(tree_binary_pruned$splits)) {
  va_split <- tree_binary_pruned$splits["volatile.acidity", "index"]
  cat("   Volatile acidity threshold:", round(va_split, 3), "g/L\n")
}

# ==============================================================================
# 9. SAVE OUTPUTS
# ==============================================================================

cat("\n", rep("=", 80), "\n", sep = "")
cat("SAVING RESULTS\n")
cat(rep("=", 80), "\n", sep = "")

# Save models
saveRDS(tree_binary_pruned, "tree_binary_pruned_week2.rds")
saveRDS(tree_multiclass_pruned, "tree_multiclass_pruned_week2.rds")
saveRDS(tree_binary_tuned, "tree_binary_tuned_week2.rds")

# Save feature importance
write.csv(var_importance_df_binary, "feature_importance_binary.csv", row.names = FALSE)
write.csv(var_importance_df_multi, "feature_importance_multiclass.csv", row.names = FALSE)

# Save comparison results
write.csv(comparison_df, "model_comparison_week2.csv", row.names = FALSE)

# Save performance metrics
performance_summary <- data.frame(
  Model = c("Binary Tree (Pruned)", "Binary Tree (Tuned)", 
            "Multi-class Tree", "Logistic Regression"),
  Accuracy = c(
    conf_matrix_binary$overall["Accuracy"],
    best_params$accuracy,
    conf_matrix_multi$overall["Accuracy"],
    conf_matrix_glm$overall["Accuracy"]
  ),
  Kappa = c(
    conf_matrix_binary$overall["Kappa"],
    NA,
    conf_matrix_multi$overall["Kappa"],
    conf_matrix_glm$overall["Kappa"]
  )
)
write.csv(performance_summary, "performance_summary_week2.csv", row.names = FALSE)

cat("\nAll results saved successfully!\n")
cat("\nFiles created:\n")
cat("  - tree_binary_pruned_week2.rds\n")
cat("  - tree_multiclass_pruned_week2.rds\n")
cat("  - tree_binary_tuned_week2.rds\n")
cat("  - feature_importance_binary.csv\n")
cat("  - feature_importance_multiclass.csv\n")
cat("  - model_comparison_week2.csv\n")
cat("  - performance_summary_week2.csv\n")
cat("  - binary_tree_visualization.pdf\n")
cat("  - multiclass_tree_visualization.pdf\n")
cat("  - feature_importance_comparison.pdf\n")
cat("  - roc_curve_binary_tree.pdf\n")

cat("\n", rep("=", 80), "\n", sep = "")
cat("WEEK 2 ANALYSIS COMPLETE!\n")
cat(rep("=", 80), "\n", sep = "")

  