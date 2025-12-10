# Load libraries (already done)
library(randomForest)
library(gbm)
library(dplyr)
library(ggplot2)
library(caret)

# Set working directory
setwd("~/BUS235C/Project/WineQualityProject")

# Load the processed data
wine_data <- read.csv("wine_processed.csv")

# Convert quality_binary to factor
wine_data$quality_binary <- factor(wine_data$quality_binary, levels = c("Bad", "Good"))

# Recreate train/test split with SAME seed as before
set.seed(123)  # IMPORTANT: Use same seed as Week 1
train_index <- createDataPartition(wine_data$quality_binary, p = 0.8, list = FALSE)
wine_train <- wine_data[train_index, ]
wine_test <- wine_data[-train_index, ]

# Add quality_numeric for GBM (needed for predictions)
wine_train$quality_numeric <- as.numeric(wine_train$quality_binary) - 1
wine_test$quality_numeric <- as.numeric(wine_test$quality_binary) - 1

# Verify split
cat("Training set size:", nrow(wine_train), "\n")
cat("Test set size:", nrow(wine_test), "\n")

# Load saved models (already done)
setwd("~/BUS235C/Project/WineQualityProject/Week 3")
rf_model_final <- readRDS("rf_model_final.rds")
gbm_model_final <- readRDS("gbm_model_final.rds")

# Get optimal trees for GBM (already done)
gbm_best_iter <- gbm.perf(gbm_model_final, method = "cv", plot.it = FALSE)

# Define features (already done)
features <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
              "residual.sugar", "chlorides", "free.sulfur.dioxide",
              "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")

# Verify everything is loaded
cat("\nModels loaded successfully!\n")
cat("RF ntree:", rf_model_final$ntree, "\n")
cat("GBM optimal trees:", gbm_best_iter, "\n")


cat("\n=== ALCOHOL SENSITIVITY ANALYSIS ===\n")

# Create a range of alcohol values to test
alcohol_range <- seq(min(wine_data$alcohol), max(wine_data$alcohol), by = 0.1)

# ============================================================================
# Method 1: Hold all features at MEDIAN
# ============================================================================

# Get median values for all features except alcohol
feature_medians <- wine_train %>%
  select(all_of(features)) %>%
  select(-alcohol) %>%
  summarise(across(everything(), median))

# Create prediction dataset with median values
prediction_data_median <- data.frame(
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
rf_probs_median <- predict(rf_model_final, prediction_data_median, type = "prob")[, "Good"]
gbm_probs_median <- predict(gbm_model_final, newdata = prediction_data_median,
                            n.trees = gbm_best_iter, type = "response")

# ============================================================================
# Method 2: Test at different PERCENTILES (25th, 50th, 75th)
# ============================================================================

percentiles <- c(0.25, 0.5, 0.75)
sensitivity_results_rf <- list()
sensitivity_results_gbm <- list()

for(p in percentiles) {
  # Get values at this percentile
  feature_values <- wine_train %>%
    select(all_of(features)) %>%
    select(-alcohol) %>%
    summarise(across(everything(), ~quantile(., p)))
  
  # Create prediction dataset
  pred_data <- data.frame(
    fixed.acidity = rep(feature_values$fixed.acidity, length(alcohol_range)),
    volatile.acidity = rep(feature_values$volatile.acidity, length(alcohol_range)),
    citric.acid = rep(feature_values$citric.acid, length(alcohol_range)),
    residual.sugar = rep(feature_values$residual.sugar, length(alcohol_range)),
    chlorides = rep(feature_values$chlorides, length(alcohol_range)),
    free.sulfur.dioxide = rep(feature_values$free.sulfur.dioxide, length(alcohol_range)),
    total.sulfur.dioxide = rep(feature_values$total.sulfur.dioxide, length(alcohol_range)),
    density = rep(feature_values$density, length(alcohol_range)),
    pH = rep(feature_values$pH, length(alcohol_range)),
    sulphates = rep(feature_values$sulphates, length(alcohol_range)),
    alcohol = alcohol_range
  )
  
  # Predictions
  rf_probs <- predict(rf_model_final, pred_data, type = "prob")[, "Good"]
  gbm_probs <- predict(gbm_model_final, newdata = pred_data,
                       n.trees = gbm_best_iter, type = "response")
  
  # Store results
  sensitivity_results_rf[[paste0("p", p*100)]] <- data.frame(
    alcohol = alcohol_range,
    probability = rf_probs,
    percentile = paste0(p*100, "th percentile")
  )
  
  sensitivity_results_gbm[[paste0("p", p*100)]] <- data.frame(
    alcohol = alcohol_range,
    probability = gbm_probs,
    percentile = paste0(p*100, "th percentile")
  )
}

# Combine results
rf_sensitivity_df <- bind_rows(sensitivity_results_rf)
gbm_sensitivity_df <- bind_rows(sensitivity_results_gbm)

# ============================================================================
# Method 3: Test with GOOD vs BAD wine profiles
# ============================================================================

# Get typical "good" wine profile (from wines rated Good)
good_wine_profile <- wine_train %>%
  filter(quality_binary == "Good") %>%
  select(all_of(features)) %>%
  select(-alcohol) %>%
  summarise(across(everything(), median))

# Get typical "bad" wine profile (from wines rated Bad)
bad_wine_profile <- wine_train %>%
  filter(quality_binary == "Bad") %>%
  select(all_of(features)) %>%
  select(-alcohol) %>%
  summarise(across(everything(), median))

# Create prediction datasets
pred_good_profile <- data.frame(
  fixed.acidity = rep(good_wine_profile$fixed.acidity, length(alcohol_range)),
  volatile.acidity = rep(good_wine_profile$volatile.acidity, length(alcohol_range)),
  citric.acid = rep(good_wine_profile$citric.acid, length(alcohol_range)),
  residual.sugar = rep(good_wine_profile$residual.sugar, length(alcohol_range)),
  chlorides = rep(good_wine_profile$chlorides, length(alcohol_range)),
  free.sulfur.dioxide = rep(good_wine_profile$free.sulfur.dioxide, length(alcohol_range)),
  total.sulfur.dioxide = rep(good_wine_profile$total.sulfur.dioxide, length(alcohol_range)),
  density = rep(good_wine_profile$density, length(alcohol_range)),
  pH = rep(good_wine_profile$pH, length(alcohol_range)),
  sulphates = rep(good_wine_profile$sulphates, length(alcohol_range)),
  alcohol = alcohol_range
)

pred_bad_profile <- data.frame(
  fixed.acidity = rep(bad_wine_profile$fixed.acidity, length(alcohol_range)),
  volatile.acidity = rep(bad_wine_profile$volatile.acidity, length(alcohol_range)),
  citric.acid = rep(bad_wine_profile$citric.acid, length(alcohol_range)),
  residual.sugar = rep(bad_wine_profile$residual.sugar, length(alcohol_range)),
  chlorides = rep(bad_wine_profile$chlorides, length(alcohol_range)),
  free.sulfur.dioxide = rep(bad_wine_profile$free.sulfur.dioxide, length(alcohol_range)),
  total.sulfur.dioxide = rep(bad_wine_profile$total.sulfur.dioxide, length(alcohol_range)),
  density = rep(bad_wine_profile$density, length(alcohol_range)),
  pH = rep(bad_wine_profile$pH, length(alcohol_range)),
  sulphates = rep(bad_wine_profile$sulphates, length(alcohol_range)),
  alcohol = alcohol_range
)

# Predictions for profiles
rf_probs_good <- predict(rf_model_final, pred_good_profile, type = "prob")[, "Good"]
rf_probs_bad <- predict(rf_model_final, pred_bad_profile, type = "prob")[, "Good"]

gbm_probs_good <- predict(gbm_model_final, pred_good_profile, 
                          n.trees = gbm_best_iter, type = "response")
gbm_probs_bad <- predict(gbm_model_final, pred_bad_profile,
                         n.trees = gbm_best_iter, type = "response")

# ============================================================================
# FIND ALCOHOL THRESHOLDS
# ============================================================================

# Threshold for median features
rf_threshold_median <- alcohol_range[which(rf_probs_median >= 0.5)[1]]
gbm_threshold_median <- alcohol_range[which(gbm_probs_median >= 0.5)[1]]

cat("\n--- Alcohol Thresholds (50% probability) ---\n")
cat("Random Forest (median features):", round(rf_threshold_median, 2), "%\n")
cat("GBM (median features):", round(gbm_threshold_median, 2), "%\n")

# Thresholds for different percentiles
cat("\n--- Thresholds by Percentile ---\n")
for(p in percentiles) {
  rf_data <- sensitivity_results_rf[[paste0("p", p*100)]]
  gbm_data <- sensitivity_results_gbm[[paste0("p", p*100)]]
  
  rf_thresh <- rf_data$alcohol[which(rf_data$probability >= 0.5)[1]]
  gbm_thresh <- gbm_data$alcohol[which(gbm_data$probability >= 0.5)[1]]
  
  cat(sprintf("%dth percentile - RF: %.2f%%, GBM: %.2f%%\n", 
              p*100, rf_thresh, gbm_thresh))
}

# Thresholds for wine profiles
rf_thresh_good <- alcohol_range[which(rf_probs_good >= 0.5)[1]]
rf_thresh_bad <- alcohol_range[which(rf_probs_bad >= 0.5)[1]]
gbm_thresh_good <- alcohol_range[which(gbm_probs_good >= 0.5)[1]]
gbm_thresh_bad <- alcohol_range[which(gbm_probs_bad >= 0.5)[1]]

cat("\n--- Thresholds by Wine Profile ---\n")
cat("Good wine profile - RF:", round(rf_thresh_good, 2), "%, GBM:", 
    round(gbm_thresh_good, 2), "%\n")
cat("Bad wine profile  - RF:", round(rf_thresh_bad, 2), "%, GBM:", 
    round(gbm_thresh_bad, 2), "%\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Plot 1: Basic comparison at median
plot1 <- ggplot() +
  geom_line(aes(x = alcohol_range, y = rf_probs_median, color = "Random Forest"), 
            size = 1.2) +
  geom_line(aes(x = alcohol_range, y = gbm_probs_median, color = "Gradient Boosting"), 
            size = 1.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +
  geom_vline(xintercept = rf_threshold_median, linetype = "dotted", 
             color = "#F8766D", alpha = 0.7) +
  geom_vline(xintercept = gbm_threshold_median, linetype = "dotted", 
             color = "#00BFC4", alpha = 0.7) +
  annotate("text", x = rf_threshold_median + 0.3, y = 0.15, 
           label = sprintf("RF: %.2f%%", rf_threshold_median),
           color = "#F8766D", size = 3.5) +
  annotate("text", x = gbm_threshold_median + 0.3, y = 0.25, 
           label = sprintf("GBM: %.2f%%", gbm_threshold_median),
           color = "#00BFC4", size = 3.5) +
  labs(title = "Alcohol Sensitivity: Models Comparison",
       subtitle = "All other features held at median values",
       x = "Alcohol Content (%)",
       y = "Probability of Good Quality",
       color = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 2: RF sensitivity by percentile
plot2 <- ggplot(rf_sensitivity_df, aes(x = alcohol, y = probability, 
                                       color = percentile)) +
  geom_line(size = 1.1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +
  labs(title = "Random Forest: Sensitivity by Feature Percentile",
       x = "Alcohol Content (%)",
       y = "Probability of Good Quality",
       color = "Feature Values") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 3: GBM sensitivity by percentile
plot3 <- ggplot(gbm_sensitivity_df, aes(x = alcohol, y = probability, 
                                        color = percentile)) +
  geom_line(size = 1.1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +
  labs(title = "GBM: Sensitivity by Feature Percentile",
       x = "Alcohol Content (%)",
       y = "Probability of Good Quality",
       color = "Feature Values") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 4: Wine profile comparison
profile_data <- data.frame(
  alcohol = rep(alcohol_range, 4),
  probability = c(rf_probs_good, rf_probs_bad, gbm_probs_good, gbm_probs_bad),
  model = rep(c(rep("Random Forest", 2*length(alcohol_range)), 
                rep("Gradient Boosting", 2*length(alcohol_range)))),
  profile = rep(c(rep("Good Wine Profile", length(alcohol_range)),
                  rep("Bad Wine Profile", length(alcohol_range))), 2)
)

plot4 <- ggplot(profile_data, aes(x = alcohol, y = probability, 
                                  color = profile, linetype = model)) +
  geom_line(size = 1.1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +
  labs(title = "Alcohol Impact by Wine Profile",
       subtitle = "Comparing typical Good vs Bad wine chemical profiles",
       x = "Alcohol Content (%)",
       y = "Probability of Good Quality",
       color = "Wine Profile",
       linetype = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Display plots
print(plot1)
print(plot2)
print(plot3)
print(plot4)

# ============================================================================
# SUMMARY TABLE
# ============================================================================

summary_table <- data.frame(
  Alcohol = seq(8, 15, by = 0.5),
  RF_Median = NA,
  GBM_Median = NA,
  RF_Good_Profile = NA,
  GBM_Good_Profile = NA,
  RF_Bad_Profile = NA,
  GBM_Bad_Profile = NA
)

for(i in 1:nrow(summary_table)) {
  alc_val <- summary_table$Alcohol[i]
  idx <- which.min(abs(alcohol_range - alc_val))
  
  summary_table$RF_Median[i] <- round(rf_probs_median[idx], 3)
  summary_table$GBM_Median[i] <- round(gbm_probs_median[idx], 3)
  summary_table$RF_Good_Profile[i] <- round(rf_probs_good[idx], 3)
  summary_table$GBM_Good_Profile[i] <- round(gbm_probs_good[idx], 3)
  summary_table$RF_Bad_Profile[i] <- round(rf_probs_bad[idx], 3)
  summary_table$GBM_Bad_Profile[i] <- round(gbm_probs_bad[idx], 3)
}

cat("\n--- Probability Summary by Alcohol Content ---\n")
print(summary_table)

# Save results
write.csv(summary_table, "alcohol_sensitivity_summary.csv", row.names = FALSE)

cat("\n=== SENSITIVITY ANALYSIS COMPLETE ===\n")