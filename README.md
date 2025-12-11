# 🍷 Wine Quality Prediction Using Machine Learning

[![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com/tayabhavsar/Wine-Quality-Prediction)

> Predicting Portuguese red wine quality using ensemble machine learning: achieving 81% accuracy while identifying actionable production thresholds for winemakers.

## 📊 Project Overview

This project demonstrates how machine learning can replace costly, subjective expert wine tastings with fast, objective chemical measurements. Using the UCI Wine Quality dataset (1,599 Portuguese Vinho Verde red wines), I developed and compared four predictive models to identify which physicochemical properties most accurately predict expert quality ratings.

**The Challenge**: Traditional wine quality assessment relies on expert sensory evaluation—a process that is expensive, time-consuming, and inherently subjective. Can data mining provide a reliable alternative?

## 🎯 Key Results

- **81.2% prediction accuracy** using Random Forest ensemble (AUC = 0.89)
- **Top 3 chemical predictors identified**: Alcohol content, volatile acidity, and sulphates
- **73% reduction** in required lab tests (3 vs 11 measurements) with only 3% accuracy loss
- **Actionable threshold discovered**: 10-11% alcohol content yields 85-90% probability of good quality
- **Cost-effective screening**: Minimal feature set enables rapid quality control during production

## 🔍 Research Questions Answered

1. **Can quality be predicted using only 3 measurements?**  
   ✅ Yes—alcohol, volatile acidity, and sulphates retain 95-97% of full-model performance

2. **Is alcohol alone sufficient, or are acidity measures necessary?**  
   ✅ Acidity measures are essential—alcohol alone is insufficient for reliable prediction

3. **What minimum alcohol percentage gives >50% probability of good quality?**  
   ✅ 8.4-10% depending on wine's overall chemical profile; optimal target is 10-11%

## 📁 Repository Structure

```
Wine-Quality-Prediction/
├── README.md                          # Project overview and documentation
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── README.md                      # Data documentation and dictionary
│   ├── raw/                           # Original dataset location
│   └── processed/                     # Preprocessed data
│
├── notebooks/                         # R Markdown analysis notebooks
│
├── reports/
│   ├── proposal.pdf                   # Project proposal
│   ├── presentation.pdf               # Final presentation slides
│   └── final_report.pdf               # Comprehensive analysis report
│
├── results/
│   └── figures/                       # Visualizations and plots
│
└── scripts/                           # R scripts for data processing and modeling
```

### Key Components

- **README.md**: Complete project documentation with methodology, results, and usage instructions
- **data/README.md**: Detailed data dictionary with variable descriptions and preprocessing steps
- **reports/**: Three comprehensive PDF documents covering the entire project lifecycle
- **results/figures/**: All visualizations including feature importance, ROC curves, and threshold analyses
- **notebooks/** & **scripts/**: R code for reproducible analysis

## 🚀 Getting Started

### Prerequisites

- **R version**: 4.0 or higher
- **Required packages**:
  ```r
  randomForest, gbm, caret, ggplot2, dplyr, tidyr, 
  ROCR, pROC, corrplot, rpart, rpart.plot
  ```

### Installation

```r
# Install required packages
install.packages(c(
  "randomForest", "gbm", "caret", "ggplot2", "dplyr", 
  "tidyr", "ROCR", "pROC", "corrplot", "rpart", "rpart.plot"
))
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/tayabhavsar/Wine-Quality-Prediction.git
cd Wine-Quality-Prediction

# Open R/RStudio and explore the analysis
# Start with the reports in reports/ folder
# Then examine code in notebooks/ or scripts/
```

## 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression (Initial) | 54.4% | 0.544 | 1.000 | 0.705 | 0.500 |
| Logistic Regression (Improved) | 69.4% | 0.712 | 0.695 | 0.703 | 0.750 |
| Classification Tree | 74.1% | 0.758 | 0.744 | 0.751 | 0.780 |
| **Random Forest** | **81.2%** | **0.842** | **0.791** | **0.816** | **0.890** |
| Gradient Boosting | 77.7% | 0.801 | 0.767 | 0.784 | 0.843 |

**Winner**: Random Forest achieves the best overall performance with superior discrimination (AUC = 0.89)

### Three-Feature Model Performance

Using only the top 3 predictors (alcohol, volatile acidity, sulphates):

| Model | Full Features | Top 3 Only | Accuracy Loss |
|-------|---------------|------------|---------------|
| Logistic Regression | 69.4% | 67.5% | -1.9% |
| Classification Tree | 74.1% | 71.9% | -2.2% |
| Random Forest | 81.2% | 78.1% | -3.1% |
| Gradient Boosting | 77.7% | 75.0% | -2.7% |

**Result**: 95-97% of full-model performance retained with 73% fewer measurements required

## 🛠️ Methodology

### Models Implemented

1. **Logistic Regression**
   - Purpose: Interpretable baseline model
   - Strength: Clear coefficient interpretation
   - Use case: Stakeholder communication

2. **Classification Trees**
   - Purpose: Threshold identification
   - Strength: Actionable decision rules
   - Use case: Operational quality control

3. **Random Forest** ⭐
   - Purpose: Maximum predictive accuracy
   - Strength: Robust to overfitting, captures interactions
   - Use case: Production quality predictions

4. **Gradient Boosting**
   - Purpose: Handle class imbalance
   - Strength: Better calibration, sequential error correction
   - Use case: Probabilistic quality estimates

### Experimental Design

- **Data split**: 80% training (1,279 samples) / 20% test (320 samples)
- **Cross-validation**: 5-fold CV on training set
- **Feature engineering**: Z-score normalization (post-split)
- **Hyperparameter tuning**: Grid search with cross-validation
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📊 Key Visualizations

### Feature Importance

![Feature Importance](results/figures/feature_importance.png)

**Top 3 Predictors Across All Models:**
1. **Alcohol** (32-35% importance): Strongest positive predictor
2. **Volatile Acidity** (16-18% importance): Strongest negative predictor
3. **Sulphates** (14-17% importance): Secondary positive predictor

### Alcohol Threshold Analysis

![Alcohol Threshold](results/figures/Alcohol%20Threshold%20Analysis.jpeg)

**Key Finding**: Quality probability increases non-linearly with alcohol content:
- 8.4% alcohol → 60-65% good quality probability
- 10-11% alcohol → 85-90% good quality probability (optimal target)
- 12%+ alcohol → 90-95% good quality probability (diminishing returns)

### Model Comparison

![ROC Curves](results/figures/ROC%20Curve%20RF%20vs.%20GBM.jpeg)

Random Forest and Gradient Boosting significantly outperform simpler models, with Random Forest achieving the highest AUC (0.89).

## 💼 Business Impact

### For Winemakers

- **Reduced costs**: Streamlined quality control requiring only 3 chemical tests
- **Production guidance**: Clear thresholds for harvest timing (target 10-11% alcohol)
- **Quality improvement**: Identified compensatory relationships (low alcohol can be offset by optimal acidity)
- **Faster decisions**: Objective measurements replace slow expert panels

### Practical Recommendations

1. **Harvest Timing**: Target grape ripeness for 10-11% alcohol potential
2. **Fermentation Control**: Monitor volatile acidity closely (strong negative predictor)
3. **Sulphate Management**: Optimize levels for preservation and flavor
4. **Quality Screening**: Use 3-feature model for rapid batch assessment

## 📚 Dataset

**Source**: UCI Machine Learning Repository  
**Dataset**: Wine Quality Data Set  
**Samples**: 1,599 red wines  
**Features**: 11 physicochemical properties  
**Target**: Quality scores (3-8 scale, median = 6)  

**Citation**:
> P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
> *Modeling wine preferences by data mining from physicochemical properties.*  
> Decision Support Systems, Volume 47, Issue 4, 2009, Pages 547-553.

**Download**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## 🔬 Technical Highlights

✅ Rigorous experimental design with proper train-test splits  
✅ Handled class imbalance (46.5% poor, 53.5% good quality)  
✅ Addressed multicollinearity through correlation analysis  
✅ Comprehensive model evaluation (5 metrics across 5 models)  
✅ Advanced interpretability (SHAP values, feature importance)  
✅ Sensitivity analysis for business thresholds  
✅ Proper statistical validation (confidence intervals, McNemar's test)  

## ⚠️ Limitations

- **Geographic scope**: Limited to Portuguese Vinho Verde wines
- **Quality range**: Dataset lacks extreme quality ratings (only 3-8 of 0-10 scale)
- **Class imbalance**: More medium-quality wines than excellent/poor
- **Temporal stability**: Single vintage; results may vary across years
- **Causal inference**: Models identify correlations, not causation

## 🔮 Future Work

- [ ] Expand to white wines and other varietals
- [ ] Incorporate multi-vintage data for temporal analysis
- [ ] Integrate sensory evaluation data alongside chemical properties
- [ ] Develop real-time quality monitoring dashboard
- [ ] Test transfer learning to other wine regions
- [ ] Explore deep learning for feature interaction discovery

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Taya Bhavsar**

- 📧 Email: [tayab492@gmail.com](mailto:tayab492@gmail.com)
- 💼 LinkedIn: [https://www.linkedin.com/in/tayabhavsar/](https://www.linkedin.com/in/tayabhavsar/)
- 🌐 GitHub: [@tayabhavsar](https://github.com/tayabhavsar)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Prof. Shaonan Tian for guidance throughout this project
- Cortez et al. (2009) for the original research and dataset curation

---

⭐ **If you found this project helpful, please consider giving it a star!**

📫 **Questions or suggestions?** Feel free to open an issue or reach out directly.
