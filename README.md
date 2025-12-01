# 🎓 Student Performance Prediction — Complete Machine Learning Project  
*End-to-end regression pipeline with PCA, From-Scratch Algorithms, Boosting, SVM, and Stacking*

This project predicts a student's **Performance Index** using a full machine learning workflow.  
The goal is to demonstrate **deep ML understanding**, not deployment — including:

## 📘 Project Highlights

- End-to-end ML workflow with clean modular structure  
- PCA-based dimensionality reduction  
- From-scratch ML implementations for deeper understanding  
- Multiple sklearn models for comparison  
- Boosting and stacking ensembles  
- Strong result evaluation with R², MSE, and visual analysis  
- Organized outputs: results, plots, predictions, models, processed data  


---

## 📘 Dataset
**Student Performance Dataset**  
🔗 https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

**Target Variable:** `Performance_Index`

---

## 🧠 Algorithms Implemented

<div style="display: flex; gap: 40px;">

<div style="flex: 1;">

### 🟦 From Scratch  
- Linear Regression  
- k-Nearest Neighbors  
- Support Vector Regression  
- Manual Stacking  

</div>

<div style="flex: 1;">

### 🟩 Using Sklearn  
- Linear Regression  
- Ridge & Lasso  
- KNN Regressor  
- Decision Tree  
- SVR (RBF Kernel)  
- AdaBoost  
- Gradient Boosting  
- Stacking Regressor  

</div>

</div>


---

## 📂 Project Structure

To keep the repository clean and easy to navigate, files are grouped into logical folders:

- 📘 **Notebook** → [`notebook/student_performance_ml_project.ipynb`](notebook/)
- 📊 **Results** (metrics, comparison tables) → [`results/`](results/)
- 📈 **Plots** (visual outputs) → [`plots/`](plots/)
- 📄 **Predictions** → [`predictions/`](predictions/)
- 🗂 **Raw Data** → [`data_raw/`](data_raw/)
- 🧮 **Processed Data** → [`data_processed/`](data_processed/)
- 🔧 **Models & Encoders** → [`models/`](models/)
- 📦 **All Outputs ZIP** → [`student_performance_project_outputs.zip`](student_performance_project_outputs.zip)

> ✔ This keeps the README clean  
> ✔ Readers can navigate instantly with links  
> ✔ Avoids long directory trees that clutter the page  


---

## 📈 Visualizations Included

- Heatmap of correlations  
- PCA explained variance plot  
- Actual vs Predicted (best model)  
- Model comparison bar chart  
- Error distribution plot  

All available inside the `plots/` folder.

---
## 📊 Results Summary

After evaluating all models across MSE and R² metrics, the **Stacking Regressor (Sklearn)** delivered the best overall performance.

### ⭐ Best Model: Stacking Regressor (Sklearn)
- **Highest R² Score** (~0.90 depending on run)
- **Lowest Mean Squared Error**
- **Most stable and consistent predictions**

### 🔍 Key Insights from Model Comparison
- **Boosting models** (Gradient Boosting, AdaBoost) performed significantly better than single weak learners.
- **Regularized Linear Models** (Ridge & Lasso) showed improvement over standard Linear Regression.
- **From-Scratch Models** (kNN, Linear Regression, SVR) closely matched sklearn performance, validating correctness.
- **PCA** reduced dimensionality while maintaining predictive power, improving model stability.

Detailed metrics for all models are available in the [`results/`](results/) directory.  
Visual comparisons (bar charts, error distributions, predictions vs actual) are in [`plots/`](plots/).

See:  
📄 `final_model_summary_with_rank.csv`  
📊 `plot_model_comparison.png`

---
## ▶️ How to Run the Project

### 1️⃣ Set up the environment
Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate
```
install all requiered dependencies- 
```
pip install -r requirements.txt
```
### 2️⃣ Run the notebook-
`jupyter notebook notebook/student_performance_ml_project.ipynb`
### 3️⃣ Project File Paths (auto-recognized)

The notebook automatically loads and saves files in the following folders:

- **data_raw/** — raw dataset  
- **data_processed/** — scaled data, PCA outputs  
- **plots/** — all visualizations and graphs  
- **predictions/** — model prediction CSVs  
- **results/** — evaluation metrics, comparison tables  
- **models/** — encoders, scaler, and saved preprocessing objects  
No manual path changes are required.

### 4️⃣ (Optional) Reproduce Everything Automatically

If you have **papermill** installed, you can regenerate all results, plots, and outputs with a single command:

`bash
./run_all.sh`
This will create a fully executed notebook at: **notebook/executed_student_performance.ipynb** 


---

### 5️⃣ (Optional) Run in Google Colab

```markdown

1. Open Google Colab  
2. Upload the main notebook:  **notebook/student_performance_ml_project.ipynb**
3. Upload the **data_raw/** folder  
4. Run all cells
```
All other folders (`data_processed`, `plots`, `models`, etc.) will be created automatically by the notebook.



## 🧠 Key Learnings Demonstrated

<div style="display: flex; gap: 40px;">

<div style="flex: 1;">

### 🔹 Data Processing & Preparation  
- Categorical encoding  
- Scaling and normalization  
- PCA dimensionality reduction  
- Clean handling of training/testing sets  
- Organized saving of processed data  

### 🔹 From-Scratch ML  
- Linear Regression using matrix algebra  
- kNN using distance computation  
- SVR with simplified gradient updates  
- Manual stacking using meta-learners  

</div>

<div style="flex: 1;">

### 🔹 Model Training & Evaluation  
- Regression models (Linear, Ridge, Lasso)  
- Tree-based and boosting models  
- SVM with RBF kernel  
- Ensemble stacking (sklearn + manual)  
- R², MSE evaluation metrics  
- Error distribution & prediction analysis  
- Final model ranking & comparison  

</div>

</div>
  

This project reflects depth of understanding, not just model usage.


### Install required libraries:
`bash pip install -r requirements.txt 


## 🚀 Future Work

Here are several extensions planned for the next iteration:

- Hyperparameter tuning with GridSearchCV / Optuna  
- Adding Random Forest / XGBoost / LightGBM  
- Feature importance analysis using SHAP  
- Cross-validation pipelines  
- Outlier detection and data quality checks  
- Interactive dashboard using Streamlit  
- Model deployment (FastAPI + Docker)  

## 👤 Author

**Ankush Patil**  
📍 India  

📧 Email: ankpatil1203@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/ankush-patil-48989739a
🐙 GitHub: https://github.com/Ankush-Patil99  

Feel free to reach out for collaborations or suggestions.

