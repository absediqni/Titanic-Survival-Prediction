# 🚢 Titanic Survival Prediction: End-to-End Analytics & Automation

On April 15, 1912, the RMS Titanic sank, resulting in the loss of over 1,500 lives. Survival was not random — it was strongly influenced by socio-economic status, age, and gender. 

This project applies comprehensive **data analysis, feature engineering, and machine learning** to uncover systemic patterns in survival and build production-ready predictive models using the classic Titanic dataset.

---

## 🧰 Tech Stack & Ecosystem

### Data & Machine Learning
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=flat" alt="Seaborn">
  <img src="https://img.shields.io/badge/SHAP-Model%20Explainability-black?style=flat" alt="SHAP">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

### Business Intelligence & Automation
<p align="left">
  <img src="https://img.shields.io/badge/Power%20BI-Dashboarding-F2C811?style=flat&logo=powerbi&logoColor=black" alt="Power BI">
  <img src="https://img.shields.io/badge/Excel-Power%20Query-217346?style=flat&logo=microsoft-excel&logoColor=white" alt="Excel Power Query">
</p>

---

## 📊 Dataset Reference

* **Source:** [Kaggle Titanic Competition Dataset](https://www.kaggle.com/competitions/titanic/data)
* **Core Variables:** Features passenger demographics, ticket class, fare rates, family structures, cabin assignments, and historical survival outcomes.

---

## 🔍 Project Workflow

| Phase | Core Objectives & Methodology |
| :--- | :--- |
| **1. Exploratory Data Analysis** | Evaluated survival distribution across `Sex`, `Age`, and `Pclass`. Analyzed fare allocations and cross-referenced historical records with family structure trends. |
| **2. Feature Engineering** | Extracted passenger titles (`Mr`, `Mrs`, `Miss`, etc.). Engineered `FamilySize` ($SibSp + Parch + 1$), parsed deck positions from `Cabin` keys, and encoded variables for algorithmic processing. |
| **3. Machine Learning Pipeline** | Established a Logistic Regression baseline. Benchmarked **Random Forest** and **Gradient Boosting Classifiers**, using `GridSearchCV` for exhaustive hyperparameter optimization. |
| **4. Model Explainability** | Integrated **SHAP (SHapley Additive exPlanations)** to break down predictive weights, isolate core feature vectors, and provide absolute transparency over individual model decisions. |
| **5. Interactive Analytics Layers** | Deployed interactive `ipywidgets` for targeted, parameter-driven filtering inside the environment alongside a companion **Power BI Dashboard** for executive-level data exploration. |

---

## 📈 Key Technical Insights

* **Demographic Variance:** `Sex` emerged as the highest-weighted survival feature, heavily reflecting historical maritime rescue protocols.
* **Socio-Economic Leverage:** First-class passengers (`Pclass = 1`) experienced a profound statistical advantage in lifeboat accessibility.
* **Feature Interactions:** Family size exhibited a non-linear optimization curve, and custom title extractions significantly reduced model log-loss metrics.
* **Algorithm Benchmarks:** Non-linear, tree-based models (Random Forest and Gradient Boosting) consistently outperformed linear classification baselines.

---

## 📊 Power BI Dashboard Architecture

The reporting dashboard acts as a dedicated production exploration layer built to monitor key metrics:
* **🧭 Operational Overview:** Core KPIs tracking total volumes, survival percentages, and overall passenger breakdowns.
* **👥 Demographic Segments:** Distribution visualizers isolating metrics across age brackets, gender profiles, and cabin classes.
* **💰 Financial Correlation:** Scatter plots and matrix grids mapping fare distribution against survival rates.
* **👨‍👩‍👧 Family Composition:** Metrics outlining behavioral trends among solo travelers versus varying family-unit sizes.

---

## 📂 Project Architecture

```hl
Titanic-Survival-Prediction/
├── data/                  # Titanic dataset files (train.csv, test.csv)
├── notebooks/             # Core Jupyter development scripts (.ipynb)
├── dashboards/            # Power BI template files (.pbix)
├── requirements.txt       # Hardcoded environmental dependencies
└── README.md              # Project documentation repository index

# 🚀 Future Enhancements
  * Deconstruct the workflow into an automated execution loop using Scikit-Learn Pipelines.
  * Create a lightweight cloud application layer using Streamlit or Flask.
  * Implement automated data ingestion or validation checks utilizing n8n integration engines.
  * Publish and host the interactive dashboard directly on the Power BI Service.