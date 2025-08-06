Here's a clean and complete results comparison table for both datasets you’ve tested so far using:

GridSearchCV

RandomizedSearchCV

BRAGSGridSearch

BRAGSRandomSearch

📊 BRAGS Evaluation Table (So Far)
| Dataset | Method | R² / Accuracy | Time (s) | Best Params |
| :--- | :--- | :--- | :--- | :--- |
| **Iris (Classification)** | GridSearchCV | 1.000 | 1.28 | `{'criterion': 'gini', 'max_depth': None, 'n_estimators': 10}` |
| | RandomizedSearchCV | 1.000 | 0.38 | `{'n_estimators': 10, 'max_depth': None, 'criterion': 'gini'}` |
| | BRAGSGridSearch | 1.000 | 0.46 | `{'n_estimators': 10, 'max_depth': None, 'criterion': 'gini'}` |
| | BRAGSRandomSearch | 1.000 | 0.21 | `{'n_estimators': 100, 'max_depth': 3, 'criterion': 'entropy'}` |
| **California Housing (Regression)** | GridSearchCV | 0.8438 | 16.36 | `{'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 100, 'num_leaves': 31}` |
| | RandomizedSearchCV | 0.8276 | 4.65 | `{'num_leaves': 15, 'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.2}` |
| | BRAGSGridSearch | 0.8438 | 4.91 | `{'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 10, 'num_leaves': 31}` |
| | BRAGSRandomSearch | 0.8350 | 1.38 | `{'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 10, 'num_leaves': 15}` |

### 🧠 Key Takeaways
- **BRAGSGrid** consistently matches full Grid accuracy/R², with **~3x less time**
- **BRAGSRandom** outperforms RandomSearch in **score and time**
- Even in tiny datasets like **Iris**, BRAGS avoids redundancy
- In larger datasets like **California Housing**, BRAGS proves **resource-efficient and accurate**
