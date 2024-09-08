# 🚢 Titanic Survival Predictor: Will You Make It? 🧊

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Dataset](#-dataset)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Data Preprocessing](#-data-preprocessing)
7. [Model Training](#-model-training)
8. [Model Performance](#-model-performance)
9. [Visualizations](#-visualizations)
10. [Future Improvements](#-future-improvements)
11. [Contributing](#-contributing)
12. [License](#-license)

## 🌟 Overview

Ahoy, data enthusiasts and Titanic aficionados! Welcome aboard the Titanic Survival Predictor, where we're about to embark on a thrilling journey through the world of machine learning to predict who would have survived the infamous Titanic disaster. Buckle up (or should we say, put on your life vests?) as we dive into this exciting project! 🏊‍♂️

On that fateful night of April 15, 1912, the "unsinkable" Titanic met its match in the form of a sneaky iceberg, leading to one of history's most notorious maritime disasters. But fear not! We're here to unravel the mysteries of survival using the power of data science and machine learning. 🕵️‍♀️🔍

In this project, we'll analyze passenger data faster than the Titanic's top speed (a whopping 23 knots, in case you're curious) to predict survival outcomes. We've trained a model that's more accurate than the Titanic's iceberg detection system, achieving an impressive 83% accuracy! So, are you ready to test your fate? Let's set sail into the sea of data! 🌊📊

## 📊 Dataset

Our treasure trove of data comes from the famous Titanic dataset. Let's take a look at our feature lineup:

| Emoji | Feature Name | Description |
|-------|--------------|-------------|
| 🆔 | PassengerId | Unique identifier for each passenger (because even data needs a boarding pass) |
| 💖 | Survived | Target variable (0 = Davy Jones' Locker, 1 = Safe and Sound) |
| 🎫 | Pclass | Passenger class (1 = Fancy, 2 = Less Fancy, 3 = Steerage) |
| 📛 | Name | Passenger's full name (may or may not include "Unsinkable" as a middle name) |
| ♀️♂️ | Sex | Passenger's gender |
| 🎂 | Age | Passenger's age in years (from babes to seasoned sailors) |
| 👨‍👩‍👧‍👦 | SibSp | Number of siblings/spouses aboard (family cruise, anyone?) |
| 👪 | Parch | Number of parents/children aboard (all hands on deck!) |
| 🎟️ | Ticket | Ticket number (not valid for future voyages) |
| 💰 | Fare | Passenger fare (in 1912 dollars, not adjusted for inflation) |
| 🛏️ | Cabin | Cabin number (oceanview not guaranteed) |
| 🚢 | Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

This dataset is like a well-packed suitcase – full of valuable information, but with a few challenges to unpack:
- Missing values (maybe they fell overboard?)
- Categorical variables that need some TLC
- Potential correlations that could rock our boat

But don't worry, we'll navigate these choppy waters with the skill of a seasoned captain! ⚓️

## 🗂 Project Structure

Our ship's manifest (project structure) is as follows:

- `titanic_survival_prediction.ipynb`: The captain's log (main Jupyter notebook)
- `train.csv`: Our training data (like lifeboat drills, but with numbers)
- `test.csv`: Testing data (the real deal, no lifejackets allowed)
- `submissions.csv`: Final predictions (our crystal ball for Titanic fates)
- `README.md`: You are here! (X marks the spot 🗺️)

## 🛠 Installation

Before we set sail, let's make sure our ship is fully equipped. Run these commands to install the necessary supplies (libraries):

```bash
pip install pandas sklearn matplotlib optuna pickle joblib
```

Need an upgrade? No problem! Just run:

```bash
pip install --upgrade library_name
```

Now, let's bring this ship into your harbor (clone the repository):

```bash
git clone https://github.com/AbdouENSIA/Titanic-Survival-Predictor.git
cd Titanic-Survival-Predictor
```

## 🚀 Usage

Ready to predict some fates? Here's how to captain this ship:

1. Fire up the `titanic_survival_prediction.ipynb` notebook in Jupyter Lab or Jupyter Notebook.
2. Run the cells in order, and watch the magic happen!
3. Check out the `submissions.csv` file for the final predictions. No peeking ahead!

## 🧹 Data Preprocessing

Our data preprocessing steps include:

## 🔧 Data Preprocessing and Feature Engineering

### 1. Handling Missing Data
- **Age Imputation:** Missing ages were imputed using the median values, grouped by `Pclass` and `Sex`. This approach helps to preserve the distribution of ages within each class and gender.
- **Embarkation Port:** Missing values in the `Embarked` column were filled with the most common value, 'S' for Southampton. This approach assumes that the majority of passengers embarked at Southampton.
- **Cabin Column:** The `Cabin` column was dropped due to a high percentage of missing values. This decision helps to avoid introducing bias or inaccuracies from a column with significant gaps in data.

### 2. Feature Engineering
- **FamilySize Feature:** A new feature, `FamilySize`, was created by combining `SibSp` (siblings/spouses) and `Parch` (parents/children). This feature captures the total number of family members aboard.
- **Title Extraction:** Titles were extracted from passenger names (e.g., Mr., Mrs., Miss) and grouped into categories. This helps to identify social status or role which might influence survival.
- **IsAlone Feature:** A binary feature, `IsAlone`, was created based on `FamilySize`. It indicates whether a passenger was traveling alone or with family.

### 3. Encoding Categorical Variables
- **One-Hot Encoding:** Applied to the `Embarked` and `Title` features. This method converts categorical variables into a set of binary features.
- **Binary Encoding:** Used for the `Sex` feature to convert it into a numerical format.
- **Ordinal Encoding:** Applied to the `Title` feature to represent different titles with ordered numerical values.
- **Label Encoding:** Used for the `Pclass` feature to convert class categories into numerical labels.

### 4. Additional Enhancements
- **Feature Factorization:** Some features were factorized to simplify their representation and improve model performance.
- **Ticket Group Size:** Created a `TicketGroupSize` feature to denote the number of people sharing the same ticket. This can provide additional context about the ticket's significance.

These preprocessing and feature engineering steps have helped refine the dataset, making it more suitable for modeling and potentially improving the performance of the Titanic Survival Predictor.

## 🤖 Model Training

Alright, time to train our model to be the best fortune-teller it can be!

1. We started with a simple model (the SS Baseline):
   - Split our data into training (80%) and validation (20%) sets
   - Trained a Random Forest Classifier with default parameters
   - Used `n_jobs=-1` to harness the power of all CPU cores (because more power = more fun!)
   - Achieved a respectable 75% accuracy (not bad, but not quite unsinkable)

   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
   
   baseline_model = RandomForestClassifier(n_jobs=-1, random_state=42)
   baseline_model.fit(X_train, y_train)
   baseline_accuracy = baseline_model.score(X_val, y_val)
   print(f"Baseline Model Accuracy: {baseline_accuracy:.2f}")
   ```

2. Enter Optuna, our hyperparameter optimization superhero! 🦸‍♂️
   Optuna is like having a team of expert shipbuilders who try out thousands of different ship designs to find the perfect one. Here's how it worked its magic:

   - Created an Optuna study to optimize these hyperparameters:
     - n_estimators (number of trees in our random forest)
     - max_depth (how deep our trees can grow)
     - min_samples_split (minimum samples required to split a node)
     - min_samples_leaf (minimum samples required in a leaf node)
     - max_features (maximum number of features to consider for splits)

   - Used 5-fold cross-validation (like having 5 different test voyages)
   - Ran 100 trials (tested 100 different ship designs)

3. Final model:
   - Trained a Random Forest Classifier with the best hyperparameters found by Optuna
   - Achieved 83% accuracy on average in cross-validation

## 📈 Model Performance

The improvement from 75% to 83% accuracy demonstrates the effectiveness of our feature engineering and hyperparameter tuning efforts.

## 📊 Visualizations

We’ve included some cool visualizations to help you grasp the data and model performance:

- 📈 **Feature Importance Bar Graph (Model 1):** See which features are making the biggest impact in our first model.
- 🔍 **Optuna Study History:** Check out how our hyperparameters evolved during the Optuna study.
- 📊 **Parameter History:** Explore the history of three different parameters and their impact.
- 🌟 **Feature Importance Bar Chart:** Get a detailed look at the importance of features in our final model.

These visualizations provide great insights into the dataset and help us understand our model’s behavior better.

## 🔮 Future Improvements

To elevate our Titanic Survival Predictor to new heights, we can explore several enhancements:

1. **Ensemble Methods:** Combine predictions from multiple models, such as Random Forests, Gradient Boosting, and Logistic Regression. This could improve our model’s robustness and accuracy by leveraging the strengths of various algorithms.
2. **Deep Learning Approaches:** Dive into advanced neural networks. Implementing deep learning models, such as feedforward neural networks or recurrent neural networks, might uncover deeper patterns in the data.
3. **Data Collection and Augmentation:** Expand our dataset by collecting more data or generating synthetic samples, especially for underrepresented groups. This can help in building a more balanced and generalizable model.
4. **Advanced Feature Engineering:** Explore creating new features by generating interaction terms between existing features. This can capture more complex relationships within the data and potentially improve model performance.
5. **Enhanced Imputation Techniques:** Move beyond basic imputation methods by using advanced techniques like multiple imputation or K-nearest neighbors (KNN) imputation. These methods can provide more accurate estimates for missing values and improve the quality of the dataset.
6. **Stacking Models:** Implement a stacking model that combines predictions from multiple base models. This technique can leverage the strengths of different algorithms and improve overall prediction accuracy.

## 🤝 Contributing

We’re excited to have you contribute to the Titanic Survival Predictor project! Here’s how you can get involved:

1. **Fork the Repository:** Create your own copy of the repository to work on.
2. **Create a New Branch:** For each feature or bug fix, create a new branch from the `main` branch to keep your work organized.
3. **Make and Commit Changes:** Implement your changes and ensure they are well-documented with descriptive commit messages. This helps maintain clarity in the project’s development history.
4. **Push Your Changes:** Push your commits to your forked repository on GitHub.
5. **Submit a Pull Request:** Open a pull request with a detailed description of the changes you’ve made. This allows us to review and discuss your contributions.

**Guidelines:**
- Follow PEP 8 style guidelines to ensure code consistency.
- Include appropriate comments and documentation to make your code easy to understand and maintain.

We appreciate your help in making the Titanic Survival Predictor even better. Happy coding!

## 📜 License

This project is licensed under the MIT License. For detailed information, please refer to the LICENSE file in the repository.

Happy predicting, and may the odds be ever in your favor! 🍀
