import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz

# Load dataset
data = pd.read_csv('heart.csv')

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

# Visualize Decision Tree
dot_data = export_graphviz(
    dtree, out_file=None, 
    feature_names=X.columns, 
    class_names=['No Disease', 'Disease'],
    filled=True, rounded=True, special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render('decision_tree_heart', format='png', cleanup=True)  # Saves as PNG

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Accuracy Comparison
dt_acc = accuracy_score(y_test, dtree.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Decision Tree accuracy: {dt_acc:.3f}")
print(f"Random Forest accuracy: {rf_acc:.3f}")

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Feature Importances:\n", importances.sort_values(ascending=False))

# Cross-validation score
rf_cv_score = cross_val_score(rf, X, y, cv=5)
print(f"Random Forest Cross-Validation Accuracy: {rf_cv_score.mean():.3f}")

# For Jupyter Notebook: display the tree image
# from IPython.display import Image
# Image(filename='decision_tree_heart.png')