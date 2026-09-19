import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("loading the data")
df = pd.read_csv('sero_features_4.csv')
print("data loaded successfully")
print(df.shape)
df.head()

missing_values = df.isnull().sum().sum()
print(missing_values)

df=df.drop_duplicates()

df.describe()


# there isnot missing value
#there isnot categorical data
# we donot remove outlier because it is important


class_names = {
    1: 'Sudden Acceleration',
    2: 'Sudden Right Turn',
    3: 'Sudden Left Turn',
    4: 'Sudden Break'
}
df['Behavior_Name'] = df['Target'].map(class_names)
df['AccTotalMagnitude'] = np.sqrt(df['AccMeanX']**2 + df['AccMeanY']**2 + df['AccMeanZ']**2)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=df, x='Behavior_Name', palette='Set2', ax=axes[0])
axes[0].set_title('Distribution of Driving Behaviors', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)


sns.scatterplot(data=df, x='AccTotalMagnitude', y='AccMeanY', hue='Behavior_Name', palette='Set1', ax=axes[1])
axes[1].set_title('Total Acceleration vs Y-Axis Mean', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

X = df.drop(columns=['Target', 'Behavior_Name'])
y = df['Target'] - 1

print(y.value_counts().sort_index())

#the target is balanced

train_indices = []
test_indices = []

# (No Data Leakage)
for class_label in y.unique():
    class_idx = df[y == class_label].index.tolist()
    split_point = int(len(class_idx) * 0.80) 
    train_indices.extend(class_idx[:split_point])
    test_indices.extend(class_idx[split_point:])
    

X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

print("training the model")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVC (Support Vector)': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')
}

results = {}
best_model_name = ""
best_accuracy = 0
best_y_pred = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc * 100

    print(f" {name:20} -> Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_y_pred = y_pred
        
        
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Models Accuracy Comparison (Real Performance)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
for i, v in enumerate(results.values()):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n The best model is {best_model_name} بدقة {best_accuracy * 100:.2f}%\n")

ordered_class_names = [class_names[1], class_names[2], class_names[3], class_names[4]]

print("(Classification Report):")
print(classification_report(y_test, best_y_pred, target_names=ordered_class_names))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ordered_class_names, yticklabels=ordered_class_names)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Behavior ', fontsize=12)
plt.ylabel('Actual Behavior ', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
cm_normalized = confusion_matrix(y_test, best_y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=ordered_class_names, yticklabels=ordered_class_names)
plt.title(f'Normalized Confusion Matrix (%) - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Behavior', fontsize=12)
plt.ylabel('Actual Behavior', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("the best feature")
best_model = models[best_model_name]

if hasattr(best_model, 'feature_importances_'):

    importances = best_model.feature_importances_
    feature_names = X.columns

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)


    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='magma')
    plt.title(f'Top 10 Important Features in {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature Name ', fontsize=12)
    plt.tight_layout()
    plt.show()
    

plt.figure(figsize=(10, 6))


sns.boxplot(data=df, x='Behavior_Name', y='AccTotalMagnitude', palette='Pastel1', showfliers=False)


sns.stripplot(data=df, x='Behavior_Name', y='AccTotalMagnitude', color='black', alpha=0.4, jitter=True)


plt.title('Statistical Distribution of Total Acceleration by Behavior', fontsize=14, fontweight='bold')
plt.xlabel('Driving Behavior', fontsize=12, fontweight='bold')
plt.ylabel('Total Acceleration Magnitude', fontsize=12, fontweight='bold')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()    