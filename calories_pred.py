# Final CatBoost + Ensemble Enhanced Calorie Prediction 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/calories-dataset/train.csv')
test = pd.read_csv('/kaggle/input/calories-dataset/test.csv')

def preprocess(df):
    df = df.copy()
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    df = df.select_dtypes(include=[np.number])
    df = df.fillna(df.mean())

    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
        df['Height_to_Weight'] = df['Height'] / df['Weight']

    poly_cols = [c for c in ['Age', 'Weight'] if c in df.columns]
    if poly_cols:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df[poly_cols])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(poly_cols), index=df.index)
        df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    return df

X = preprocess(train.drop(columns=['Calories']))
y = train['Calories']
X_test = preprocess(test)

# EDA Visualizations
plt.figure(figsize=(12,10))
sns.heatmap(pd.concat([X, y], axis=1).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

scaler_vis = StandardScaler()
X_scaled_vis = scaler_vis.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled_vis)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=clusters, cmap='rainbow', alpha=0.6)
ax.set_title("KMeans Clusters on PCA Components")
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.colorbar(scatter)
plt.show()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Define tuned CatBoost model 
catboost_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    verbose=0,
    random_state=42
)

# Define other models
ridge = Ridge(alpha=0.1)
xgb = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
# Ensemble
estimators = [
    ('cat', catboost_model),
    ('ridge', ridge),
    ('xgb', xgb)
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    passthrough=True,
    n_jobs=-1,
    cv=3
)

# Cross-validation
cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
scores = cross_val_score(
    stacking,
    X_scaled,
    y,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1
)
print(f"\n✅ Stacking Regressor CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

# Fit model
stacking.fit(X_scaled, y)
joblib.dump(stacking, 'final_stacking_model.pkl')

# Predictions
preds = stacking.predict(X_test_scaled)
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': preds
})
submission.to_csv('final_submission.csv', index=False)
print("✅ Submission saved to final_submission.csv")

# Residual Plot
X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
stacking.fit(X_tr, y_tr)
y_val_pred = stacking.predict(X_val)
residuals = y_val - y_val_pred

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_val_pred, y=residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Calories")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Calories")
plt.show()

# Predicted vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Predicted vs Actual Calories")
plt.show()

# Save the scaler for future use
import joblib
joblib.dump(scaler, 'scaler_model.pkl')