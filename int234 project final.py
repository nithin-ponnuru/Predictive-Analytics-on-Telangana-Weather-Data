import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('TG Weather data November 2025.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

print("\nMISSING VALUES")
print(df.isnull().sum())

df = df.dropna()
print("\nDuplicate Rows:", df.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

numerical_cols = ['Rain (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 'Min Humidity (%)', 'Max Humidity (%)', 'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)']

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

plt.figure(figsize=(8,5))
df['Rain (mm)'].hist(bins=30)
plt.xlabel("Rain (mm)")
plt.ylabel("Frequency")
plt.title("Rainfall Distribution")
plt.show()

plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    plt.boxplot(df[col].dropna())
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(df['Min Temp (°C)'], df['Rain (mm)'], alpha=0.5)
plt.xlabel('Min Temp (°C)')
plt.ylabel('Rain (mm)')
plt.title('Min Temp vs Rain')

plt.subplot(2, 3, 2)
plt.scatter(df['Max Temp (°C)'], df['Rain (mm)'], alpha=0.5)
plt.xlabel('Max Temp (°C)')
plt.ylabel('Rain (mm)')
plt.title('Max Temp vs Rain')

plt.subplot(2, 3, 3)
plt.scatter(df['Min Humidity (%)'], df['Rain (mm)'], alpha=0.5)
plt.xlabel('Min Humidity (%)')
plt.ylabel('Rain (mm)')
plt.title('Min Humidity vs Rain')

plt.subplot(2, 3, 4)
plt.scatter(df['Max Humidity (%)'], df['Rain (mm)'], alpha=0.5)
plt.xlabel('Max Humidity (%)')
plt.ylabel('Rain (mm)')
plt.title('Max Humidity vs Rain')

plt.subplot(2, 3, 5)
plt.scatter(df['Min Wind Speed (Kmph)'], df['Rain (mm)'], alpha=0.5)
plt.xlabel('Min Wind Speed (Kmph)')
plt.ylabel('Rain (mm)')
plt.title('Min Wind Speed vs Rain')

plt.subplot(2, 3, 6)
plt.scatter(df['Max Wind Speed (Kmph)'], df['Rain (mm)'], alpha=0.5)
plt.xlabel('Max Wind Speed (Kmph)')
plt.ylabel('Rain (mm)')
plt.title('Max Wind Speed vs Rain')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    plt.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

df = df.drop(columns=['District', 'Mandal', 'Date'])

print("="*60)
print("PART 1: MULTIPLE REGRESSION")
print("="*60)

X_reg = df[['Min Temp (°C)', 'Max Temp (°C)', 'Min Humidity (%)', 'Max Humidity (%)', 'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)']]
Y_reg = df['Rain (mm)']

X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_reg, Y_reg, test_size=0.3, random_state=2)

l_model = LinearRegression()
l_model.fit(X_train_reg, Y_train_reg)
y_pred_reg = l_model.predict(X_test_reg)

r2 = r2_score(Y_test_reg, y_pred_reg)
mse = mean_squared_error(Y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_reg, y_pred_reg)

print("R2 Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(Y_test_reg, y_pred_reg)
plt.xlabel("Actual Rain (mm)")
plt.ylabel("Predicted Rain (mm)")
plt.title("Multiple Regression")
plt.plot([min(Y_test_reg), max(Y_test_reg)], [min(Y_test_reg), max(Y_test_reg)], color='red')

print("="*60)
print("PART 2: POLYNOMIAL REGRESSION")
print("="*60)

X_poly = df[['Max Temp (°C)']]
Y_poly = df['Rain (mm)']

poly_reg = PolynomialFeatures(degree=3)
x_pol = poly_reg.fit_transform(X_poly)

model_p = LinearRegression()
model_p.fit(x_pol, Y_poly)
y_pred_poly = model_p.predict(x_pol)

mse_poly = mean_squared_error(Y_poly, y_pred_poly)
mae_poly = mean_absolute_error(Y_poly, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print("MSE:", mse_poly)
print("MAE:", mae_poly)
print("RMSE:", rmse_poly)

plt.subplot(1,3,2)
plt.scatter(X_poly, Y_poly, color="blue", label="Actual")
plt.scatter(X_poly, y_pred_poly, color="red", label="Predicted", alpha=0.5)
plt.xlabel("Max Temp (°C)")
plt.ylabel("Rain (mm)")
plt.title("Polynomial Regression")
plt.legend()

print("="*60)
print("PART 3: K-NEAREST NEIGHBORS CLASSIFICATION")
print("="*60)

df['Rain_Category'] = pd.cut(df['Rain (mm)'], bins=[-1, 0.1, 5, 100], labels=[0, 1, 2])

X_knn = df[['Min Temp (°C)', 'Max Temp (°C)', 'Min Humidity (%)', 'Max Humidity (%)', 'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)']]
Y_knn = df['Rain_Category']

X_train_knn, X_test_knn, Y_train_knn, Y_test_knn = train_test_split(X_knn, Y_knn, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
knn.fit(X_train_knn, Y_train_knn)

y_pred_knn = knn.predict(X_test_knn)
con = confusion_matrix(Y_test_knn, y_pred_knn)
rep = classification_report(Y_test_knn, y_pred_knn)
acc = accuracy_score(Y_test_knn, y_pred_knn)

y_pred_proba = knn.predict_proba(X_test_knn)
r_a = roc_auc_score(Y_test_knn, y_pred_proba, multi_class='ovr')
ll = log_loss(Y_test_knn, y_pred_proba)

print("Confusion matrix:\n", con)
print("\nClassification report:\n", rep)
print("\nAccuracy:", acc)
print("\nROC AUC (ovr):", r_a)
print("\nLog loss:", ll)

plt.subplot(1,3,3)
unique_classes = np.unique(Y_knn)
for cls in unique_classes:
    plt.hist(y_pred_proba[Y_test_knn == cls, int(cls)], alpha=0.5, label=f'Class {cls}')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("KNN Probabilities")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), [KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2).fit(X_train_knn, Y_train_knn).score(X_test_knn, Y_test_knn) for k in range(1, 21)], marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs Number of Neighbors')
plt.grid(True)
plt.show()



from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

print("="*60)
print("PART 4: K-MEANS CLUSTERING (UNSUPERVISED)")
print("="*60)

X_unsup = df[['Min Temp (°C)', 'Max Temp (°C)',
              'Min Humidity (%)', 'Max Humidity (%)',
              'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)',
              'Rain (mm)']]

wcss = []
for i in range(1, 8):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_unsup)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,8), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for K-Means")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_unsup)

plt.figure(figsize=(6,4))
plt.scatter(df['Max Temp (°C)'], df['Rain (mm)'], c=df['KMeans_Cluster'])
plt.xlabel("Max Temp (°C)")
plt.ylabel("Rain (mm)")
plt.title("K-Means Clustering")
plt.show()

print("K-Means Cluster Counts:")
print(df['KMeans_Cluster'].value_counts())

from scipy.cluster.hierarchy import dendrogram, linkage

print("="*60)
print("PART 5: HIERARCHICAL CLUSTERING (ALL LINKAGE METHODS)")
print("="*60)

X_unsup = df[['Min Temp (°C)', 'Max Temp (°C)',
              'Min Humidity (%)', 'Max Humidity (%)',
              'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)',
              'Rain (mm)']]

sample_data = X_unsup.sample(300, random_state=42)

Z_single = linkage(sample_data, method='single')
plt.figure(figsize=(10,6))
dendrogram(Z_single)
plt.title("Hierarchical Clustering - Single Linkage")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

Z_complete = linkage(sample_data, method='complete')
plt.figure(figsize=(10,6))
dendrogram(Z_complete)
plt.title("Hierarchical Clustering - Complete Linkage")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

Z_average = linkage(sample_data, method='average')
plt.figure(figsize=(10,6))
dendrogram(Z_average)
plt.title("Hierarchical Clustering - Average Linkage")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

Z_centroid = linkage(sample_data, method='centroid')
plt.figure(figsize=(10,6))
dendrogram(Z_centroid)
plt.title("Hierarchical Clustering - Centroid Linkage")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

Z_ward = linkage(sample_data, method='ward')
plt.figure(figsize=(10,6))
dendrogram(Z_ward)
plt.title("Hierarchical Clustering - Ward Linkage")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

