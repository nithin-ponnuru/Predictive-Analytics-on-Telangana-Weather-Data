import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import io

st.set_page_config(page_title="TG Weather ML Dashboard", layout="wide")
st.title("🌦 Telangana Weather Data Analysis & Machine Learning Dashboard")

# ================= LOAD DATA =================
try:
    df = pd.read_csv('TG Weather data November 2025.csv')
except FileNotFoundError:
    st.error("Dataset not found. Make sure CSV file is in same folder as app.py")
    st.stop()

st.header("DATASET OVERVIEW")
st.write("Dataset Shape:", df.shape)
st.write("First 5 Rows")
st.dataframe(df.head())

st.write("Dataset Info")

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# ================= PREPROCESSING =================
st.header("DATA PREPROCESSING")
st.write("Missing Values")
st.write(df.isnull().sum())

df = df.dropna()
st.write("Duplicate Rows:", df.duplicated().sum())
st.write("Statistical Summary")
st.write(df.describe())

numerical_cols = ['Rain (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 'Min Humidity (%)', 
                  'Max Humidity (%)', 'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)']

# ================= EDA =================
st.header("EXPLORATORY DATA ANALYSIS")

# Histogram Rain
fig1 = plt.figure()
df['Rain (mm)'].hist(bins=30)
plt.title("Rainfall Distribution")
st.pyplot(fig1)

# Boxplots
fig2 = plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3,3,i)
    plt.boxplot(df[col])
    plt.title(col)
plt.tight_layout()
st.pyplot(fig2)

# Scatter plots
fig3 = plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols[1:], 1):
    plt.subplot(2,3,i)
    plt.scatter(df[col], df['Rain (mm)'], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("Rain (mm)")
plt.tight_layout()
st.pyplot(fig3)

# Histograms
fig4 = plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3,3,i)
    plt.hist(df[col], bins=30)
    plt.title(col)
plt.tight_layout()
st.pyplot(fig4)

# Heatmap
fig5 = plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
st.pyplot(fig5)

df = df.drop(columns=['District', 'Mandal', 'Date'])

# ================= MULTIPLE REGRESSION =================
st.header("PART 1: MULTIPLE REGRESSION")

X_reg = df[['Min Temp (°C)', 'Max Temp (°C)', 'Min Humidity (%)',
            'Max Humidity (%)', 'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)']]
Y_reg = df['Rain (mm)']

X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(
    X_reg, Y_reg, test_size=0.3, random_state=2)

l_model = LinearRegression()
l_model.fit(X_train_reg, Y_train_reg)
y_pred_reg = l_model.predict(X_test_reg)

st.write("R2 Score:", r2_score(Y_test_reg, y_pred_reg))
st.write("MSE:", mean_squared_error(Y_test_reg, y_pred_reg))
st.write("RMSE:", np.sqrt(mean_squared_error(Y_test_reg, y_pred_reg)))
st.write("MAE:", mean_absolute_error(Y_test_reg, y_pred_reg))

fig6 = plt.figure()
plt.scatter(Y_test_reg, y_pred_reg)
plt.xlabel("Actual")
plt.ylabel("Predicted")
st.pyplot(fig6)

# ================= POLYNOMIAL REGRESSION =================
st.header("PART 2: POLYNOMIAL REGRESSION")

X_poly = df[['Max Temp (°C)']]
Y_poly = df['Rain (mm)']

poly_reg = PolynomialFeatures(degree=3)
x_pol = poly_reg.fit_transform(X_poly)

model_p = LinearRegression()
model_p.fit(x_pol, Y_poly)
y_pred_poly = model_p.predict(x_pol)

st.write("MSE:", mean_squared_error(Y_poly, y_pred_poly))
st.write("MAE:", mean_absolute_error(Y_poly, y_pred_poly))
st.write("RMSE:", np.sqrt(mean_squared_error(Y_poly, y_pred_poly)))

fig7 = plt.figure()
plt.scatter(X_poly, Y_poly)
plt.scatter(X_poly, y_pred_poly, color="red")
st.pyplot(fig7)

# ================= KNN =================
st.header("PART 3: KNN CLASSIFICATION")

df['Rain_Category'] = pd.cut(df['Rain (mm)'], bins=[-1, 0.1, 5, 100], labels=[0,1,2])

X_knn = X_reg
Y_knn = df['Rain_Category']

X_train_knn, X_test_knn, Y_train_knn, Y_test_knn = train_test_split(
    X_knn, Y_knn, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_knn, Y_train_knn)

y_pred_knn = knn.predict(X_test_knn)

st.write("Accuracy:", accuracy_score(Y_test_knn, y_pred_knn))
st.write("Confusion Matrix")
st.write(confusion_matrix(Y_test_knn, y_pred_knn))
st.write("Classification Report")
st.text(classification_report(Y_test_knn, y_pred_knn))

# Accuracy vs K
fig8 = plt.figure()
plt.plot(range(1,21),
         [KNeighborsClassifier(n_neighbors=k).fit(X_train_knn, Y_train_knn)
          .score(X_test_knn, Y_test_knn) for k in range(1,21)])
st.pyplot(fig8)

# ================= KMEANS =================
st.header("PART 4: K-MEANS CLUSTERING")

X_unsup = df[['Min Temp (°C)', 'Max Temp (°C)',
              'Min Humidity (%)', 'Max Humidity (%)',
              'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)',
              'Rain (mm)']]

wcss = []
for i in range(1,8):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_unsup)
    wcss.append(km.inertia_)

fig9 = plt.figure()
plt.plot(range(1,8), wcss)
st.pyplot(fig9)

kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_unsup)

fig10 = plt.figure()
plt.scatter(df['Max Temp (°C)'], df['Rain (mm)'], c=df['KMeans_Cluster'])
st.pyplot(fig10)

st.write("Cluster Counts")
st.write(df['KMeans_Cluster'].value_counts())

# ================= HIERARCHICAL =================
st.header("PART 5: HIERARCHICAL CLUSTERING")

sample_data = X_unsup.sample(300, random_state=42)

for method in ['single','complete','average','centroid','ward']:
    Z = linkage(sample_data, method=method)
    fig = plt.figure(figsize=(8,4))
    dendrogram(Z)
    plt.title(f"{method.capitalize()} Linkage")
    st.pyplot(fig)

st.success("=== ANALYSIS COMPLETE ===")
