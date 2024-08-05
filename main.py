import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data (replace with your data path)
data = pd.read_csv("Mall_Customers.csv")

# Select relevant features
features = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']

# Data preprocessing
# Handle missing values (replace with appropriate methods)
data.fillna(method='ffill', inplace=True)

# Convert 'InvoiceDate' to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Extract features (e.g., month, day, year) from 'InvoiceDate'
data['InvoiceMonth'] = data['InvoiceDate'].dt.month
data['InvoiceDay'] = data['InvoiceDate'].dt.day
data['InvoiceYear'] = data['InvoiceDate'].dt.year

# Aggregate data by customer
customer_data = data.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'InvoiceMonth': lambda x: x.nunique()
}).reset_index()

# Feature scaling
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data.drop('CustomerID', axis=1))

# Define number of clusters
n_clusters = 4

# Create K-means model and fit the data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(customer_data_scaled)

# Add cluster labels to customer data
customer_data['Cluster'] = kmeans.labels_

# Analyze results
print(customer_data.groupby('Cluster').describe())
