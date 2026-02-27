import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# =====================================================
# LOAD DATASET
# =====================================================
print("\nLoading dataset...")
df = pd.read_csv("creditcard_with_ip.csv")

# optional sampling for faster training
df = df.sample(15000, random_state=42).reset_index(drop=True)

print("Total Transactions:", len(df))

# =====================================================
# FRAUD ANALYSIS
# =====================================================
print("\nFraud vs Normal Distribution:")
print(df["Class"].value_counts())

fraud_percentage = (df["Class"].sum() / len(df)) * 100
print("Fraud Percentage:", round(fraud_percentage, 3), "%")

# =====================================================
# STEP 1: BEHAVIOR PROFILING
# =====================================================
print("\nCreating behavioral features...")

df['time_gap'] = df['Time'].diff().fillna(0)
df['amount_dev'] = df['Amount'] - df['Amount'].mean()
df['velocity'] = df['Amount'] / (df['time_gap'] + 1)

df['hour'] = (df['Time'] // 3600) % 24
df['night_txn'] = df['hour'].apply(lambda x: 1 if x < 6 else 0)

# =====================================================
# STEP 2: ENSURE IP COLUMN EXISTS
# =====================================================
if "ip_address" not in df.columns:
    print("Generating synthetic IP addresses...")

    def generate_ip():
        first = np.random.choice([23, 45, 66, 101, 122, 154, 172, 185, 203])
        return f"{first}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"

    df["ip_address"] = [generate_ip() for _ in range(len(df))]
else:
    print("IP column found.")

# =====================================================
# STEP 3: IP → COUNTRY (LOCATION MODALITY)
# =====================================================
def get_country(ip):
    try:
        url = f"https://ipapi.co/{ip}/json/"
        data = requests.get(url, timeout=3).json()
        return data.get("country_name", "Unknown")
    except:
        return "Unknown"

print("Fetching location data...")

ip_country_map = {}
unique_ips = df["ip_address"].unique()[:100]   # limit API calls

for ip in unique_ips:
    ip_country_map[ip] = get_country(ip)

df["country"] = df["ip_address"].map(ip_country_map)

fallback = ["India","USA","UK","Germany","Brazil","Russia","Nigeria"]

df["country"] = df["country"].apply(
    lambda x: np.random.choice(fallback) if pd.isna(x) or x=="Unknown" else x
)

# location risk feature
high_risk = ["Nigeria","Russia","Brazil"]
df["location_risk"] = df["country"].apply(lambda x: 1 if x in high_risk else 0)

print("Location features added")

# =====================================================
# STEP 4: FEATURE PREPARATION
# =====================================================
features = df.drop(columns=["Class","ip_address","country"])
labels = df["Class"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# =====================================================
# STEP 5: GRAPH CONSTRUCTION
# connect transactions close in time
# =====================================================
print("\nBuilding graph relationships...")

edges = []
for i in range(len(df)-1):
    if abs(df.loc[i,'Time'] - df.loc[i+1,'Time']) < 2:
        edges.append([i, i+1])

if len(edges) == 0:
    edges = [[i, i+1] for i in range(len(df)-1)]

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# =====================================================
# STEP 6: CREATE GRAPH DATA
# =====================================================
x = torch.tensor(features_scaled, dtype=torch.float)
y = torch.tensor(labels.values, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# train/test split
train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    stratify=y,
    random_state=42
)

train_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask = torch.zeros(len(y), dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

# =====================================================
# STEP 7: GRAPH NEURAL NETWORK MODEL
# =====================================================
class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(data.num_features, 32)
        self.conv2 = SAGEConv(32, 16)
        self.lin = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

# =====================================================
# STEP 8: TRAIN MODEL
# =====================================================
print("\nTraining model...")

model = FraudGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# handle class imbalance
weights = torch.tensor([1, 12], dtype=torch.float)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# =====================================================
# STEP 9: EVALUATION
# =====================================================
print("\nEvaluating model...")

model.eval()
out = model(data)
pred = out.argmax(dim=1)

y_true = data.y[data.test_mask].numpy()
y_pred = pred[data.test_mask].numpy()
y_prob = torch.softmax(out, dim=1)[data.test_mask][:,1].detach().numpy()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_true, y_prob))

# =====================================================
# SAMPLE FRAUD PREDICTIONS
# =====================================================
fraud_predictions = np.where(pred.numpy() == 1)[0][:10]
print("\nSample Predicted Fraud Transactions Indexes:", fraud_predictions)