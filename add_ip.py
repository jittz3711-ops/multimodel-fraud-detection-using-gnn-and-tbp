import pandas as pd
import random

# load dataset
df = pd.read_csv("creditcard.csv")

# generate realistic public IPs
def generate_ip():
    first = random.choice([23, 45, 66, 101, 122, 154, 172, 185, 203])
    return f"{first}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

# add IP column
df["ip_address"] = [generate_ip() for _ in range(len(df))]

# save new file
df.to_csv("creditcard_with_ip.csv", index=False)

print("✅ IP column added")