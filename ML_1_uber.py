import pandas as pd, numpy as np ,seaborn as sns , matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv(r"C:\Users\Asus\Downloads\ML DATSETS\Uber\uber.csv")


# --- 1) Pre-process ---
df = df.dropna(subset=["fare_amount","pickup_datetime","pickup_longitude","pickup_latitude",
                       "dropoff_longitude","dropoff_latitude","passenger_count"])
print(f"✅ After removing NaN: {len(df):,}")

df = df[(df.fare_amount > 0) & (df.fare_amount < 500)]
df = df[(df.passenger_count >= 1) & (df.passenger_count <= 6)]
print(f"✅ After fixing fare & passenger count: {len(df):,}")

# df = df[(df.pickup_latitude.between(40, 42)) &
#         (df.dropoff_latitude.between(40, 42)) &
#         (df.pickup_longitude.between(-75, -72)) &
#         (df.dropoff_longitude.between(-75, -72))]
df = df[(df['pickup_latitude'] >= -90) & (df['pickup_latitude'] <= 90)]
df = df[(df['pickup_longitude'] >= -180) & (df['pickup_longitude'] <= 180)]
df = df[(df['dropoff_latitude'] >= -90) & (df['dropoff_latitude'] <= 90)]
df = df[(df['dropoff_longitude'] >= -180) & (df['dropoff_longitude'] <= 180)]

print(f"✅ After NYC geo-filter: {len(df):,}")

# Time features
dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
df = df[dt.notna()].copy()
df["hour"] = dt.dt.hour
df["dow"]  = dt.dt.dayofweek

print("✅ Added time features (hour, day of week)")

# Distance calc (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.009
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2-lat1)
    dlmb  = np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

df["dist_km"] = haversine(df.pickup_latitude, df.pickup_longitude,
                          df.dropoff_latitude, df.dropoff_longitude)

df = df[(df.dist_km > 0) & (df.dist_km < 200)]
print(f"✅ After distance cleanup: {len(df):,}")



# --- 2) Outlier removal (IQR) ---
for col in ["fare_amount","dist_km"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    before = len(df)
    df = df[(df[col] >= low) & (df[col] <= high)]
    print(f"✅ Removed outliers in {col}: {before - len(df):,}")

print(f"📌 Final rows after cleaning: {len(df):,}\n")



# --- 3) Correlation ---
feat = ["dist_km","hour","dow","passenger_count",
        "pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]
corr = df[feat + ["fare_amount"]].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# --- 4) Split data ---
X, y = df[feat], df["fare_amount"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_tr, y_tr)

rf = RandomForestRegressor(random_state=42).fit(X_tr, y_tr)
# rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42).fit(X_tr, y_tr)


# --- 5) Evaluation ---
def eval_model(name, mdl):
    p = mdl.predict(X_te)
    r2 = r2_score(y_te, p)
    rmse = np.sqrt(mean_squared_error(y_te, p))  # ✅ works in all sklearn versions

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📌 Model: {name}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.3f}")

eval_model("Linear Regression", lr)
eval_model("Random Forest", rf)
