import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==============================
# STEP 1: Load Dataset
# ==============================
df = pd.read_csv("spotify_dataset.csv")


# ==============================
# STEP 2: Data Preprocessing
# ==============================
df = df.dropna(subset=[
    "track_name",
    "track_artist",
    "track_album_name"
])

df["track_album_release_date"] = pd.to_datetime(
    df["track_album_release_date"],
    errors="coerce"
)


# ==============================
# STEP 3: Feature Selection
# ==============================
audio_features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo"
]

X = df[audio_features]


# ==============================
# STEP 4: Exploratory Data Analysis
# ==============================
# Feature distributions
X.hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribution of Spotify Audio Features")
plt.show()

# Outlier detection
plt.figure(figsize=(14, 6))
X.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of Audio Features")
plt.show()

# Genre-wise comparison
plt.figure(figsize=(12, 6))
sns.boxplot(x="playlist_genre", y="danceability", data=df)
plt.title("Danceability Across Playlist Genres")
plt.xticks(rotation=45)
plt.show()


# ==============================
# STEP 5: Correlation Analysis
# ==============================
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Audio Features")
plt.show()


# ==============================
# STEP 6: Feature Scaling
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# STEP 6.1: Elbow Method
# ==============================
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# ==============================
# STEP 6.2: K-Means Clustering
# ==============================
kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)


# ==============================
# STEP 6.3: Cluster Visualization
# ==============================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["danceability"],
    y=df["energy"],
    hue=df["cluster"],
    palette="viridis"
)
plt.title("Spotify Song Clusters (Danceability vs Energy)")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.legend(title="Cluster")
plt.show()


# ==============================
# STEP 7: Cluster Profiling
# ==============================
cluster_profile = df.groupby("cluster")[[
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness"
]].mean()

print(cluster_profile)


# ==============================
# STEP 7.1: Genre vs Cluster Distribution
# ==============================
genre_cluster = pd.crosstab(df["playlist_genre"], df["cluster"])
genre_cluster.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Cluster Distribution Across Playlist Genres")
plt.xlabel("Playlist Genre")
plt.ylabel("Number of Songs")
plt.show()


# ==============================
# STEP 7.2: Recommendation Function
# ==============================
def recommend_songs(song_name, df, n=5):
    cluster_id = df[df["track_name"] == song_name]["cluster"].values[0]
    recommendations = df[df["cluster"] == cluster_id]
    return recommendations[[
        "track_name",
        "track_artist",
        "playlist_genre"
    ]].head(n)
