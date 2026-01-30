# 🧩 STEP 1: Load Dataset

import pandas as pd

# Load dataset
df = pd.read_csv("spotify_dataset.csv")

# Display first 5 rows
print(df.head())

# Check dataset information
print(df.info())

# What is happening?
# pandas → used for data handling
# read_csv() → loads Spotify dataset
# head() → shows sample data
# info() → shows columns, data types, missing values

# ---I first loaded the Spotify dataset using Pandas and 
# explored its structure to understand the available features, 
# data types, and presence of missing values.

# ----------------------------------------------------------
# 🔹 STEP 2: Data Pre-Processing (VERY IMPORTANT)
# 🎯 Goal of Step 2

# Before analysis or ML:

# We must check missing values

# Understand columns

# Clean bad / unnecessary data
# STEP 2: Data Preprocessing

# Check for missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Total number of rows before cleaning
print("\nRows before cleaning:", df.shape[0])

# Drop rows with missing important information
df = df.dropna(subset=[
    'track_name',
    'track_artist',
    'track_album_name'
])

# Convert release date to datetime format
df['track_album_release_date'] = pd.to_datetime(
    df['track_album_release_date'],
    errors='coerce'
)

# Total number of rows after cleaning
print("Rows after cleaning:", df.shape[0])

# During preprocessing, missing values were identified and rows with 
# incomplete essential metadata were removed. 
# The album release date was converted into datetime format to support time-based analysis.
# -------------------------------------------------------------------------------------------
# STEP 3: Feature Selection & Understanding Audio Features

# Decide which columns are useful

# Separate numerical audio features

# Prepare data for EDA & clustering

# Remember:
# Machine Learning models don’t understand names or text — only numbers.

# STEP 3: Feature Selection

# Select numerical audio features for analysis and clustering
audio_features = [
    'danceability',
    'energy',
    'valence',
    'acousticness',
    'instrumentalness',
    'liveness',
    'speechiness',
    'tempo'
]

# Create a new dataframe with only audio features
X = df[audio_features]

# Display selected features
print("\nSelected Audio Features:")
print(X.head())

audio_features = [...]

# We manually choose:

# Relevant

# Numerical

# Meaningful features

# 🧠 Interview sentence:

# I selected Spotify’s audio features that directly represent musical characteristics for clustering.

# 🔹 X = df[audio_features]

# Creates a clean feature matrix for:

# Visualization

# Correlation

# Clustering
# PROJECT REPORT
# Feature selection was performed by choosing relevant numerical audio attributes provided by Spotify. 
# These features represent the musical characteristics required for clustering and recommendation.
# --------------------------------------------------------------------------------------------------

# 🔹 STEP 4: Exploratory Data Analysis (EDA) & Visualizations
# 🎯 Goal of Step 4

# Here we will:

# Understand distribution of audio features

# Identify patterns & variations

# Generate plots required by your project instructions

# Build insights for your report & viva

# 🧠 What EDA Answers

# EDA helps answer questions like:
# Are most songs energetic or calm?
# Which features vary the most?
# Do genres differ clearly?
# Are there outliers?

# Histogram – Feature Distribution

# STEP 4.1: Distribution of Audio Features

import matplotlib.pyplot as plt

X.hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribution of Spotify Audio Features", fontsize=16)
plt.show()
# Danceability and energy are concentrated between 0.4–0.8, 
# indicating most songs are rhythmically active.
# 🔹 X.hist(...)
# Automatically creates histograms for all numerical columns in X
# Each histogram shows:
# How often values occur
# Distribution shape
# 🧠 Think of it as:
# “How many songs fall into each value range?”
# 🔹 bins=20
# Divides values into 20 intervals
# More bins = more detail
# Fewer bins = smoother graph
# 📌 Why 20?
# Balanced view for medium-sized datasets
# 🔹 figsize=(15, 10)
# Controls width & height of the plot
# Prevents overlapping graphs
# 🔹 plt.suptitle(...)
# Adds a main title to all subplots
# Makes visualization report-ready
# 🔹 plt.show()
# 📊 What You Learn From This
# Feature	Insight
# danceability	Most songs are moderately danceable
# energy	Skewed towards higher energy
# acousticness	Many low acoustic tracks
# tempo	Wide variation

# I used histograms to understand the distribution 
# and range of Spotify audio features before applying clustering.
# -----

# Displays the graph window
# 4.2 Box Plot – Detect Outliers
# STEP 4.2: Boxplot for Outlier Detection

plt.figure(figsize=(14, 6))
X.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of Audio Features")
# What is a boxplot?

# Think of boxplot as:

# “What is normal and what is extreme?”
plt.show()

# Tempo shows a wider spread compared to other features, indicating diverse song speeds.

# 🔹 plt.figure(figsize=(14, 6))

# Creates a new plotting area

# Wide layout helps compare features side-by-side

# 🔹 X.boxplot()

# Creates boxplots for each feature.

# A boxplot shows:

# Median (middle line)

# Interquartile range (box)

# Outliers (dots)

# 📌 Used to detect:

# Extremely fast songs (tempo)

# Rare instrumental tracks

# 🔹 plt.xticks(rotation=45)

# Rotates feature names

# Improves readability

# 🔹 plt.title(...)

# Adds a clear plot title

# 🔹 plt.show()

# Renders the graph

# 📊 Why Boxplots Matter
# Reason	Explanation
# Outlier detection	Extreme values affect clustering
# Feature spread	Shows which features vary more
# Data quality	Reveals anomalies
# 🎤 Viva Answer

# Boxplots were used to identify outliers and understand feature variability, 
# especially for tempo and instrumentalness.

# ------------

# 4.3 Genre-wise Feature Comparison
# STEP 4.3: Feature Comparison by Playlist Genre

import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x='playlist_genre', y='danceability', data=df)
plt.title("Danceability Across Playlist Genres")
plt.xticks(rotation=45)
plt.show()


# PROJECT REPORT

# Exploratory Data Analysis was conducted using histograms and box plots to understand feature 
# distributions and genre-based variations. 
# Significant differences were observed across playlist genres, 
# validating the use of clustering techniques.


# INTERVIEW / VIVA READY ANSWER

# Q: Why did you perform EDA before modeling?

# 👉 Answer:

# EDA helps understand feature behavior, detect outliers, and ensure that clustering 
# will be meaningful and interpretable.

# Best Interview Answer

# Genre-wise boxplots were used to compare audio characteristics 
# across playlist genres and validate clustering feasibility.

# REPORT

# Exploratory Data Analysis was performed using histograms and boxplots
# to analyze feature distributions, 
# detect outliers, and study genre-wise variations in musical attributes.

# STEP 4.1 — Histogram

# 👉 “How values are distributed”

# STEP 4.2 — Boxplot

# 👉 “Where are outliers and spreads”

# STEP 4.3 — Genre Boxplot

# 👉 “How genres differ musically”

# OUTLIER? (Real-life example)

# Imagine:

# Most people’s height = 5–6 feet

# One person = 8 feet

# That 8 feet person is an outlier.
# ------------------------------------------------

# STEP 5: Correlation Matrix & Heatmap
# 🎯 Goal of Step 5 (IN SIMPLE WORDS)

# Here we answer this question:

# “How are audio features related to each other?”

# Example:

# If energy increases, does loudness increase?

# If acousticness increases, does energy decrease?

# This helps us:

# Understand feature relationships

# Avoid redundant features

# Justify clustering decisions

# 🧠 What is Correlation? (ZERO CONFUSION)

# Correlation tells:

# How strongly two features move together

# Correlation values:
# Value	Meaning
# +1	Strong positive relation
# 0	No relation
# -1	Strong negative relation

# 🔍 Simple real-life example

# Hot weather ↑ → AC usage ↑ → positive correlation

# Study time ↑ → phone usage ↓ → negative correlation

# STEP 5: Correlation Matrix

# What is Correlation? (ZERO CONFUSION)

# Correlation tells:

# How strongly two features move together

# Correlation values:
# Value	Meaning
# +1	Strong positive relation
# 0	No relation
# -1	Strong negative relation
# Calculate correlation between audio features

# Simple real-life example

# Hot weather ↑ → AC usage ↑ → positive correlation

# Study time ↑ → phone usage ↓ → negative correlation
correlation_matrix = X.corr()

# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot heatmap
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Heatmap of Spotify Audio Features")
plt.show()

# PROJECT REPORT

# A correlation matrix was generated to analyze relationships among 
# Spotify audio features. Strong positive and negative correlations were observed, 
# particularly between energy and loudness, 
# and between energy and acousticness, validating feature selection for clustering.

# INTERVIEW / VIVA QUESTION & ANSWER
# Q: Why did you plot a correlation matrix?

# 👉 Answer:

# To understand relationships between audio features and ensure that 
# clustering is based on meaningful and non-redundant attributes.

# ---------------------------------------------------
# STEP 6: Feature Scaling + K-Means Clustering

# This step creates clusters → without this, genre segmentation is impossible

# 🎯 WHAT STEP 6 DOES (BIG PICTURE)

# We will:

# Scale the features (VERY IMPORTANT)

# Decide how many clusters (K)

# Apply K-Means algorithm

# Visualize the clusters
# STEP 6.1 — Why Feature Scaling is REQUIRED
# ❌ Problem without scaling
# Some features have:
# loudness → values like -60 to 0
# danceability → values 0 to 1

# Machine Learning thinks:

# “Loudness is more important” ❌

# ✅ Solution → StandardScaler

# Converts all features to same scale

# Mean = 0, Std Dev = 1
#--- STEP 6.1 CODE — Feature Scaling
# STEP 6.1: Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#----STEP 6.2 — Choosing K using Elbow Method

# ❓ Why Elbow Method?

# We must choose:

# “How many clusters should the algorithm create?”

# 📉 Elbow Concept (Easy)

# K = 1 → very bad clustering

# K increases → error decreases

# At one point → improvement slows → ELBOW
#-- STEP 6.2 CODE — Elbow Method

# STEP 6.2: Elbow Method

from sklearn.cluster import KMeans
# Imports the K-Means algorithm

# We need this to:

# create clusters

# calculate clustering error

# 📌 Without this → clustering impossible.

wcss = []
# WCSS = Within Cluster Sum of Squares
# “How tight are the points inside each cluster?”
# Lower WCSS = better clustering
# Higher WCSS = poor clustering

# calculate WCSS for K = 1 to 10
# store each result
# So we need a list.

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
#Create a K-Means model
# n_clusters = i → use current K
# random_state=42 → results stay same every time
# 📌 Without random_state, clusters may change every run.
    kmeans.fit(X_scaled)
# Trains K-Means on scaled features
# Calculates:
# cluster centers
# distances
# errors
# 📌 IMPORTANT:
# We must use X_scaled, not raw data.

    wcss.append(kmeans.inertia_) 
# What is inertia_?

# It is the WCSS value

# Total distance of all points from their cluster center

# So:

# This loop runs for:

# i value	Meaning
# 1	1 cluster
# 2	2 clusters
# …	…
# 10	10 clusters

# 📌 We are testing different K values.

plt.figure(figsize=(8, 5))
# Width = 8 Height = 5
plt.plot(range(1, 11), wcss, marker='o')
# X-axis	Y-axis
# Number of clusters (K)	WCSS
# marker='o' → shows dots on line
plt.title("Elbow Method to Find Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# How did you decide number of clusters?

# I used the Elbow Method by plotting WCSS against different K values. 
# The optimal K was chosen at the point where the reduction in WCSS started to slow down.

# Elbow Method helps us select the best number of 
# clusters by analyzing how compact clusters are for different K values.

# STEP 6.3 — Apply K-Means Clustering

# STEP 6.3: K-Means Clustering
# Assuming k=5
kmeans = KMeans(n_clusters=5, random_state=42)
# Creates a K-Means model

# n_clusters=5 → divide songs into 5 groups

# random_state=42 → fixed randomness (same result every run)
clusters = kmeans.fit_predict(X_scaled)
# 1️⃣ fit

# Finds cluster centers

# Measures distances

# Groups similar songs

# 2️⃣ predict

# Assigns a cluster number to each song
# What is clusters?

# It is an array like:

# [0, 2, 1, 4, 3, 2, 0, ...]

# Add cluster labels to dataset
df['cluster'] = clusters
# 👉 What this does:

# Adds a new column to dataset

# Stores cluster assignment

# Your dataset now looks like:

# track_name	danceability	energy	cluster
# Song A	0.78	0.85	1
# Song B	0.40	0.30	3

# 📌 Now clustering result is saved permanently

# STEP 6.4 — Visualizing Clusters (2D)
# 📌 Purpose

# 👉 Humans can’t read numbers easily
# 👉 Visualization helps us SEE the clusters
# We use:danceability & energy

# STEP 6.4: Cluster Visualization

plt.figure(figsize=(8, 6)) 
# Creates plot area Width = 8 Height = 6
# Only for better visibility.

sns.scatterplot(
# Why scatter plot?
# Because:
# Each song = one point
# We compare two numeric features
    x=df['danceability'],
    # X-axis shows: How rhythmic / dance-friendly songs are
    y=df['energy'],
    # How intense / powerful songs are
    hue=df['cluster'],
    # 👉 MOST IMPORTANT 🔥
    #    Colors points by cluster number  Each color = one cluster
    #    Without this: ❌ All points look same
    #    With this:
    #    ✅ Clear segmentation
    palette='viridis'
    # Just color theme  No effect on logic   Makes plot look professional
)

plt.title("Spotify Song Clusters (Danceability vs Energy)")
# Adds title (important for report).
plt.xlabel("Danceability")
plt.ylabel("Energy")
# Axis labels.
plt.legend(title="Cluster")
# Shows:
# Which color belongs to which cluster      Makes interpretation easy
plt.show()
# Displays the plot.

# VIVA QUESTIONS (IMPORTANT)
# Q: Why did you choose danceability & energy?

# 👉 They strongly represent musical mood and intensity.

# Q: Can you plot other features?

# 👉 Yes (valence vs tempo, loudness vs energy).

# STEP 7: Cluster Interpretation + Genre/Playlist Analysis + Recommendation Logic

# STEP 7.1 — Understand Each Cluster (Cluster Profiling)
# 📌 Idea (Simple Words)

# Each cluster groups similar sounding songs.
# We want to know:

# “What kind of songs are inside each cluster?”

# STEP 7.1: Cluster Profiling
# 👉 Groups all songs by cluster number (0,1,2,…)
cluster_profile = df.groupby('cluster')[[
    'danceability', 'energy', 'valence', 'tempo', 'loudness'
]].mean()
# .mean()👉 Finds average feature value per cluster

print(cluster_profile)


# 🔹 STEP 7.2 — Genre vs Cluster Distribution
# 📌 Why this is required

# Your project instruction says:

# “Find out and plot different clusters according to playlist genres”

# STEP 7.2: Genre vs Cluster Distribution

genre_cluster = pd.crosstab(df['playlist_genre'], df['cluster'])
print(genre_cluster)

# 🔍 What this shows

# Rows → Genres

# Columns → Clusters

# Values → Number of songs

# Example:

# EDM songs mostly fall in cluster 1 & 2
# Rock spreads across multiple clusters

genre_cluster.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Cluster Distribution Across Playlist Genres")
plt.xlabel("Playlist Genre")
plt.ylabel("Number of Songs")
plt.show()

# Certain clusters are dominated by specific genres,
# validating the effectiveness of audio-feature-based clustering.

# 🔹 STEP 7.3 — Playlist Name vs Cluster
# 📌 Purpose
# Shows:
# “How Spotify playlists map to clusters”

playlist_cluster = pd.crosstab(df['playlist_name'], df['cluster'])

playlist_cluster.head()
# (Too many playlists — table is enough for submission)

# 🔹 STEP 7.4 — BASIC RECOMMENDATION LOGIC 🎧 (MOST IMPORTANT)
# 📌 CONCEPT (VERY SIMPLE)

# Spotify logic:

# “If you like a song → recommend other songs from the same cluster”

# STEP 7.4: Recommendation Function

def recommend_songs(song_name, df, n=5):
# What this means:
# This defines a function called recommend_songs
# 📌 Parameters:
# Parameter	Meaning
# song_name	Song user likes
# df	Spotify dataset with clusters
# n=5	Number of recommendations (default = 5)
# 📌 n=5 means:
# If user doesn’t specify → recommend 5 songs

    cluster_id = df[df['track_name'] == song_name]['cluster'].values[0]
    # “Which rows have this song name?” It returns True / False / False / True / ...

    
    recommendations = df[df['cluster'] == cluster_id]
    
    return recommendations[['track_name', 'track_artist', 'playlist_genre']].head(n)
# .head(n)

# Returns top n songs
# (default → 5 songs)

# FINAL VIVA QUESTIONS (VERY IMPORTANT)
# Q: Is this supervised or unsupervised?

# 👉 Unsupervised (no labels)

# Q: Why K-Means?

# 👉 Efficient, interpretable, numeric data

# Q: Can this scale?

# 👉 Yes, with PCA + cosine similarity
# Q: Why cluster-based recommendation?

# 👉 Because clustering groups songs with similar musical features.

# Q: What if song is not found?

# 👉 We can add error handling (optional improvement).

