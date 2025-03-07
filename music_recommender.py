import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from minisom import MiniSom
import matplotlib.pyplot as plt

# ========================
# 1. Load and Inspect Data
# ========================

# Update 'data.csv' if your file is named differently.
data_file = "data.csv"
df = pd.read_csv(data_file)

# Print basic information about the dataset
print("Dataset info:")
print(df.info())
print("First few rows:")
print(df.head())

# =============================
# 2. Preprocess the Dataset
# =============================

# Specify which columns are numerical and which are categorical.
numerical_cols = ['tempo']          # Add more numerical columns if needed.
categorical_cols = ['genre']          # Add other categorical columns (e.g., 'artist') if desired.

# Create a ColumnTransformer to scale numerical features and one-hot encode categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Apply the transformations to create a feature matrix.
# This matrix (features) will be used to train the SOM.
features = preprocessor.fit_transform(df)

# Convert to a NumPy array (if not already)
features = np.array(features)

print("Feature matrix shape:", features.shape)

# ==============================
# 3. Train the Self-Organizing Map (SOM)
# ==============================

# Set the dimensions of the SOM grid (e.g., 10x10 grid).
som_width, som_height = 10, 10
input_len = features.shape[1]  # Dimensionality of each feature vector

# Initialize the SOM.
# Adjust 'sigma' and 'learning_rate' as needed.
som = MiniSom(som_width, som_height, input_len, sigma=1.0, learning_rate=0.5, random_seed=42)

# Initialize weights (you can also use som.random_weights_init(features) to initialize from the dataset)
som.random_weights_init(features)

# Train the SOM with the feature matrix.
num_iterations = 1000  # Increase if your dataset is larger or more complex.
print("Training the SOM...")
som.train_random(features, num_iterations)
print("SOM training complete.")

# ==============================
# 4. Map Items to the SOM Grid
# ==============================

# For each item (song) in the dataset, get its Best Matching Unit (BMU) on the SOM grid.
item_mapping = [som.winner(x) for x in features]

# Optionally, visualize the SOM distance (U-Matrix) along with item indices.
plt.figure(figsize=(7, 7))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # U-matrix visualization
plt.colorbar(label='Distance')
for idx, (x, y) in enumerate(item_mapping):
    plt.text(x + 0.5, y + 0.5, str(idx), color='red', fontsize=9)
plt.title("SOM U-Matrix with Item Indices")
plt.xlabel("SOM X-axis")
plt.ylabel("SOM Y-axis")
plt.show()

# ==============================
# 5. Generate Recommendations
# ==============================

def recommend_items(query_vector, som, features, item_mapping, radius=1, top_n=5):
    """
    Given a query feature vector, find and recommend similar items.
    
    Parameters:
      - query_vector: The feature vector for the user's selected song/artist.
      - som: The trained MiniSom object.
      - features: The full feature matrix of all items.
      - item_mapping: List of BMU positions for each item.
      - radius: Maximum grid distance from the query's BMU to consider.
      - top_n: Number of recommendations to return.
      
    Returns:
      - List of indices corresponding to recommended items.
    """
    # Find the BMU for the query vector.
    query_bmu = som.winner(query_vector)
    
    # Find candidate items whose BMU is within the given radius.
    candidate_indices = [i for i, pos in enumerate(item_mapping)
                         if np.linalg.norm(np.array(pos) - np.array(query_bmu)) <= radius]
    
    if not candidate_indices:
        print("No candidates found within the specified radius.")
        return []
    
    # Compute cosine similarity between the query and candidate items.
    candidate_features = features[candidate_indices]
    sim_scores = cosine_similarity(query_vector.reshape(1, -1), candidate_features)[0]
    
    # Pair indices with similarity scores and sort in descending order.
    ranked_candidates = sorted(zip(candidate_indices, sim_scores), key=lambda x: x[1], reverse=True)
    
    # Return the indices of the top_n similar items.
    recommended_indices = [idx for idx, score in ranked_candidates[:top_n]]
    return recommended_indices

# Example usage:
# Let’s say a user selects the first item in the dataset.
query_index = 0
query_vector = features[query_index]

# Get recommendations based on the SOM clustering.
recommended_indices = recommend_items(query_vector, som, features, item_mapping, radius=1, top_n=5)

print("Recommendations (by row index):", recommended_indices)
print("Recommended songs and artists:")
print(df.loc[recommended_indices, ['song', 'artist']])
