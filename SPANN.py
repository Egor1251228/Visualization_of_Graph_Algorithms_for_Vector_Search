import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

st.set_page_config(page_title="SPANN Simulator", layout="wide")

st.title("SPANN Search Simulator 🔍")
st.markdown("Interactive visualization of Approximate Nearest Neighbor (ANN) hybrid search.")

if 'data_initialized' not in st.session_state:
    np.random.seed(42)
    st.session_state.n_points = 300
    st.session_state.X = np.random.rand(st.session_state.n_points, 2)
    st.session_state.n_clusters = 6
    
    kmeans = KMeans(n_clusters=st.session_state.n_clusters, random_state=42, n_init="auto")
    kmeans.fit(st.session_state.X)
    st.session_state.centroids = kmeans.cluster_centers_
    
    st.session_state.data_initialized = True

X = st.session_state.X
centroids = st.session_state.centroids
n_points = st.session_state.n_points
n_clusters = st.session_state.n_clusters

with st.sidebar:
    st.header("SPANN Algorithm Settings")
    eps1 = st.slider("Epsilon 1 (Boundary Duplication)", 
                     min_value=0.0, max_value=1.5, value=0.10, step=0.05)
    
    eps2 = st.slider("Epsilon 2 (Cluster Pruning Threshold)", 
                     min_value=0.0, max_value=2.0, value=0.20, step=0.05)
    
    st.divider()
    
    st.header("Query Point")
    st.write("Use sliders to move the query (red star):")
    query_x = st.slider("X Coordinate", 0.0, 1.0, 0.50, step=0.01)
    query_y = st.slider("Y Coordinate", 0.0, 1.0, 0.50, step=0.01)
    query_point = np.array([[query_x, query_y]])

dist_to_c = cdist(X, centroids)
min_dist_to_c = np.min(dist_to_c, axis=1)

thresholds = min_dist_to_c * (1 + eps1)

cluster_assignments = [[] for _ in range(n_clusters)]
for i in range(n_points):
    assigned = np.where(dist_to_c[i] <= thresholds[i])[0]
    for c_id in assigned:
        cluster_assignments[c_id].append(i)

q_dist = cdist(query_point, centroids)[0]
q_min = np.min(q_dist)

q_threshold = q_min * (1 + eps2)
searched_clusters = np.where(q_dist <= q_threshold)[0]

active_candidates = set()
for c_id in searched_clusters:
    active_candidates.update(cluster_assignments[c_id])

boundary_count = 0
boundary_indices = []
active_indices = list(active_candidates)

for i in active_indices:
    assigned_count = np.sum(dist_to_c[i] <= thresholds[i])
    if assigned_count > 1:
        boundary_count += 1
        boundary_indices.append(i)

normal_active_indices = list(set(active_indices) - set(boundary_indices))

st.subheader("Search Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Clusters Searched", f"{len(searched_clusters)} of {n_clusters}")
col2.metric("Candidates Evaluated", f"{len(active_candidates)} of {n_points}")
col3.metric("Boundary Duplicates", f"{boundary_count}")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, label='Hidden Vectors')

if normal_active_indices:
    ax.scatter(X[normal_active_indices, 0], X[normal_active_indices, 1], 
               c='dodgerblue', s=50, label='Loaded Candidates')

if boundary_indices:
    ax.scatter(X[boundary_indices, 0], X[boundary_indices, 1], 
               c='orange', s=50, label='Boundary Candidates')

unsearched_centroids = list(set(range(n_clusters)) - set(searched_clusters))
if unsearched_centroids:
    ax.scatter(centroids[unsearched_centroids, 0], centroids[unsearched_centroids, 1], 
               c='black', marker='X', s=150, label='Skipped Centroids')

if len(searched_clusters) > 0:
    ax.scatter(centroids[searched_clusters, 0], centroids[searched_clusters, 1], 
               c='lime', marker='X', s=150, label='Searched Centroids')

ax.scatter(query_point[:, 0], query_point[:, 1], 
           c='red', marker='*', s=350, label='Query Point')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Vector Space Map", pad=15, fontsize=14)

ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

st.pyplot(fig)