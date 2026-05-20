import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import distance

st.set_page_config(page_title="CAGRA Simulator", layout="centered")
st.title("CAGRA Graph Simulator")

def build_cagra_graph(points, num_nodes, out_degree):
    dist_matrix = distance.cdist(points, points)
    
    pruned = {}
    for i in range(num_nodes):
        # Sort all neighbors by distance (excluding the node itself)
        candidates = np.argsort(dist_matrix[i])[1:]
        
        kept = []
        skipped = []
        
        # Rank-based Pruning
        for j in candidates:
            if len(kept) == out_degree:
                break
                
            is_detourable = False
            for k in kept:
                if dist_matrix[k, j] < dist_matrix[i, j]:
                    is_detourable = True
                    break
                    
            if not is_detourable:
                kept.append(j)
            else:
                skipped.append(j)
                
        # If pruning leaves us with fewer than 'out_degree' edges,
        # forcefully fill the remaining slots with the closest skipped nodes
        for j in skipped:
            if len(kept) == out_degree:
                break
            kept.append(j)
            
        pruned[i] = kept

    # Reversal (adding reverse edges for strong connectivity)
    final_graph = {i: list(pruned[i]) for i in range(num_nodes)}
    for i in range(num_nodes):
        for neighbor in pruned[i]:
            if i not in final_graph[neighbor]:
                final_graph[neighbor].append(i)
    
    edges = []
    for u, neighbors in final_graph.items():
        for v in neighbors:
            edges.append([points[u], points[v]])
            
    return final_graph, edges

st.sidebar.header("Graph Settings")
num_nodes = st.sidebar.slider("Nodes", min_value=10, max_value=200, value=60, step=1)
out_degree = st.sidebar.slider("Degree", min_value=2, max_value=12, value=4, step=1)
ef_search = st.sidebar.slider("Search Beam (ef_search)", min_value=1, max_value=50, value=15, step=1)

if ("graph_data" not in st.session_state or 
    st.session_state.get("prev_nodes") != num_nodes or 
    st.session_state.get("prev_degree") != out_degree):
    
    points = np.random.rand(num_nodes, 2)
    graph, edges = build_cagra_graph(points, num_nodes, out_degree)
    
    st.session_state.graph_data = {
        "points": points,
        "graph": graph,
        "edges": edges
    }
    st.session_state.prev_nodes = num_nodes
    st.session_state.prev_degree = out_degree
    
    st.session_state.search_path = None
    st.session_state.query_pt = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Search", type="primary"):
        points = st.session_state.graph_data["points"]
        graph = st.session_state.graph_data["graph"]
        
        query_pt = np.random.rand(2)
        start_node = np.random.randint(num_nodes)
        
        visited = {start_node}
        candidates = [(distance.euclidean(query_pt, points[start_node]), start_node)]
        top_k = [(distance.euclidean(query_pt, points[start_node]), start_node)]
        
        path = [start_node]
        best_dist_so_far = candidates[0][0]

        while candidates:
            candidates.sort(key=lambda x: x[0])
            curr_dist, curr_node = candidates.pop(0)
            
            if curr_dist > top_k[-1][0] and len(top_k) >= ef_search:
                break
                
            for neighbor in graph[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = distance.euclidean(query_pt, points[neighbor])
                    candidates.append((n_dist, neighbor))
                    
                    top_k.append((n_dist, neighbor))
                    top_k.sort(key=lambda x: x[0])
                    top_k = top_k[:ef_search]
                    
                    if n_dist < best_dist_so_far:
                        best_dist_so_far = n_dist
                        path.append(neighbor)
        
        best_node = top_k[0][1]
        if path[-1] != best_node:
            path.append(best_node)
            
        st.session_state.search_path = path
        st.session_state.query_pt = query_pt

with col2:
    if st.button("Regenerate Graph"):
        points = np.random.rand(num_nodes, 2)
        graph, edges = build_cagra_graph(points, num_nodes, out_degree)
        st.session_state.graph_data = {
            "points": points,
            "graph": graph,
            "edges": edges
        }
        st.session_state.search_path = None
        st.session_state.query_pt = None
        st.rerun()

points = st.session_state.graph_data["points"]
edges = st.session_state.graph_data["edges"]
search_path = st.session_state.search_path
query_pt = st.session_state.query_pt

fig, ax = plt.subplots(figsize=(10, 7))

lc = LineCollection(edges, colors='lightgray', linewidths=0.6, alpha=0.5)
ax.add_collection(lc)

ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=35, zorder=3, label="Nodes")

if query_pt is not None:
    ax.scatter(query_pt[0], query_pt[1], c='crimson', marker='*', s=250, label='Query Point', zorder=5)
    
    if search_path:
        path_pts = [points[idx] for idx in search_path]
        px, py = zip(*path_pts)
        
        ax.plot(px, py, color='crimson', linestyle='-', linewidth=2, marker='o', markersize=5, zorder=4, label='Search Path')
        
        ax.scatter(path_pts[0][0], path_pts[0][1], c='darkorange', s=90, zorder=4, label='Start Node')
        ax.scatter(path_pts[-1][0], path_pts[-1][1], c='limegreen', s=100, zorder=5, label='Final Node')

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.4)

st.pyplot(fig)

if search_path and query_pt is not None:
    st.subheader("Search Results:")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Steps", f"{len(search_path)}")
    
    final_node_idx = search_path[-1]
    final_dist = distance.euclidean(query_pt, points[final_node_idx])
    m2.metric("Final Distance", f"{final_dist:.4f}")
    m3.metric("Final Node Index", f"#{final_node_idx}")
    
    st.markdown(f"**Query Coordinates:** `[{query_pt[0]:.4f}, {query_pt[1]:.4f}]`")
    
    path_chain = " -> ".join([f"`{node}`" for node in search_path])
    st.markdown(f"**Path:** {path_chain}")