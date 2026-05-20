import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import networkx as nx

def get_knn_graph(data, k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    knn_graph = {}
    for i in range(len(data)):
        knn_graph[i] = indices[i][1:].tolist()
    return knn_graph

def search_on_graph(knn_graph, data, start_node, query_point, l_pool_size):
    visited = set()
    visited.add(start_node)
    
    candidates = [(np.linalg.norm(data[start_node] - query_point), start_node)]
    checked_nodes = set()
    result_path_nodes = set([start_node])
    
    while True:
        current_node = None
        for dist, node in candidates:
            if node not in checked_nodes:
                current_node = node
                break
                
        if current_node is None:
            break
            
        checked_nodes.add(current_node)
        
        for neighbor in knn_graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                result_path_nodes.add(neighbor)
                dist = np.linalg.norm(data[neighbor] - query_point)
                candidates.append((dist, neighbor))
                
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > l_pool_size:
            candidates = candidates[:l_pool_size]
            
    return list(result_path_nodes)

def build_nsg(data, k=10, m=5, l_pool=20):
    n_points = len(data)
    
    knn_graph = get_knn_graph(data, k)
    
    centroid = np.mean(data, axis=0)
    distances_to_centroid = np.linalg.norm(data - centroid, axis=1)
    nav_node = int(np.argmin(distances_to_centroid))
    
    nsg_graph = {i: [] for i in range(n_points)}
    
    for v in range(n_points):
        if v == nav_node:
            continue
            
        visited_nodes = search_on_graph(knn_graph, data, nav_node, data[v], l_pool)
        
        E_set = set(visited_nodes)
        E_set.update(knn_graph[v])
        if v in E_set:
            E_set.remove(v)
            
        E_list = list(E_set)
        E_distances = [np.linalg.norm(data[p] - data[v]) for p in E_list]
        sorted_indices = np.argsort(E_distances)
        E_sorted = [E_list[i] for i in sorted_indices]
        
        R = []
        if E_sorted:
            p0 = E_sorted[0]
            R.append(p0)
            E_sorted = E_sorted[1:]
            
        for p in E_sorted:
            if len(R) >= m:
                break
                
            conflict = False
            for r in R:
                dist_pr = np.linalg.norm(data[p] - data[r])
                dist_pv = np.linalg.norm(data[p] - data[v])
                if dist_pr < dist_pv:
                    conflict = True
                    break
                    
            if not conflict:
                R.append(p)
                
        nsg_graph[v] = R
        
    nsg_graph[nav_node] = knn_graph[nav_node][:m]

    nx_graph = nx.DiGraph(nsg_graph)
    dfs_edges = list(nx.dfs_edges(nx_graph, source=nav_node))
    dfs_nodes = set([nav_node])
    for u, v in dfs_edges:
        dfs_nodes.add(v)
        
    unreached = set(range(n_points)) - dfs_nodes
    
    for u in unreached:
        best_dist = float('inf')
        best_target = None
        for in_tree_node in dfs_nodes:
            dist = np.linalg.norm(data[u] - data[in_tree_node])
            if dist < best_dist:
                best_dist = dist
                best_target = in_tree_node
        if best_target is not None:
            nsg_graph[u].append(best_target)
            dfs_nodes.add(u)

    return nsg_graph, knn_graph, nav_node

def find_greedy_path(graph, data, start_idx, target_idx):
    target_pt = data[target_idx]
    path = [start_idx]
    current = start_idx
    visited = set([start_idx])
    
    while current != target_idx:
        neighbors = graph.get(current, [])
        if not neighbors:
            break
        
        best_dist = np.linalg.norm(data[current] - target_pt)
        next_node = None
        
        for neighbor in neighbors:
            dist = np.linalg.norm(data[neighbor] - target_pt)
            if dist < best_dist:
                best_dist = dist
                next_node = neighbor
                
        if next_node is None or next_node in visited:
            break 
            
        visited.add(next_node)
        path.append(next_node)
        current = next_node
        
    return path

st.set_page_config(page_title="NSG Algorithm Simulator", layout="wide")

st.title("Simulator: Navigating Spreading-out Graph (NSG)")
st.markdown("""
Visualization of the **NSG** algorithm. Hover over any point to see its ID.
Specify the start and finish points in the left menu to view the greedy routing path.
* 🔴 Red point: Navigating Node (default entry point)
* 🟢 Green point: Search start
* 🟣 Purple point: Finish (target node)
* 🟠 Orange line: Algorithm path
""")

st.sidebar.header("Graph Parameters")
N = st.sidebar.slider("Number of points (N)", min_value=50, max_value=500, value=150, step=10)
k = st.sidebar.slider("Base kNN degree (k)", min_value=3, max_value=30, value=15, step=1)
m = st.sidebar.slider("Max out-degree of NSG (m)", min_value=2, max_value=15, value=4, step=1)
l_pool = st.sidebar.slider("Candidate pool size (l)", min_value=5, max_value=50, value=20, step=1)

@st.cache_data
def generate_data(n_points):
    np.random.seed(42)
    return np.random.rand(n_points, 2) * 100

data = generate_data(N)

with st.spinner("Building graphs..."):
    nsg_graph, knn_graph, nav_node = build_nsg(data, k=k, m=m, l_pool=l_pool)

knn_edges_count = sum(len(v) for v in knn_graph.values())
nsg_edges_count = sum(len(v) for v in nsg_graph.values())

st.sidebar.markdown("---")
st.sidebar.subheader("Path Search")
st.sidebar.markdown("Specify node IDs for path search (from `0` to `" + str(N-1) + "`)")

start_node = st.sidebar.number_input("Start (ID)", min_value=0, max_value=N-1, value=int(nav_node))
target_node = st.sidebar.number_input("Finish (ID)", min_value=0, max_value=N-1, value=N-1)

path_nsg = find_greedy_path(nsg_graph, data, start_node, target_node)
path_knn = find_greedy_path(knn_graph, data, start_node, target_node)

st.sidebar.markdown("---")
st.sidebar.subheader("Statistics")
st.sidebar.write(f"**kNN Edges:** {knn_edges_count}")
st.sidebar.write(f"**NSG Edges:** {nsg_edges_count}")

if path_nsg[-1] != target_node:
    st.sidebar.warning(f"⚠️ NSG stuck at a local minimum (node {path_nsg[-1]}). Try increasing `k` or `m`.")
else:
    st.sidebar.success(f"✅ NSG reached the target in {len(path_nsg)-1} steps.")

def plot_graph(data, graph_dict, title, nav_node, start_node, target_node, path):
    edge_x = []
    edge_y = []
    
    for u, neighbors in graph_dict.items():
        x0, y0 = data[u]
        for v in neighbors:
            x1, y1 = data[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(200, 200, 200, 0.7)'),
        hoverinfo='none',
        mode='lines'
    )

    node_colors = ['#1f77b4'] * len(data)
    node_sizes = [6] * len(data)
    
    node_colors[nav_node] = '#d62728' 
    node_sizes[nav_node] = 10
    
    node_colors[start_node] = '#2ca02c' 
    node_sizes[start_node] = 14
    
    node_colors[target_node] = '#9467bd' 
    node_sizes[target_node] = 14

    hover_texts = [f"Node ID: {i}" for i in range(len(data))]

    nodes_trace = go.Scatter(
        x=data[:, 0], y=data[:, 1],
        mode='markers',
        hovertext=hover_texts,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=1,
            line_color='white'
        )
    )

    traces = [edges_trace, nodes_trace]

    if path:
        path_x = [data[n][0] for n in path]
        path_y = [data[n][1] for n in path]
        
        path_trace = go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(width=3, color='#ff7f0e'), 
            marker=dict(size=8, color='#ff7f0e'),
            hovertext=[f"Step {i}: Node {n}" for i, n in enumerate(path)],
            hoverinfo='text'
        )
        traces.append(path_trace)

    fig = go.Figure(data=traces,
                    layout=go.Layout(
                        title=dict(text=f"<b>{title}</b>", font=dict(size=16)), 
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    )
                   )
    return fig

col1, col2 = st.columns(2)
with col1:
    fig_knn = plot_graph(data, knn_graph, f"Base kNN Graph (k={k})", nav_node, start_node, target_node, path_knn)
    st.plotly_chart(fig_knn, use_container_width=True)
with col2:
    fig_nsg = plot_graph(data, nsg_graph, f"NSG Graph (max_degree m={m})", nav_node, start_node, target_node, path_nsg)
    st.plotly_chart(fig_nsg, use_container_width=True)