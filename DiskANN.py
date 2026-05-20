import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def greedy_search(graph, data, start_node, query_point, k, L):
    visited = set()
    candidates = [(get_distance(data[start_node], query_point), start_node)]
    path = [] 
    
    while True:
        unvisited = [c for c in candidates if c[1] not in visited]
        if not unvisited:
            break
            
        unvisited.sort(key=lambda x: x[0])
        current_node = unvisited[0][1]
        
        path.append(current_node)
        visited.add(current_node)
        
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                dist = get_distance(data[neighbor], query_point)
                if not any(c[1] == neighbor for c in candidates):
                    candidates.append((dist, neighbor))
        
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:L]
        
    return [c[1] for c in candidates[:k]], visited, path

def robust_prune(graph, data, p, V, alpha, R):
    candidates_set = set(V) | set(graph[p])
    if p in candidates_set:
        candidates_set.remove(p)
        
    graph[p] = [] 
    
    while candidates_set:
        p_star = min(candidates_set, key=lambda p_prime: get_distance(data[p], data[p_prime]))
        graph[p].append(p_star)
        
        if len(graph[p]) == R:
            break
            
        to_remove = set()
        for p_prime in candidates_set:
            if alpha * get_distance(data[p_star], data[p_prime]) <= get_distance(data[p], data[p_prime]):
                to_remove.add(p_prime)
                
        candidates_set -= to_remove

def build_vamana(data, alpha, L, R, two_passes=True):
    n = len(data)
    graph = {i: random.sample([x for x in range(n) if x != i], min(R, n-1)) for i in range(n)}
    
    centroid = np.mean(data, axis=0)
    medoid = min(range(n), key=lambda i: get_distance(data[i], centroid))
    
    passes = 2 if two_passes else 1
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for pass_num in range(passes):
        current_alpha = 1.0 if (two_passes and pass_num == 0) else alpha
        permutation = list(range(n))
        random.shuffle(permutation)
        
        for idx, p in enumerate(permutation):
            status_text.text(f"Pass {pass_num + 1}/{passes}: Processing point {idx + 1}/{n}")
            progress_bar.progress((pass_num * n + idx + 1) / (passes * n))
            
            _, V, _ = greedy_search(graph, data, medoid, data[p], 1, L)
            robust_prune(graph, data, p, V, current_alpha, R)
            
            for j in graph[p]:
                if len(graph[j]) + 1 > R:
                    robust_prune(graph, data, j, graph[j] + [p], current_alpha, R)
                else:
                    if p not in graph[j]:
                        graph[j].append(p)
                        
    progress_bar.empty()
    status_text.empty()
    return graph, medoid

st.set_page_config(page_title="DiskANN (Vamana) Simulator", layout="wide")
st.title("Vamana Graph Simulator with Query Routing")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("1. Graph Construction")
    N = st.slider("Number of points (N)", 50, 500, 200, 50)
    R = st.slider("Maximum degree (R)", 5, 50, 15, 1)
    L = st.slider("Search list size (L)", 10, 100, 30, 5)
    alpha = st.slider("Distance threshold (Alpha)", 1.0, 2.0, 1.2, 0.1)
    two_passes = st.checkbox("Two passes", value=True)
    
    generate_btn = st.button("Build Graph", type="primary", use_container_width=True)
    
    st.divider()
    
    st.header("2. Graph Search")
    search_L = st.slider("Search beam width (Search L)", 1, 50, 15, 1, help="Number of candidates the algorithm keeps in memory during search.")
    search_btn = st.button("Generate query and find path", type="secondary", use_container_width=True, disabled="graph" not in st.session_state)

with col2:
    if generate_btn:
        np.random.seed()
        st.session_state.data = np.random.rand(N, 2)
        with st.spinner('Building Vamana graph...'):
            st.session_state.graph, st.session_state.medoid = build_vamana(st.session_state.data, alpha, L, R, two_passes)
        
        if "query" in st.session_state:
            del st.session_state.query
            del st.session_state.search_path
            del st.session_state.found_nns

    if search_btn and "graph" in st.session_state:
        st.session_state.query = np.random.rand(2)
        found_nns, visited, path = greedy_search(
            st.session_state.graph, 
            st.session_state.data, 
            st.session_state.medoid, 
            st.session_state.query, 
            k=1, 
            L=search_L
        )
        st.session_state.search_path = path
        st.session_state.found_nns = found_nns

    if "graph" in st.session_state and st.session_state.graph is not None:
        data = st.session_state.data
        graph = st.session_state.graph
        medoid = st.session_state.medoid
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i in range(len(data)):
            for j in graph[i]:
                ax.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], 
                        color='lightgray', alpha=0.3, linewidth=0.5, zorder=1)
        ax.scatter(data[:, 0], data[:, 1], c='gray', s=10, zorder=2)
        
        if "query" in st.session_state:
            query = st.session_state.query
            path = st.session_state.search_path
            
            ax.scatter(query[0], query[1], c='red', marker='*', s=300, edgecolor='black', zorder=10, label='Query point')
            
            ax.scatter(data[medoid, 0], data[medoid, 1], c='green', marker='s', s=100, edgecolor='black', zorder=9, label='Entry point (Medoid)')
            
            if len(path) > 1:
                path_coords = data[path]
                ax.plot(path_coords[:, 0], path_coords[:, 1], color='orange', linewidth=2.5, alpha=0.8, zorder=5, label='Search path')
                ax.scatter(path_coords[:, 0], path_coords[:, 1], c='orange', s=40, zorder=6)
            elif len(path) == 1:
                ax.scatter(data[path[0], 0], data[path[0], 1], c='orange', s=40, zorder=6)
                
            best_node = st.session_state.found_nns[0]
            ax.scatter(data[best_node, 0], data[best_node, 1], c='magenta', s=120, edgecolor='black', zorder=8, label='Found neighbor')
            
            ax.legend(loc='upper right')
            ax.set_title(f"Search completed. Hops: {len(path)}", fontsize=16)
        else:
            ax.set_title(f"Vamana Graph (N={N}, R={R}, Alpha={alpha})", fontsize=16)
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        st.pyplot(fig)
    else:
        st.info("Configure parameters on the left and click 'Build Graph', then you can test the search.")