import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import random
import heapq

class HNSW:
    def __init__(self, M=5, efConstruction=30, Mmax=None, Mmax0=None, m_L=None):
        self.M = M
        self.Mmax = Mmax if Mmax else M
        self.Mmax0 = Mmax0 if Mmax0 else M * 2
        self.efConstruction = efConstruction
        self.m_L = m_L if m_L is not None else 1.0 / math.log(M)
        
        self.data = {}
        self.graphs = {}
        self.enter_point = None
        self.max_layer = -1
        self.current_id = 0

    def _distance(self, v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def _get_random_level(self):
        return math.floor(-math.log(random.uniform(0.0001, 1.0)) * self.m_L)

    def search_layer(self, q, ep, ef, layer):
        visited = {ep}
        candidates = [(self._distance(q, self.data[ep]), ep)]
        W = [(self._distance(q, self.data[ep]), ep)]
        heapq.heapify(candidates)
        heapq.heapify(W)

        parent = {ep: None}

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            furthest_W_dist = max(W, key=lambda x: x[0])[0]
            
            if dist_c > furthest_W_dist:
                break

            for e in self.graphs[layer].get(c, []):
                if e not in visited:
                    visited.add(e)
                    parent[e] = c 
                    furthest_W_dist = max(W, key=lambda x: x[0])[0]
                    dist_e = self._distance(q, self.data[e])
                    
                    if dist_e < furthest_W_dist or len(W) < ef:
                        heapq.heappush(candidates, (dist_e, e))
                        heapq.heappush(W, (dist_e, e))
                        if len(W) > ef:
                            W.remove(max(W, key=lambda x: x[0]))
                            heapq.heapify(W)
                            
        best_node = min(W, key=lambda x: x[0])[1]
        path_nodes = []
        curr = best_node
        while curr is not None:
            path_nodes.append(curr)
            curr = parent[curr]
        path_nodes.reverse() 
        
        hops = []
        for i in range(len(path_nodes)-1):
            hops.append({
                'type': 'hop', 
                'from': path_nodes[i], 
                'to': path_nodes[i+1], 
                'layer': layer
            })

        return W, hops

    def insert(self, q):
        node_id = self.current_id
        self.data[node_id] = q
        self.current_id += 1

        ep = self.enter_point
        L = self.max_layer
        l = self._get_random_level()

        for layer in range(l + 1):
            if layer not in self.graphs:
                self.graphs[layer] = {}
            self.graphs[layer][node_id] = []

        if ep is not None:
            for layer in range(L, l, -1):
                W, _ = self.search_layer(q, ep, ef=1, layer=layer)
                ep = min(W, key=lambda x: x[0])[1]

            for layer in range(min(L, l), -1, -1):
                W, _ = self.search_layer(q, ep, self.efConstruction, layer)
                neighbors = [item[1] for item in heapq.nsmallest(self.M, W)]
                
                for n in neighbors:
                    self.graphs[layer][node_id].append(n)
                    self.graphs[layer][n].append(node_id)
                    M_limit = self.Mmax0 if layer == 0 else self.Mmax
                    if len(self.graphs[layer][n]) > M_limit:
                        self.graphs[layer][n].sort(key=lambda x: self._distance(self.data[n], self.data[x]))
                        self.graphs[layer][n] = self.graphs[layer][n][:M_limit]
                ep = min(W, key=lambda x: x[0])[1]

        if l > L:
            self.max_layer = l
            self.enter_point = node_id

    def knn_search(self, q, K, ef):
        if self.enter_point is None:
            return [], []
            
        full_path = []
        ep = self.enter_point
        L = self.max_layer

        for layer in range(L, 0, -1):
            W, hops = self.search_layer(q, ep, ef=1, layer=layer)
            full_path.extend(hops)
            
            next_ep = min(W, key=lambda x: x[0])[1]
            full_path.append({
                'type': 'drop', 
                'node': next_ep, 
                'from_layer': layer, 
                'to_layer': layer-1
            })
            ep = next_ep

        W, hops = self.search_layer(q, ep, ef, layer=0)
        full_path.extend(hops)
        
        return heapq.nsmallest(K, W), full_path

def draw_hnsw_3d(index, query_point=None, neighbors=None, search_path=None):
    fig = go.Figure()

    edge_x, edge_y, edge_z = [], [], []
    for layer, graph in index.graphs.items():
        for u, neighbors_list in graph.items():
            for v in neighbors_list:
                edge_x.extend([index.data[u][0], index.data[v][0], None])
                edge_y.extend([index.data[u][1], index.data[v][1], None])
                edge_z.extend([layer, layer, None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.25)', width=1),
        hoverinfo='none',
        name='Graph Edges'
    ))

    vert_x, vert_y, vert_z = [], [], []
    for u in index.data:
        max_l = max([l for l in index.graphs if u in index.graphs[l]], default=-1)
        if max_l > 0:
            vert_x.extend([index.data[u][0], index.data[u][0], None])
            vert_y.extend([index.data[u][1], index.data[u][1], None])
            vert_z.extend([0, max_l, None])

    fig.add_trace(go.Scatter3d(
        x=vert_x, y=vert_y, z=vert_z,
        mode='lines',
        line=dict(color='rgba(200, 200, 200, 0.2)', width=2, dash='dash'),
        hoverinfo='none',
        name='Node Projections'
    ))

    if search_path:
        path_hop_x, path_hop_y, path_hop_z = [], [], []
        path_drop_x, path_drop_y, path_drop_z = [], [], []
        
        ep_u = index.enter_point
        ep_l = index.max_layer
        fig.add_trace(go.Scatter3d(
            x=[index.data[ep_u][0]], y=[index.data[ep_u][1]], z=[ep_l],
            mode='markers',
            marker=dict(size=10, color='yellow', symbol='diamond'),
            name='Entry Point'
        ))

        for step in search_path:
            if step['type'] == 'hop':
                u, v, l = step['from'], step['to'], step['layer']
                path_hop_x.extend([index.data[u][0], index.data[v][0], None])
                path_hop_y.extend([index.data[u][1], index.data[v][1], None])
                path_hop_z.extend([l, l, None])
            elif step['type'] == 'drop':
                u = step['node']
                l_from, l_to = step['from_layer'], step['to_layer']
                path_drop_x.extend([index.data[u][0], index.data[u][0], None])
                path_drop_y.extend([index.data[u][1], index.data[u][1], None])
                path_drop_z.extend([l_from, l_to, None])

        if path_hop_x:
            fig.add_trace(go.Scatter3d(
                x=path_hop_x, y=path_hop_y, z=path_hop_z,
                mode='lines',
                line=dict(color='#FFD700', width=6),
                name='Search Path (Layer)'
            ))
        
        if path_drop_x:
            fig.add_trace(go.Scatter3d(
                x=path_drop_x, y=path_drop_y, z=path_drop_z,
                mode='lines',
                line=dict(color='#FFD700', width=6, dash='dot'),
                name='Drop between layers'
            ))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for layer in index.graphs.keys():
        node_x = [index.data[n][0] for n in index.graphs[layer]]
        node_y = [index.data[n][1] for n in index.graphs[layer]]
        node_z = [layer] * len(node_x)
        
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(size=6, color=colors[layer % len(colors)], opacity=0.8),
            name=f'Nodes (Layer {layer})'
        ))

    if query_point:
        fig.add_trace(go.Scatter3d(
            x=[query_point[0]], y=[query_point[1]], z=[0],
            mode='markers',
            marker=dict(size=12, color='red', symbol='square'),
            name='Query Point'
        ))
    
    if neighbors:
        nx = [index.data[n][0] for _, n in neighbors]
        ny = [index.data[n][1] for _, n in neighbors]
        nz = [0] * len(nx)
        fig.add_trace(go.Scatter3d(
            x=nx, y=ny, z=nz,
            mode='markers',
            marker=dict(size=10, color='magenta', symbol='circle-open', line=dict(width=4)),
            name='Found Neighbors'
        ))

    fig.update_layout(
        title="3D Interactive HNSW Simulator",
        scene=dict(
            xaxis=dict(title='X', range=[-55, 55], showbackground=False),
            yaxis=dict(title='Y', range=[-55, 55], showbackground=False),
            zaxis=dict(title='Layer', range=[0, max(2, index.max_layer + 1)], 
                       dtick=1, showbackground=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
        plot_bgcolor='black',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

st.set_page_config(layout="wide", page_title="HNSW Simulator")

st.title("HNSW Graph Simulator 🌐")
st.markdown("Change algorithm parameters in the left panel, generate the graph, and test K-nearest neighbors search.")

if 'hnsw_index' not in st.session_state:
    st.session_state.hnsw_index = None

with st.sidebar:
    st.header("Build Parameters")
    num_points = st.slider("Number of points", 10, 300, 138)
    M = st.slider("M (Edges per node)", 2, 15, 5)
    mL = st.slider("mL (Probability multiplier)", 0.1, 1.5, 0.60, 0.1)
    ef_construction = st.slider("efConstruction", 10, 100, 52)
    
    if st.button("Generate Graph"):
        index = HNSW(M=M, efConstruction=ef_construction, m_L=mL)
        np.random.seed(42) 
        points = np.random.uniform(-50, 50, (num_points, 2))
        
        progress = st.progress(0)
        for i, p in enumerate(points):
            index.insert(p.tolist())
            progress.progress((i + 1) / num_points)
            
        st.session_state.hnsw_index = index

    st.header("Search Parameters")
    k_neighbors = st.slider("K (Number of neighbors)", 1, 10, 4)
    ef_search = st.slider("ef (Search pool size)", 1, 50, 18)
    
    do_search = st.button("Random Query")

if st.session_state.hnsw_index is not None:
    index = st.session_state.hnsw_index
    query_pt = None
    results = None
    search_path = None
    
    if do_search:
        query_pt = np.random.uniform(-50, 50, 2).tolist()
        results, search_path = index.knn_search(query_pt, K=k_neighbors, ef=ef_search)
        
    fig = draw_hnsw_3d(index, query_point=query_pt, neighbors=results, search_path=search_path)
    st.plotly_chart(fig, use_container_width=True)
    
    if results:
        st.write("### Search Results:")
        for dist, n_id in results:
            coord = index.data[n_id]
            st.write(f"- Node ID: {n_id}, Distance: **{dist:.2f}**, Coordinates: ({coord[0]:.1f}, {coord[1]:.1f})")
else:
    st.info("👈 Click 'Generate Graph' in the left menu to start.")