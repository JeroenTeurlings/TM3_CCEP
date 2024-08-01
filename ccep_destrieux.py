#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

implemented from:
https://stackoverflow.com/questions/69805091/how-to-create-an-interactive-brain-shaped-graph

@author: jeroen
"""
import os
import ast
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import networkx as nx
import pandas as pd

# Switches
HEMI = 'lh'  # 'lh' or 'rh' or 'both'. 'both' will combine both hemispheres faulty

#%% Define function
def mesh_properties(mesh_coords):
    """Calculate center and radius of sphere minimally containing a 3-D mesh

    Parameters
    ----------
    mesh_coords : tuple
        3-tuple with x-, y-, and z-coordinates (respectively) of 3-D mesh vertices
    """

    radii = []
    center = []

    for coords in mesh_coords:
        c_max = max(c for c in coords)
        c_min = min(c for c in coords)
        center.append((c_max + c_min) / 2)

        radius = (c_max - c_min) / 2
        radii.append(radius)

    return(center, max(radii))

#%% Make graph
if HEMI == 'lh':
    base_folder = 'C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/class_3.4/left/'
elif HEMI == 'rh':
    base_folder = 'C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/class_3.4/right/'
elif HEMI == 'both':
    base_folder = 'C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/class_3.4/both/'
else:
    raise ValueError('HEMI must be either "lh", "rh", or "both"')

# Make list of subjects based on folder names in base_folder starting with sub-
subjects = [f for f in os.listdir(base_folder) if f.startswith('sub-')]
print("Including subjects: ", subjects)

# total_output = pd.DataFrame()
edge_unique = []
mean_coordinates = {}
mean_coords = {}
# Initialize a dictionary to hold lists of coordinates for each label
coord_lists = {label: {'x': [], 'y': [], 'z': []} for label in mean_coords.keys()}

for subject in subjects:
    folder = os.path.join(base_folder, subject, 'output')
    f = f'{subject}_output.tsv'

    # Load subject_output file
    graph_data = pd.read_csv(os.path.join(folder, f), sep='\t')
    graph_data = graph_data[graph_data['label'] == 0] # only use label 0

    # Create a graph
    G = nx.Graph()

    # Create nodes from unique entries in destrieux_stim and destrieux_rec columns
    G.add_nodes_from(graph_data['destrieux_stim'].unique())
    G.add_nodes_from(graph_data['destrieux_rec'].unique())

    # Create edges from combinations of destrieux_stim and destrieux_rec columns
    G.add_edges_from(graph_data[['destrieux_stim', 'destrieux_rec']].values)

    edge_unique.append(list(G.edges))

    # Convert string representations of lists into actual lists
    graph_data['xyz_stim'] = graph_data['xyz_stim'].apply(ast.literal_eval)
    graph_data['xyz_rec'] = graph_data['xyz_rec'].apply(ast.literal_eval)

    # Split into x, y, z
    graph_data[['x_stim', 'y_stim', 'z_stim']] = pd.DataFrame(graph_data['xyz_stim'].to_list(),
                                                              index=graph_data.index)
    graph_data[['x_rec', 'y_rec', 'z_rec']] = pd.DataFrame(graph_data['xyz_rec'].to_list(),
                                                           index=graph_data.index)

    # Concatenate 'stim' and 'rec' columns
    destrieux = pd.concat([graph_data['destrieux_stim'], graph_data['destrieux_rec']],
                          ignore_index=True)
    x = pd.concat([graph_data['x_stim'], graph_data['x_rec']], ignore_index=True)
    y = pd.concat([graph_data['y_stim'], graph_data['y_rec']], ignore_index=True)
    z = pd.concat([graph_data['z_stim'], graph_data['z_rec']], ignore_index=True)

    # Create a new DataFrame
    combined_data = pd.DataFrame({
        'destrieux': destrieux,
        'x': x,
        'y': y,
        'z': z
    })

    # Calculate the mean coordinates for each unique destrieux label
    mean_coords = combined_data.groupby('destrieux')[['x', 'y', 'z']].mean().to_dict('index')

    # Append the coordinates to the lists
    for label, coord in mean_coords.items():
        for k, v in coord.items():
            if label not in coord_lists:
                coord_lists[label] = {'x': [], 'y': [], 'z': []}
            coord_lists[label][k].append(v)

# Calculate the averages
mean_coordinates = {label: {k: sum(v) / len(v) for k, v in coords.items()}
                    for label, coords in coord_lists.items()}

# Add missing labels to the dictionary
for i in range(0, 75):
    if i not in mean_coordinates:
        # Prepare new entry
        mean_coordinates[i] = {'x': 0, 'y': 0, 'z': 0}

# Sort the dictionary by the keys
mean_coordinates = dict(sorted(mean_coordinates.items()))

G_total = nx.Graph()
for subject in edge_unique:
    for index in subject:
        # Count occurence of entry (x,y) in both lists
        COUNT_TOTAL = 0
        for i in edge_unique:
            if index in i:
                COUNT_TOTAL += 1

        # check for every entry if the first element was 44. If one is found, go up one loop
        COUNT_STIM = 0
        for i in edge_unique:
            for j in i:
                if j[0] == index[0]:
                    COUNT_STIM += 1
                    break

        weight = COUNT_TOTAL / COUNT_STIM if COUNT_STIM != 0 else 0

        # Add edge to total graph with weigth
        G_total.add_edge(index[0], index[1], weight=weight)

G_total.remove_edges_from(nx.selfloop_edges(G_total))

# Check if every number is included in nodes up until 73
for i in range(1, 74):
    if i not in G_total.nodes:
        # Add node to graph
        G_total.add_node(i)

G = nx.Graph()
G.add_nodes_from(sorted(G_total.nodes(data=True)))
G.add_edges_from(G_total.edges(data=True))

#%% Download and prepare dataset from BrainNet repo
coords = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=1, max_rows=53469)
x, y, z = coords.T

triangles = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=53471, dtype=int)
triangles_zero_offset = triangles - 1
i, j, k = triangles_zero_offset.T

# Generate 3D mesh.  Simply replace with 'fig = go.Figure()' or turn opacity to zero if seeing brain mesh is not desired.
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                 i=i, j=j, k=k,
                                 color='lightpink', opacity=0.25, name='', showscale=False, hoverinfo='none')])

#%% Visualize network
# Get the positions of the nodes from the dictionary
nodes_x = [v['x'] for v in mean_coordinates.values()]
nodes_y = [v['y'] for v in mean_coordinates.values()]
nodes_z = [v['z'] for v in mean_coordinates.values()]

# edge_x = []
# edge_y = []
# edge_z = []

# for s, t in G.edges():
#     edge_x += [mean_coordinates[s]['x'], mean_coordinates[t]['x']]
#     edge_y += [mean_coordinates[s]['y'], mean_coordinates[t]['y']]
#     edge_z += [mean_coordinates[s]['z'], mean_coordinates[t]['z']]


# # Decide some more meaningful logic for coloring certain nodes.
# Currently the squared distance from the mesh point at index 42.
# node_colors = []
# for node in G.nodes():
#     if np.sum((pos_brain[node] - coords[42]) ** 2) < 1000:
#         node_colors.append('red')
#     else:
#         node_colors.append('gray')

# Add node plotly trace
fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z,
                        #    text=labels,
                           mode='markers',
                           hoverinfo='text',
                           name='Nodes',
                           marker=dict(
                                       size=5,
                                    #    color=node_colors
                                      )
                           ))

# Add edge plotly trace. Comment out or turn opacity to zero if not desired.
for s, t, data in G.edges(data=True):
    if s in mean_coordinates and t in mean_coordinates:
        weight = data['weight']  # assuming 'weight' is between 0 and 1
        fig.add_trace(go.Scatter3d(x=[mean_coordinates[s]['x'], mean_coordinates[t]['x']],
                                   y=[mean_coordinates[s]['y'], mean_coordinates[t]['y']],
                                   z=[mean_coordinates[s]['z'], mean_coordinates[t]['z']],
                                   mode='lines',
                                   hoverinfo='none',
                                   name='Edges',
                                   opacity=weight,
                                   line=dict(color='black',
                                             width=3)
                                   ))

# Make axes invisible
fig.update_scenes(xaxis_visible=False,
                  yaxis_visible=False,
                  zaxis_visible=False)

# Manually adjust size of figure
fig.update_layout(autosize=False,
                  width=1200,
                  height=1200)

#%% Show figure
#fig.show()
plot(fig, auto_open=True)