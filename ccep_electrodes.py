#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implemented from:
https://stackoverflow.com/questions/69805091/how-to-create-an-interactive-brain-shaped-graph
"""
import os
import ast
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Switches
HEMI = 'lh'  # 'lh' or 'rh' or 'both'. 'both' will combine both hemispheres faulty
NODES = True  # True or False. If True, nodes will be plotted. If False, only edges will be plotted
EDGES = True  # True or False. If True, edges will be plotted. If False, only nodes will be plotted
MINIMUM_LENGTH = 80  # Minimum length of edges to be plotted
MAXIMUM_LENGTH = 500 # Maximum length of edges to be plotted

# Define mesh function
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

# Define locations of electrode data to be plotted. Results from ccep_merger.py
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

edge_unique = []
mean_coordinates = {}
# Empty DataFrame to store all data
total_data = pd.DataFrame()
# Initialize a dictionary to hold lists of coordinates for each label
coord_lists = {label: {'x': [], 'y': [], 'z': []} for label in mean_coordinates.keys()}

for subject in subjects:
    folder = os.path.join(base_folder, subject, 'output')
    f = f'{subject}_output.tsv'

    # Load subject_output file
    graph_data = pd.read_csv(os.path.join(folder, f), sep='\t')
    total_data = pd.concat([total_data, graph_data])

    # Create a graph
    G = nx.Graph()

    node_counts = {}

    # Create nodes from unique entries in stim_name and rec_name columns
    for node in pd.concat([graph_data['stim_name'], graph_data['rec_name']]).unique():
        if node not in node_counts:
            node_counts[node] = 0
        else:
            node_counts[node] += 1
        G.add_node(f"{node}_{node_counts[node]}")

    # Create edges from combinations of stim_name and rec_name columns
    for _, row in graph_data.iterrows():
        if row['label'] == 0:
            stim_name = f"{row['stim_name']}_{node_counts[row['stim_name']]}"
            rec_name = f"{row['rec_name']}_{node_counts[row['rec_name']]}"
            amplitude = row['amplitude']
            latency = row['latency']
            G.add_edge(stim_name, rec_name, amplitude=amplitude, latency=latency)

    edge_unique.append(list(G.edges(data=True)))  # Include edge data when appending to edge_unique

    # Convert string representations of lists into actual lists
    graph_data['xyz_stim'] = graph_data['xyz_stim'].apply(ast.literal_eval)
    graph_data['xyz_rec'] = graph_data['xyz_rec'].apply(ast.literal_eval)

    # Split into x, y, z
    graph_data[['x_stim', 'y_stim', 'z_stim']] = pd.DataFrame(graph_data['xyz_stim'].to_list(),
                                                              index=graph_data.index)
    graph_data[['x_rec', 'y_rec', 'z_rec']] = pd.DataFrame(graph_data['xyz_rec'].to_list(),
                                                           index=graph_data.index)

    # Concatenate 'stim' and 'rec' columns
    electrode = pd.concat([graph_data['stim_name'].apply(lambda x: f"{x}_{node_counts[x]}"),
                           graph_data['rec_name'].apply(lambda x: f"{x}_{node_counts[x]}")],
                          ignore_index=True)
    x = pd.concat([graph_data['x_stim'], graph_data['x_rec']], ignore_index=True)
    y = pd.concat([graph_data['y_stim'], graph_data['y_rec']], ignore_index=True)
    z = pd.concat([graph_data['z_stim'], graph_data['z_rec']], ignore_index=True)

    # Create a new DataFrame
    combined_data = pd.DataFrame({
        'electrode': electrode,
        'x': x,
        'y': y,
        'z': z
    })

    # Remove duplicates
    combined_data = combined_data.drop_duplicates(subset='electrode')

    # Append the coordinates to the lists
    for index, row in combined_data.iterrows():
        label = row['electrode']
        coord = {'x': row['x'], 'y': row['y'], 'z': row['z']}
        for k, v in coord.items():
            if label not in coord_lists:
                coord_lists[label] = {'x': [], 'y': [], 'z': []}
            coord_lists[label][k].append(v)
# Save total_data to a TSV file. Uncomment if needed
# total_data.to_csv('total_data.tsv', sep='\t', index=False)
mean_coordinates = coord_lists

G_total = nx.Graph()
for subject_edges in edge_unique:
    # Add unique edges from each subject's graph to the total graph
    G_total.add_edges_from(subject_edges)

G_total.remove_edges_from(nx.selfloop_edges(G_total))

G = nx.Graph()
G.add_nodes_from(sorted(G_total.nodes(data=True)))
G.add_edges_from(G_total.edges(data=True))  # Edge data is preserved when adding edges to G

# Download and prepare dataset from BrainNet repo for visualization
coords = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=1, max_rows=53469)
x_brain, y_brain, z_brain = coords.T

triangles = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=53471, dtype=int)
triangles_zero_offset = triangles - 1
i, j, k = triangles_zero_offset.T

# Generate 3D mesh.
fig = go.Figure(data=[go.Mesh3d(x=x_brain, y=y_brain, z=z_brain,
                                 i=i, j=j, k=k,
                                 color='lightpink', opacity=0.25, name='', showscale=False, hoverinfo='none')])

# Visualize network
# Create a set to store the nodes of the edges that meet the length criteria
nodes_to_plot = set()

print('Adding edges...')
# Add edges
edge_x = []
edge_y = []
edge_z = []
distance_dict = {}
latencies = []
edge_lengths = []

for s, t in G.edges():
    if s in mean_coordinates and t in mean_coordinates:
        # Calculate the Euclidean distance between the nodes
        distance = np.sqrt((mean_coordinates[s]['x'][0] - mean_coordinates[t]['x'][0])**2 +
                        (mean_coordinates[s]['y'][0] - mean_coordinates[t]['y'][0])**2 +
                        (mean_coordinates[s]['z'][0] - mean_coordinates[t]['z'][0])**2)
        # Add distance to distance_dict
        distance_dict[(s, t)] = distance
        # Only add the edge if its length is equal to the desired length
        if (distance > MINIMUM_LENGTH) and (distance < MAXIMUM_LENGTH):
            edge_x.extend([mean_coordinates[s]['x'][0], mean_coordinates[t]['x'][0], None])
            edge_y.extend([mean_coordinates[s]['y'][0], mean_coordinates[t]['y'][0], None])
            edge_z.extend([mean_coordinates[s]['z'][0], mean_coordinates[t]['z'][0], None])
            # Add the nodes of the edge to the set
            nodes_to_plot.add(s)
            nodes_to_plot.add(t)

            # Extract latency for the edge
            latency = G[s][t]['latency']
            # Append latency and edge length to their respective lists
            latencies.append(latency)
            edge_lengths.append(distance)
# Extract latencies to be the same format as distances
latencies = [float(l.strip('[]')) for l in latencies]

# Get the distances from the dictionary
distances = list(distance_dict.values())

# Create the histogram of distances
plt.hist(distances, bins=150)  # 'auto' determines the number of bins automatically
plt.title('Histogram of Edge Lengths')
plt.xlabel('Edge Length (mm)')
plt.ylabel('Frequency')
plt.show()

# Create histogram of latencies
plt.hist(latencies, bins=177) # 'auto' determines the number of bins automatically
plt.title('Histogram of Latencies')
plt.xlabel('Latency (s)')
plt.ylabel('Frequency')
plt.show()

# Calculate the correlation coefficient
correlation = np.corrcoef(edge_lengths, latencies)[0, 1]

# Create scatterplot of latency and edge length
plt.scatter(edge_lengths, latencies, s=1)  # s is the size of the points
plt.xlabel('Edge Length (mm)')
plt.ylabel('Latency (s)')
plt.title('Latency per Edge Length')

# Add the correlation coefficient to the plot
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top')

plt.show()

# Define your latency range
MINIMUM_LATENCY = 0.00
MAXIMUM_LATENCY = 1.00
MINIMUM_LENGTH_PLOT = 0  # Minimum length of edges to be plotted
MAXIMUM_LENGTH_PLOT = 500  # Maximum length of edges to be plotted

# # Print list of edges within the desired length and latency range. Uncomment if needed
# print('Edges within the desired length and latency range:')
# for (s, t), d in distance_dict.items():
#     latency = float(G[s][t]["latency"].strip('[]'))  # Strip brackets and convert latency to float
#     if (d > MINIMUM_LENGTH_PLOT) and (d < MAXIMUM_LENGTH_PLOT) and (latency > MINIMUM_LATENCY) and (latency < MAXIMUM_LATENCY):
#         print(f'Edge {s} - {t} has a length of {d} mm and a latency of {latency} s')

if EDGES:
    # Assuming G is your graph and it has 'amplitude' and 'latency' attributes on edges
    amplitudes = np.array([float(d['amplitude'].strip('[]')) for _, _, d in G.edges(data=True)])
    latencies = np.array([float(d['latency'].strip('[]')) for _, _, d in G.edges(data=True)])

    # Normalize amplitudes and latencies between 0 and 1
    amplitudes = (amplitudes - amplitudes.min()) / (amplitudes.max() - amplitudes.min())
    latencies = 1 - ((latencies - latencies.min()) / (latencies.max() - latencies.min()))  # Invert latencies

    # Create a list of edges with varying thickness (based on amplitude) and opacity (based on latency)
    edges = []
    for i in range(0, len(edge_x), 3):
        x = edge_x[i:i+3]
        y = edge_y[i:i+3]
        z = edge_z[i:i+3]
        amplitude = amplitudes[i // 3]  # Each amplitude corresponds to one edge, which is represented by 3 points
        latency = latencies[i // 3]  # Same for latencies
        edges.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                hoverinfo='none',
                name=f'Edge {i // 3}',
                opacity=latency,  # Opacity based on inverted latency
                line=dict(
                    color='black',
                    width=amplitude * 10  # Thickness based on normalized amplitude, scaled up for visibility
                )
            )
        )

    # Add all edges to the figure
    for edge in edges:
        fig.add_trace(edge)
    print('Amount of edges plotted:', len(edges))

print()  # To ensure the next print starts on a new line
print('Done!')

# Add only the nodes of the edges that were previously added
if NODES:
    print('Adding nodes...')
    # Get the positions of the nodes from the set
    nodes_x = [mean_coordinates[s]['x'][0] for s in nodes_to_plot]
    nodes_y = [mean_coordinates[s]['y'][0] for s in nodes_to_plot]
    nodes_z = [mean_coordinates[s]['z'][0] for s in nodes_to_plot]

    # Add node plotly trace
    fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z,
                            #    text=labels,
                            mode='markers',
                            hoverinfo='text',
                            name='Nodes',
                            marker=dict(
                                        size=2,
                                        #    color=node_colors
                                        )
                            ))

# Make axes invisible
print('Making axes invisible...')
fig.update_scenes(xaxis_visible=False,
                  yaxis_visible=False,
                  zaxis_visible=False)

# Manually adjust size of figure
print('Adjusting figure size...')
fig.update_layout(autosize=False,
                  width=1200,
                  height=1200)

#%% Show figure
fig.show()
print('Saving figure...')
plot(fig, auto_open=True)
print('Done!')
