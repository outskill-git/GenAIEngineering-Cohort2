import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
filename = 'imdb_200.csv'
data = pd.read_csv(filename)

# Create a new graph
G = nx.Graph()

# Iterate over each row in the DataFrame
for _, row in data.iterrows():
    movie = row['Title']
    director = row['Director']
    genres = str(row['Genre']).split(',')
    actors = str(row['Actors']).split(', ')
    rating = row['Rating']

    # Add nodes for movies, directors, genres, and actors
    G.add_node(movie, node_type='movie', rating=rating)
    G.add_node(director, node_type='director')
    
    for genre in genres:
        G.add_node(genre, node_type='genre')    
    
    for actor in actors:
        G.add_node(actor, node_type='actor')

    # Add edges between movie and director, genres, actors
    G.add_edge(movie, director, weight=rating)
    for genre in genres:
        G.add_edge(movie, genre, weight=rating)
    for actor in actors:
        G.add_edge(movie, actor, weight=rating)

# Set node colors based on type
color_map = {
    'movie': 'lightblue',
    'director': 'orange',
    'genre': 'green',
    'actor': 'purple'
}

colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes]

# Plot the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.5, iterations=50)  # Fruchterman-Reingold layout
nx.draw(G, pos, with_labels=False, node_color=colors, node_size=50, edge_color='gray')
nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes if G.nodes[node]['node_type'] == 'movie'})
plt.title('IMDB Movie Knowledge Graph')
plt.savefig('knowledge_graph.png')
plt.show()