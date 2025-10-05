import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('imdb.csv')
data = data.head(200)  # Use only the first 200 records

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
for index, row in data.iterrows():
    movie_title = row['Title']
    director = row['Director']
    genres = row['Genre'].split(',')
    actors = row['Actors'].split(',')
    rating = float(row['Rating'] if pd.notnull(row['Rating']) else 0)
    revenue = float(row['Revenue (Millions)'] if pd.notnull(row['Revenue (Millions)']) else 0)
    # Using rating as the weight for now
    weight = rating if rating > 0 else revenue
    
    # Add movie node
    G.add_node(movie_title, type='movie')
    
    # Add and connect director node
    G.add_node(director, type='director')
    G.add_edge(director, movie_title, weight=weight)
    
    # Add and connect genre nodes
    for genre in genres:
        genre = genre.strip()
        G.add_node(genre, type='genre')
        G.add_edge(movie_title, genre, weight=weight)
        
    # Add and connect actor nodes
    for actor in actors:
        actor = actor.strip()
        G.add_node(actor, type='actor')
        G.add_edge(movie_title, actor, weight=weight)

# Save the graph as GraphML
nx.write_graphml(G, 'movie_kg.graphml')

# Visualize the graph
plt.figure(figsize=(12, 12))

# Using spring layout for better visual separation
pos = nx.spring_layout(G, k=0.1, iterations=20)
options = {
    "with_labels": True,
    "node_color": "skyblue",
    "node_size": 50,
    "font_size": 8,
    "width": 0.5,
}

nx.draw(G, pos, **options)

# Save visualization
plt.savefig('knowledge_graph.png')
plt.show()