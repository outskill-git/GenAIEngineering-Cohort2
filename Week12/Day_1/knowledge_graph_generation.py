import networkx as nx
import matplotlib.pyplot as plt

# Sample data pulled from above

data = [{"Rank": "1", "Title": "Guardians of the Galaxy", "Genre": "Action,Adventure,Sci-Fi", "Director": "James Gunn", "Actors": "Chris Pratt, Vin Diesel, Bradley Cooper, Zoe Saldana", "Rating": "8.1"},
        {"Rank": "2", "Title": "Prometheus", "Genre": "Adventure,Mystery,Sci-Fi", "Director": "Ridley Scott", "Actors": "Noomi Rapace, Logan Marshall-Green, Michael Fassbender, Charlize Theron", "Rating": "7"},
        # ... replicate for all 200 records (reduced example size due to space limitation)
       ]

# Create a graph
G = nx.Graph()

# Add nodes and edges
for record in data:
    movie = record['Title']
    director = record['Director']
    genres = record['Genre'].split(',')
    actors = record['Actors'].split(', ')
    rating = float(record['Rating'])
    
    G.add_node(movie, type='movie')
    G.add_node(director, type='director')
    G.add_edge(movie, director, weight=rating)
    
    for genre in genres:
        G.add_node(genre, type='genre')
        G.add_edge(movie, genre, weight=rating)

    for actor in actors:
        G.add_node(actor, type='actor')
        G.add_edge(movie, actor, weight=rating)

# Define node colors based on type
color_map = {
    'movie': 'lightblue',
    'director': 'orange',
    'genre': 'green',
    'actor': 'purple'
}

# Get colors for each node in the graph
colors = [color_map[G.nodes[node]['type']] for node in G.nodes]

# Draw the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # for reproducible layout
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=50, font_size=6, edge_color='black', alpha=0.7)
plt.title('IMDB Movies Knowledge Graph (Sample Data)')
plt.savefig('knowledge_graph.png')
plt.show()