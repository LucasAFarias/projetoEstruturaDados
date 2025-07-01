import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from collections import deque
import heapq

# Classe para representar o grafo
class MarvelGraph:
    def __init__(self):
        # Usando lista de adjacência (mais eficiente para grafos esparsos)
        self.adj_list = {}
        self.characters_info = {}
        
    def add_character(self, name, movie, is_hero=True, power_level=0):
        if name not in self.adj_list:
            self.adj_list[name] = []
            self.characters_info[name] = {
                'movies': set(),
                'is_hero': is_hero,
                'power_level': power_level
            }
        self.characters_info[name]['movies'].add(movie)
        
    def add_interaction(self, char1, char2, weight=1):
        # Adiciona interação bidirecional com peso (número de filmes juntos)
        if char1 not in self.adj_list:
            raise ValueError(f"Personagem {char1} não existe no grafo.")
        if char2 not in self.adj_list:
            raise ValueError(f"Personagem {char2} não existe no grafo.")
            
        # Verifica se já existe a aresta para incrementar o peso
        edge_exists = False
        for i, (neighbor, w) in enumerate(self.adj_list[char1]):
            if neighbor == char2:
                self.adj_list[char1][i] = (neighbor, w + weight)
                edge_exists = True
                break
                
        if not edge_exists:
            self.adj_list[char1].append((char2, weight))
            
        edge_exists = False
        for i, (neighbor, w) in enumerate(self.adj_list[char2]):
            if neighbor == char1:
                self.adj_list[char2][i] = (neighbor, w + weight)
                edge_exists = True
                break
                
        if not edge_exists:
            self.adj_list[char2].append((char1, weight))
    
    def to_networkx(self):
        G = nx.Graph()
        for node in self.adj_list:
            G.add_node(node, 
                      is_hero=self.characters_info[node]['is_hero'],
                      power_level=self.characters_info[node]['power_level'])
            for neighbor, weight in self.adj_list[node]:
                G.add_edge(node, neighbor, weight=weight)
        return G
    
    def bfs(self, start):
        """Busca em Largura para encontrar todos os personagens conectados"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor, _ in self.adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result
    
    def dijkstra(self, start):
        """Encontra os caminhos mais curtos baseados no peso das arestas (mais interações)"""
        distances = {node: float('infinity') for node in self.adj_list}
        distances[start] = 0
        heap = [(0, start)]
        
        while heap:
            current_distance, current_node = heapq.heappop(heap)
            
            if current_distance > distances[current_node]:
                continue
                
            for neighbor, weight in self.adj_list[current_node]:
                distance = current_distance + (1 / weight)  # Quanto mais interações, menor o "custo"
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))
                    
        return distances

# Criando o grafo do MCU
def create_mcu_graph():
    mcu = MarvelGraph()
    
    # Adicionando personagens principais (nome, filme de estreia, é herói, nível de poder)
    mcu.add_character("Tony Stark", "Iron Man 1", True, 8)
    mcu.add_character("Steve Rogers", "Captain America 1", True, 7)
    mcu.add_character("Thor", "Thor 1", True, 9)
    mcu.add_character("Bruce Banner", "The Incredible Hulk", True, 9)
    mcu.add_character("Natasha Romanoff", "Iron Man 2", True, 6)
    mcu.add_character("Clint Barton", "Thor 1", True, 5)
    mcu.add_character("Nick Fury", "Iron Man 1", False, 4)
    mcu.add_character("Pepper Potts", "Iron Man 1", False, 2)
    mcu.add_character("Peter Parker", "Captain America: Civil War", True, 7)
    mcu.add_character("Stephen Strange", "Doctor Strange", True, 8)
    mcu.add_character("T'Challa", "Captain America: Civil War", True, 7)
    mcu.add_character("Wanda Maximoff", "Avengers: Age of Ultron", True, 9)
    mcu.add_character("Vision", "Avengers: Age of Ultron", True, 8)
    mcu.add_character("Scott Lang", "Ant-Man", True, 5)
    mcu.add_character("Loki", "Thor 1", False, 7)
    mcu.add_character("Thanos", "Avengers 1", False, 10)
    mcu.add_character("Peter Quill", "Guardians of the Galaxy", True, 6)
    mcu.add_character("Gamora", "Guardians of the Galaxy", True, 7)
    mcu.add_character("Drax", "Guardians of the Galaxy", True, 6)
    mcu.add_character("Rocket", "Guardians of the Galaxy", True, 5)
    mcu.add_character("Groot", "Guardians of the Galaxy", True, 6)
    mcu.add_character("Carol Danvers", "Captain Marvel", True, 9)
    
    # Adicionando interações (personagem1, personagem2, número de filmes juntos)
    mcu.add_interaction("Tony Stark", "Steve Rogers", 4)
    mcu.add_interaction("Tony Stark", "Thor", 3)
    mcu.add_interaction("Tony Stark", "Bruce Banner", 3)
    mcu.add_interaction("Tony Stark", "Natasha Romanoff", 4)
    mcu.add_interaction("Tony Stark", "Clint Barton", 3)
    mcu.add_interaction("Tony Stark", "Nick Fury", 5)
    mcu.add_interaction("Tony Stark", "Pepper Potts", 6)
    mcu.add_interaction("Tony Stark", "Peter Parker", 3)
    mcu.add_interaction("Tony Stark", "Stephen Strange", 1)
    mcu.add_interaction("Tony Stark", "T'Challa", 1)
    mcu.add_interaction("Tony Stark", "Wanda Maximoff", 2)
    mcu.add_interaction("Tony Stark", "Vision", 2)
    mcu.add_interaction("Tony Stark", "Scott Lang", 1)
    mcu.add_interaction("Tony Stark", "Loki", 1)
    mcu.add_interaction("Tony Stark", "Thanos", 1)
    
    mcu.add_interaction("Steve Rogers", "Thor", 3)
    mcu.add_interaction("Steve Rogers", "Bruce Banner", 3)
    mcu.add_interaction("Steve Rogers", "Natasha Romanoff", 5)
    mcu.add_interaction("Steve Rogers", "Clint Barton", 3)
    mcu.add_interaction("Steve Rogers", "Nick Fury", 3)
    mcu.add_interaction("Steve Rogers", "Peter Parker", 2)
    mcu.add_interaction("Steve Rogers", "T'Challa", 2)
    mcu.add_interaction("Steve Rogers", "Wanda Maximoff", 2)
    mcu.add_interaction("Steve Rogers", "Vision", 2)
    mcu.add_interaction("Steve Rogers", "Scott Lang", 2)
    mcu.add_interaction("Steve Rogers", "Loki", 1)
    mcu.add_interaction("Steve Rogers", "Thanos", 1)
    
    mcu.add_interaction("Thor", "Bruce Banner", 3)
    mcu.add_interaction("Thor", "Natasha Romanoff", 2)
    mcu.add_interaction("Thor", "Clint Barton", 2)
    mcu.add_interaction("Thor", "Loki", 5)
    mcu.add_interaction("Thor", "Thanos", 2)
    mcu.add_interaction("Thor", "Peter Quill", 1)
    mcu.add_interaction("Thor", "Gamora", 1)
    mcu.add_interaction("Thor", "Drax", 1)
    mcu.add_interaction("Thor", "Rocket", 2)
    mcu.add_interaction("Thor", "Groot", 1)
    
    mcu.add_interaction("Bruce Banner", "Natasha Romanoff", 3)
    mcu.add_interaction("Bruce Banner", "Clint Barton", 2)
    mcu.add_interaction("Bruce Banner", "Thanos", 1)
    
    mcu.add_interaction("Natasha Romanoff", "Clint Barton", 5)
    mcu.add_interaction("Natasha Romanoff", "Nick Fury", 3)
    mcu.add_interaction("Natasha Romanoff", "Wanda Maximoff", 2)
    mcu.add_interaction("Natasha Romanoff", "Vision", 2)
    mcu.add_interaction("Natasha Romanoff", "Scott Lang", 2)
    mcu.add_interaction("Natasha Romanoff", "Thanos", 1)
    
    mcu.add_interaction("Peter Parker", "Stephen Strange", 1)
    mcu.add_interaction("Wanda Maximoff", "Vision", 3)
    
    mcu.add_interaction("Peter Quill", "Gamora", 3)
    mcu.add_interaction("Peter Quill", "Drax", 3)
    mcu.add_interaction("Peter Quill", "Rocket", 3)
    mcu.add_interaction("Peter Quill", "Groot", 3)
    mcu.add_interaction("Peter Quill", "Thanos", 1)
    mcu.add_interaction("Gamora", "Thanos", 2)
    
    return mcu

# Análise do grafo
def analyze_graph(G):
    print("\n=== Análise da Rede do MCU ===")
    
    # Métricas básicas
    print(f"Número de nós (personagens): {G.number_of_nodes()}")
    print(f"Número de arestas (interações): {G.number_of_edges()}")
    
    # Grau dos nós
    degrees = dict(G.degree())
    print("\nPersonagens mais conectados:")
    for node, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{node}: {degree} conexões")
    
    # Centralidade de grau
    degree_centrality = nx.degree_centrality(G)
    print("\nCentralidade de grau (importância baseada em conexões):")
    for node, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{node}: {centrality:.2f}")
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    print("\nCentralidade de intermediação (quem conecta grupos distintos):")
    for node, centrality in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{node}: {centrality:.2f}")
    
    # Componentes conectados
    print(f"\nNúmero de componentes conectados: {nx.number_connected_components(G)}")
    
    # Diâmetro do maior componente
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    print(f"Diâmetro do maior componente: {nx.diameter(subgraph)}")
    
    # Densidade
    print(f"Densidade da rede: {nx.density(G):.4f}")
    
    # Coeficiente de clustering
    avg_clustering = nx.average_clustering(G, weight='weight')
    print(f"Coeficiente de clustering médio: {avg_clustering:.2f}")

# Visualização do grafo
def visualize_graph_improved(G):
    plt.figure(figsize=(16, 12))
    
    # Filtrar arestas para mostrar apenas as mais significativas
    edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 1]
    filtered_G = nx.Graph()
    filtered_G.add_edges_from(edges_to_keep)
    
    # Copiar atributos dos nós
    for node in G.nodes():
        filtered_G.add_node(node)
        filtered_G.nodes[node]['is_hero'] = G.nodes[node]['is_hero']
        filtered_G.nodes[node]['power_level'] = G.nodes[node]['power_level']
    
    # Definir paleta de cores mais contrastante
    hero_color = '#1f78b4'  # Azul mais forte
    villain_color = '#e31a1c'  # Vermelho mais vivo
    
    colors = []
    for node in filtered_G.nodes():
        if filtered_G.nodes[node]['is_hero']:
            colors.append(hero_color)
        else:
            colors.append(villain_color)
    
    # Layout mais espaçado
    pos = nx.kamada_kawai_layout(filtered_G, weight='weight')
    
    # Tamanho dos nós com escala não-linear para melhor visualização
    base_size = 800
    sizes = [base_size * (filtered_G.nodes[node]['power_level']**1.5 / 20) for node in filtered_G.nodes()]
    
    # Desenhar arestas com largura e transparência variável
    edge_weights = [filtered_G[u][v]['weight'] for u, v in filtered_G.edges()]
    edge_alphas = [min(0.2 + 0.8 * (w/max(edge_weights)), 0.7) for w in edge_weights]
    
    nx.draw_networkx_edges(
        filtered_G, pos, 
        width=np.array(edge_weights)/2, 
        alpha=edge_alphas,
        edge_color='#888888'
    )
    
    # Desenhar nós com bordas escuras
    nx.draw_networkx_nodes(
        filtered_G, pos,
        node_size=sizes,
        node_color=colors,
        alpha=0.9,
        edgecolors='#333333',
        linewidths=1.5
    )
    
    # Desenhar labels apenas para personagens importantes
    important_chars = {
        "Tony Stark", "Steve Rogers", "Thor", "Bruce Banner", 
        "Natasha Romanoff", "Nick Fury", "Peter Parker", "Thanos",
        "Loki", "T'Challa", "Wanda Maximoff", "Peter Quill"
    }
    
    labels = {node: node if node in important_chars else '' for node in filtered_G.nodes()}
    nx.draw_networkx_labels(
        filtered_G, pos, labels,
        font_size=9,
        font_family='sans-serif',
        font_weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )
    
    # Adicionar título e legenda
    plt.title("Rede de Interações do MCU (Filtrada para conexões mais relevantes)", fontsize=14, pad=20)
    
    # Legenda melhorada
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Herói',
                  markerfacecolor=hero_color, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Vilão',
                  markerfacecolor=villain_color, markersize=10),
        plt.Line2D([0], [0], color='#888888', lw=2, label='Interação (espessura = frequência)')
    ]
    
    plt.legend(
        handles=legend_elements, 
        loc='upper right',
        frameon=True,
        framealpha=0.9
    )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mcu_network_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

# Executando BFS e Dijkstra
def run_algorithms(mcu):
    print("\n=== Resultados dos Algoritmos ===")
    
    # BFS a partir do Tony Stark
    print("\nBFS a partir do Tony Stark (ordem de conexão):")
    bfs_result = mcu.bfs("Tony Stark")
    print(bfs_result[:10], "...")  # Mostrando os 10 primeiros
    
    # Dijkstra para encontrar os caminhos com mais interações
    print("\nDijkstra: 'distância' (inverso do peso) a partir do Tony Stark:")
    distances = mcu.dijkstra("Tony Stark")
    for char, dist in sorted(distances.items(), key=lambda x: x[1])[:10]:
        print(f"{char}: {dist:.2f}")

# Main
if __name__ == "__main__":
    # Criando o grafo
    mcu_graph = create_mcu_graph()
    
    # Convertendo para NetworkX para análise
    G = mcu_graph.to_networkx()
    
    # Realizando análises
    analyze_graph(G)
    
    # Executando algoritmos
    run_algorithms(mcu_graph)
    
    # Visualizando o grafo
    visualize_graph(G)