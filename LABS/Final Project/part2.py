import math

#MinHeap Class
class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.map = {}
        self.build_heap()

        # add a map based on input node
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self,index):
        return 2 * (index + 1) - 1

    def find_right_index(self,index):
        return 2 * (index + 1)

    def find_parent_index(self,index):
        return (index + 1) // 2 - 1  
    
    def sink_down(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            
            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # recursive call
            self.sink_down(smallest_known_index)

    def build_heap(self,):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink_down(i) 

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):
        
        while index > 0 and self.items[self.find_parent_index(index)].key < self.items[self.find_parent_index(index)].key:
            #swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            #update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self,):
        #xchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        #update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.sink_down(0)
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]

    def is_empty(self):
        return self.length == 0
    
    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.items[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s
    

class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value
    
    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"

#WeightedGraph Class
class WeightedGraph:

    def __init__(self,nodes):
        self.graph = [[] for _ in range(nodes)]
        self.weights = {}

    def add_node(self,node):
        self.graph[node]=[]

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_weights(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

    def get_neighbors(self, node):
        return self.graph[node]

    def get_number_of_nodes(self,):
        return len(self.graph)
    
    def get_nodes(self,):
        return [i for i in range(len(self.graph))]


def bellman_ford(graph, source):
    distances = [math.inf] * graph.get_number_of_nodes()
    distances[source] = 0

    for _ in range(graph.get_number_of_nodes() - 1):
        for u in range(graph.get_number_of_nodes()):
            for v in graph.get_neighbors(u):
                if distances[u] != math.inf and distances[u] + graph.get_weights(u, v) < distances[v]:
                    distances[v] = distances[u] + graph.get_weights(u, v)

    return distances

def dijkstra(graph, source):
    distances = [math.inf] * graph.get_number_of_nodes()
    previous_vertices = [None] * graph.get_number_of_nodes()
    distances[source] = 0

    min_heap = MinHeap([Item(node, distances[node]) for node in graph.get_nodes()])
    visited = set()

    while not min_heap.is_empty():
        current_node = min_heap.extract_min().value
        visited.add(current_node)

        for neighbor in graph.get_neighbors(current_node):
            if neighbor not in visited:
                new_distance = distances[current_node] + graph.get_weights(current_node, neighbor)
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_vertices[neighbor] = current_node  # Track previous vertex
                    min_heap.decrease_key(neighbor, new_distance)

    return distances, previous_vertices

def all_pairs_shortest_path(graph):
    # Add a new vertex with zero-weight edges to all other vertices
    s = graph.get_number_of_nodes() - 1
    for i in range(graph.get_number_of_nodes()):
        graph.add_edge(s, i, 0)

    # Run Bellman-Ford algorithm from the new vertex
    h = bellman_ford(graph, s)

    # Recalculate edge weights using the computed distances
    for u in range(graph.get_number_of_nodes()):
        for v in graph.get_neighbors(u):
            graph.weights[(u, v)] += h[u] - h[v]

    # Initialize array to store distances and previous vertices
    distances = [[math.inf] * graph.get_number_of_nodes() for _ in range(graph.get_number_of_nodes())]
    previous_vertices = [[None] * graph.get_number_of_nodes() for _ in range(graph.get_number_of_nodes())]

    # Run Dijkstra's algorithm for each vertex
    for u in range(graph.get_number_of_nodes()):
        dist, prev = dijkstra(graph, u)
        for v in range(graph.get_number_of_nodes()):
            distances[u][v] = dist[v] + h[v] - h[u]
            previous_vertices[u][v] = prev[v]

    return distances, previous_vertices

# Example usage:
# Positive weighted graph
g = WeightedGraph(8)
edges = [(0,1,15),(0,6,15),(0,7,20),
         (1,0,15),(1,2,30),(1,4,45),
         (2,1,30),(2,3,5),
         (3,2,5),(3,5,25),(3,6,30),
         (4,1,45),(4,5,15),
         (5,3,25),(5,4,15),
         (6,0,15),(6,3,30),
         (7,0,20)]

# negative weighted graph 
# g = WeightedGraph(5)
# edges = [
#     (0, 1, 1),
#     (0, 2, 4),
#     (1, 2, 3),
#     (1, 3, 2),
#     (1, 4, 2),
#     (3, 1, -1),  # Negative edge weight
#     (3, 4, 5),
#     (4, 2, 1),
# ]

for edge in edges:
    g.add_edge(*edge)

# Run all_pair_shortest_path algorithm
distances, previous_vertices = all_pairs_shortest_path(g)

# Print the resulting distances
for u in range(len(distances)):
    for v in range(len(distances[u])):
        print(f"Shortest distance from {u} to {v}: {distances[u][v]}, Previous vertex: {previous_vertices[u][v]}")





