

import math
from abc import ABC, abstractmethod

class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value
    
    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"


# --- implementation of the UML diagram ---


class ShortPathFinder():
    def __init__(self):
        self.graph = None
        self.algorithm = None

    def calc_short_path(self, source, dest):        
        return self.algorithm.calc_sp(self.graph, source, dest)
    
    def set_graph(self, graph):
        self.graph = graph

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

class ShortestPathAlgorithm():
    @abstractmethod
    def calc_sp(self, graph, source, dest):
        pass

class Graph():
    def __init__(self, nodes):
        self.graph=[]
        self.weights={}
        for node in range(nodes):
            self.graph.append([])

    @abstractmethod
    def get_adj_nodes(self, node):
        pass

    @abstractmethod
    def add_node(self, node):
        pass
    
    @abstractmethod
    def add_edge(self, start, end, w):
        pass

    @abstractmethod
    def get_num_of_nodes(self):
        pass
    
    @abstractmethod
    def w(self, node):
        pass

class WeightedGraph(Graph):
    def __init__(self, nodes):
        self.graph=[]
        self.weights={}
        for node in range(nodes):
            self.graph.append([])

    def get_adj_nodes(self, node):
        return self.graph[node]
    
    def add_node(self, node):
        self.graph[node]=[]

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_number_of_nodes(self):
        return len(self.graph)

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

class HeuristicGraph(WeightedGraph):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.heuristics = {}

    def get_heuristic(self, node):
        return self.heuristics.get(node, 0.0)

#MinHeap Class
class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.build_heap()

        # add a map based on input node
        self.map = {}
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
    

    
# Dijkstra's implementation
class Dijkstra(ShortestPathAlgorithm):
    def __str__(self):
        return "Dijkstra"
    
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
# Bellman-Ford's implementation
class Bellman_Ford(ShortestPathAlgorithm):
    def __str__(self):
        return "Bellman-Ford"
    
    def bellman_ford(graph, source):
        distances = [math.inf] * graph.get_number_of_nodes()
        distances[source] = 0

        for _ in range(graph.get_number_of_nodes() - 1):
            for u in range(graph.get_number_of_nodes()):
                for v in graph.get_neighbors(u):
                    if distances[u] != math.inf and distances[u] + graph.get_weights(u, v) < distances[v]:
                        distances[v] = distances[u] + graph.get_weights(u, v)

        return distances

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()

    def get(self):
        return self.queue.pop(0)

    def empty(self):
        return len(self.queue) == 0


# A* implementation
class A_Star(ShortestPathAlgorithm):
    def __str__(self):
        return "A*"
    
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def A_Star(graph, source, destination, heuristic):
        priority_queue = PriorityQueue()
        priority_queue.put(0, source)

        predecessor = {source: None}
        cost = {source: 0}

        while not priority_queue.empty():
            current_cost, current_node = priority_queue.get()

            if current_node == destination:
                break

            for neighbor in graph.get_neighbors(current_node):
                new_cost = cost[current_node] + graph.get_weight(current_node, neighbor)
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    priority = new_cost + heuristic[neighbor]
                    priority_queue.put(priority, neighbor)
                    predecessor[neighbor] = current_node

        path = []
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = predecessor[current_node]
        path.reverse()

        return predecessor, path
    

# --- testing the code ---


def main():
    # create a graph
    graph = WeightedGraph(5)
    graph.add_edge(0, 1, 5)
    graph.add_edge(0, 2, 2)
    graph.add_edge(1, 2, 1)
    graph.add_edge(1, 3, 3)
    graph.add_edge(2, 4, 4)
    graph.add_edge(3, 4, 2)

    # init
    path_finder = ShortPathFinder()
    path_finder.set_graph(graph)
    dijkstra = Dijkstra()
    bellman_ford = Bellman_Ford()
    a_star = A_Star(HeuristicGraph(5))

    # Dijkstra's algorithm
    path_finder.set_algorithm(dijkstra)
    path = path_finder.calc_short_path(0, 4)

    print(path_finder.algorithm)
    print("Path:", path)
    print("\n")

    # Bellman-Ford algorithm
    path_finder.set_algorithm(bellman_ford)
    path = path_finder.calc_short_path(0, 4)

    print(path_finder.algorithm)
    print("Path:", path)
    print("\n")

    # A* algorithm
    path_finder.set_algorithm(a_star)
    path = path_finder.calc_short_path(0, 4)

    print(path_finder.algorithm)
    print("Path:", path)
    print("\n")

main()