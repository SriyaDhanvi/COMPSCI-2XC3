import math

class WeightedGraph:
    def __init__(self, nodes):
        self.graph = [[] for _ in range(nodes)]
        self.weights = {}
    
    def add_node(self,node):
        self.graph.append([])

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

    def get_neighbors(self, node):
        return self.graph[node]

    def get_weights(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]
        
    def get_number_of_nodes(self,):
        return len(self.graph)
    
    def get_nodes(self,):
        return [i for i in range(len(self.graph))]
    

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
            new_cost = cost[current_node] + graph.get_weights(current_node, neighbor)
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

# Example usage:
graph = WeightedGraph(5)
edges = [
    (0, 1, 1),
    (0, 2, 4),
    (1, 2, 3),
    (1, 3, 2),
    (1, 4, 2),
    (3, 1, 1),
    (3, 4, 5),
    (4, 2, 1),
]
for edge in edges:
    graph.add_edge(*edge)

# Example heuristic function
heuristic = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}

source = 0
destination = 4

predecessor, path = A_Star(graph, source, destination, heuristic)
print("Predecessor:", predecessor)
print("Shortest path:", path)