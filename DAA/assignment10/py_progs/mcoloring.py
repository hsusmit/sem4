# We are given a graph, we need to assign colors
# to the vertices of the graph. In the graph coloring
# problem, we have a graph and m colors, we need to
# find a way to color the vertices of the graph
# using the m colors such that any two adjacent
# vertices are not having the same color.
# Apply Backtracking to generate all possible
# valid coloring of the following graph. Here m = 3.

'''
    0-------1
    |      /|
    |    /  |
    |  /    |
    |/      |
    3-------2 
'''


def is_valid(graph, colors, vertex, color):
    for i in range(len(graph)):
        if graph[vertex][i] == 1 and colors[i] == color:
            return False
    return True


def solve(graph, colors, vertex, m, solutions):
    if vertex == len(graph):
        solutions.append(colors[:])
        return
    for color in range(1, m+1):
        if is_valid(graph, colors, vertex, color):
            colors[vertex] = color
            solve(graph, colors, vertex+1, m, solutions)
            colors[vertex] = 0


def graph_coloring(graph, m):
    colors = [0] * len(graph)
    solutions = []
    solve(graph, colors, 0, m, solutions)
    return solutions


print("M - Coloring Problem in Python")
graph = [[0, 1, 0, 1],
         [1, 0, 1, 1],
         [0, 1, 0, 1],
         [1, 1, 1, 0]]
m = 3
ans = graph_coloring(graph, m)
count = len(ans)
if count == 0:
    print("No solution")
else:
    count = 1
    for i in ans:
        print("Coloring", count, ":", i)
        count += 1

# Output:
# M - Coloring Problem in Python
# Coloring 1 : [1, 2, 1, 3]
# Coloring 2 : [1, 3, 1, 2]
# Coloring 3 : [2, 1, 2, 3]
# Coloring 4 : [2, 3, 2, 1]
# Coloring 5 : [3, 1, 3, 2]
# Coloring 6 : [3, 2, 3, 1]
