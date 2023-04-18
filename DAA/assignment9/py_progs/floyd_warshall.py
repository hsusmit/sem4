# Apply Floyd Warshall algorithm to find out shortest
# path between every pairs of vertices of the graph
# shown below. The graph is stored in adjacency matrix.

#   ---               ---
#   | 0   3   8   ∞  -4 |
#   | ∞   0   ∞   1   7 |
#   | ∞   4   0   ∞   ∞ |
#   | 2   ∞  -5   0   ∞ |
#   | ∞   ∞   ∞   6   0 |
#   ---               ---


INF = float('inf')


def floyd_warshall(adj_matrix):
    n = len(adj_matrix)
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dist[i][j] = adj_matrix[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


adj_matrix = [[0, 3, 8, INF, -4],
              [INF, 0, INF, 1, 7],
              [INF, 4, 0, INF, INF],
              [2, INF, -5, 0, INF],
              [INF, INF, INF, 6, 0]]

print("Floyd Warshall algorithm in Python")
result = floyd_warshall(adj_matrix)
print("Shortest path matrix:")
for row in result:
    print(row)

# Output
# Floyd Warshall algorithm in Python
# Shortest path matrix:
# [0, 1, -3, 2, -4]
# [3, 0, -4, 1, -1]
# [7, 4, 0, 5, 3]
# [2, -1, -5, 0, -2]
# [8, 5, 1, 6, 0]
