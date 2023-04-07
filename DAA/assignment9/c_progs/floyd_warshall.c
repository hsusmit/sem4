// Apply Floyd Warshall algorithm to find out shortest
// path between every pairs of vertices of the graph
// shown below. The graph is stored in adjacency matrix.

//   ---               ---
//   | 0   3   8   ∞  -4 |
//   | ∞   0   ∞   1   7 |
//   | ∞   4   0   ∞   ∞ |
//   | 2   ∞  -5   0   ∞ |
//   | ∞   ∞   ∞   6   0 |
//   ---               ---

#include <stdio.h>
#include <limits.h>

#define V 5
#define INF INT_MAX

void floydWarshall(int graph[V][V])
{
    int dist[V][V];
    int i, j, k;

    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            dist[i][j] = graph[i][j];

    for (k = 0; k < V; k++)
        for (i = 0; i < V; i++)
            for (j = 0; j < V; j++)
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    printf("Shortest path matrix:\n");
    for (i = 0; i < V; i++)
    {
        for (j = 0; j < V; j++)
        {
            if (dist[i][j] == INF)
                printf("INF\t");
            else
                printf("%d\t", dist[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    int graph[V][V] = {{0, 3, 8, INF, -4},
                       {INF, 0, INF, 1, 7},
                       {INF, 4, 0, INF, INF},
                       {2, INF, -5, 0, INF},
                       {INF, INF, INF, 6, 0}};

    printf("Floyd Warshall algorithm in C\n");
    floydWarshall(graph);
    return 0;
}

// Output:
// Floyd Warshall algorithm in C
// Shortest path matrix:
// 0       1       -3      2       -4
// 3       0       -4      1       -1
// 7       4       0       5       3
// 2       -1      -5      0       -2
// 8       5       1       6       0