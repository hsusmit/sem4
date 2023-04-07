// A circuit which passes through every vertex exactly
// once is called a Hamilton circuit. A minimum weight
// Hamilton circuit is a Hamilton circuit that has the
// smallest possible weight of all Hamilton circuits.
// In graph theory terms, the TSP is the problem of
// finding a minimum weight Hamilton circuit. Apply dynamic
// programming to find out the minimum weight Hamiltonian circuit.

/*

            1
          / | \
         /  |  \
        /   |10 \
    20 /    |    \
      /     2     \18
     /    /   \    \
    /   /       \   \
   /  /12        6\  \
  / /               \ \
  4 ----------------- 3
           18

*/

#include <stdio.h>
#include <limits.h>

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define V 4

int tsp(int graph[][V], int dp[][1 << V], int mask, int pos)
{
    if (mask == (1 << V) - 1)
        return graph[pos][0];
    if (dp[pos][mask] != -1)
        return dp[pos][mask];
    int min_weight = INT_MAX;
    for (int i = 0; i < V; i++)
    {
        if ((mask & (1 << i)) == 0)
        {
            int weight = graph[pos][i] + tsp(graph, dp, mask | (1 << i), i);
            min_weight = MIN(min_weight, weight);
        }
    }
    return dp[pos][mask] = min_weight;
}

int main()
{
    int graph[V][V] = {{0, 10, 18, 20},
                       {10, 0, 6, 12},
                       {18, 6, 0, 18},
                       {20, 12, 18, 0}};

    int dp[V][1 << V];
    for (int i = 0; i < V; i++)
        for (int j = 0; j < (1 << V); j++)
            dp[i][j] = -1;

    printf("Travelling Salesman Problem (TSP) in C\n");
    int min_weight = tsp(graph, dp, 1, 0);
    printf("Minimum weight Hamiltonian circuit: %d", min_weight);
    return 0;
}

// OUTPUT
// Travelling Salesman Problem (TSP) in C
// Minimum weight Hamiltonian circuit: 54