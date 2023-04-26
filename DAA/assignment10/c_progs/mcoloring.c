// We are given a graph, we need to assign colors to the vertices of the graph. In the graph coloring
// problem, we have a graph and m colors, we need to find a way to color the vertices of the graph
// using the m colors such that any two adjacent vertices are not having the same color.
// Apply Backtracking to generate all possible valid coloring of the following graph. Here m = 3.

/*
    0-------1
    |      /|
    |    /  |
    |  /    |
    |/      |
    3-------2
*/

#include <stdio.h>
#include <stdbool.h>

#define V 4

bool isSafe(bool graph[V][V], int color[], int v, int c)
{
    for (int i = 0; i < V; i++)
        if (graph[v][i] && c == color[i])
            return false;
    return true;
}

void printColorings(int color[], int count)
{
    printf("Coloring %d: ", count);
    for (int i = 0; i < V; i++)
        printf("%d ", color[i]);
    printf("\n");
}

void graphColoringUtil(bool graph[V][V], int m, int v, int color[], int *count)
{
    if (v == V)
    {
        *count += 1;
        printColorings(color, *count);
        return;
    }

    for (int c = 1; c <= m; c++)
    {
        if (isSafe(graph, color, v, c))
        {
            color[v] = c;
            graphColoringUtil(graph, m, v + 1, color, count);
            color[v] = 0;
        }
    }
}

void graphColoring(bool graph[V][V], int m)
{
    int color[V] = {0};
    int count = 0;
    graphColoringUtil(graph, m, 0, color, &count);
    if (count == 0)
        printf("No valid coloring found.\n");
}

int main()
{
    bool graph[V][V] = {
        {0, 1, 0, 1},
        {1, 0, 1, 1},
        {0, 1, 0, 1},
        {1, 1, 1, 0}
    };

    int m = 3;

    printf("M-Coloring Graph problem in C\n");
    graphColoring(graph, m);
    return 0;
}

// OUTPUT
// M-Coloring Graph problem in C
// Coloring 1: 1 2 1 3
// Coloring 2: 1 3 1 2
// Coloring 3: 2 1 2 3
// Coloring 4: 2 3 2 1
// Coloring 5: 3 1 3 2
// Coloring 6: 3 2 3 1