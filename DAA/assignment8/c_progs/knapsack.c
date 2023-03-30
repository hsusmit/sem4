// For the given set of items and knapsack capacity = 5 kg,
// find the optimal solution for the 0/1 knapsack problem
// making use of dynamic programming approach.
// weight = [ 2, 3, 4, 5 ] and value = [ 3, 4, 5, 6 ]

#include <stdio.h>
#include <limits.h>
#define MAX(x, y) ((x) > (y) ? (x) : (y))

int knapsack(int wt[], int val[], int n, int maxWeight)
{
    int prev[maxWeight + 1];
    int take, notTake, W;
    for (int i = 0; i <= wt[0]; i++)
        prev[i] = 0;
    for (W = wt[0]; W <= maxWeight; W++)
        prev[W] = val[0];

    for (int ind = 1; ind < n; ind++)
    {
        for (W = maxWeight; W >= 0; W--)
        {
            notTake = prev[W];
            take = INT_MIN;
            if (wt[ind] <= W)
                take = val[ind] + prev[W - wt[ind]];
            prev[W] = MAX(take, notTake);
        }
    }
    return prev[maxWeight];
}

int main()
{
    printf("0/1 Knapsack problem in C\n");
    int wt[] = {2, 3, 4, 5};
    int val[] = {3, 4, 5, 6};
    int maxWeight = 5;
    int n = sizeof(wt) / sizeof(wt[0]);

    int profit = knapsack(wt, val, n, maxWeight);
    printf("Maximum Profit: %d", profit);
    return 0;
}

// OUTPUT
// 0/1 Knapsack problem in C
// Maximum Profit: 7