// Find out the minimal number of scaler Multiplication needed to multiply
// these matrices. The dimension of the matrices are 8 x 12, 12 x 13, 13 x 14,
// and 14 x 19. Apply Dynamic Programming (Bottom-up) approach.
// Here input is an array storing dimensions of the matrices like p={8,12,13,14,19}

#include <stdio.h>
#include <limits.h>

int mcm(int arr[], int n)
{
    int dp[n][n];
    for (int i = 1; i < n; i++)
        dp[i][i] = 0;

    for (int i = n - 1; i > 0; i--)
    {
        for (int j = i + 1; j < n; j++)
        {
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; k++)
            {
                int steps = (arr[i - 1] * arr[k] * arr[j]) + dp[i][k] + dp[k + 1][j];
                if (steps < dp[i][j])
                    dp[i][j] = steps;
            }
        }
    }
    return dp[1][n - 1];
}

int main()
{
    printf("Matrix Chain Multiplication in C\n");
    int arr[] = {8, 12, 13, 14, 19};
    int n = sizeof(arr) / sizeof(arr[0]);
    int ans = mcm(arr, n);
    printf("Minimal number of scaler multiplications: %d", ans);
}

// OUTPUT
// Matrix Chain Multiplication in C
// Minimal number of scaler multiplications : 4832