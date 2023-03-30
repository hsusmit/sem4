# Find out the minimal number of scaler Multiplication needed to multiply
# these matrices. The dimension of the matrices are 8 x 12, 12 x 13, 13 x 14,
# and 14 x 19. Apply Dynamic Programming(Bottom-up) approach.
# Here input is an array storing dimensions of the matrices like p = {8, 12, 13, 14, 19}


import sys


def mcm(arr):
    n = len(arr)
    dp = [[0 for j in range(n)] for i in range(n)]

    for i in range(n-1, 0, -1):
        for j in range(i+1, n):
            dp[i][j] = sys.maxsize
            for k in range(i, j):
                steps = dp[i][k] + dp[k+1][j] + arr[i-1]*arr[k]*arr[j]
                if steps < dp[i][j]:
                    dp[i][j] = steps
    return dp[1][n-1]


print("Matrix Chain Multiplication in Python")
arr = [8, 12, 13, 14, 19]
ans = mcm(arr)
print("Minimal number of scaler multiplications:", ans)

# OUTPUT
# Matrix Chain Multiplication in Python
# Minimal number of scaler multiplications: 4832
