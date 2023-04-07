# A circuit which passes through every vertex exactly
# once is called a Hamilton circuit. A minimum weight
# Hamilton circuit is a Hamilton circuit that has the
# smallest possible weight of all Hamilton circuits.
# In graph theory terms, the TSP is the problem of
# finding a minimum weight Hamilton circuit. Apply dynamic
# programming to find out the minimum weight Hamiltonian circuit.

'''

            1
          / | \
         /  |  \
        /   |10 \
    20 /    |    \
      /     2     \ 18
     /    /   \    \
    /   /       \   \
   /  /12        6\  \
  / /               \ \
  4 ----------------- 3
           18           

'''
import math


def tsp(adj_matrix):
    n = len(adj_matrix)
    dp = [[math.inf] * n for _ in range(2**n)]
    dp[1][0] = 0

    for mask in range(1, 2**n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            for curr in range(n):
                if curr == last or not (mask & (1 << curr)):
                    continue
                dp[mask][curr] = min(dp[mask][curr], dp[mask ^ (
                    1 << curr)][last] + adj_matrix[last][curr])

    ans = math.inf
    for last in range(n):
        ans = min(ans, dp[2**n - 1][last] + adj_matrix[last][0])
    return ans


adj_matrix = [[0, 10, 18, 20],
              [10, 0, 6, 12],
              [18, 6, 0, 18],
              [20, 12, 18, 0]]

print("Travelling Salesman Problem (TSP) in Python")
print("Minimum weight Hamiltonian circuit:", tsp(adj_matrix))

# Output:
# Travelling Salesman Problem (TSP) in Python
# Minimum weight Hamiltonian circuit: 54
