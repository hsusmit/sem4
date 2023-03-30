# For the given set of items and knapsack capacity = 5 kg,
# find the optimal solution for the 0/1 knapsack problem
# making use of dynamic programming approach.
# weight = [2, 3, 4, 5] and value = [3, 4, 5, 6]


def knapsack(wt, val, n, maxWeight):
    prev = [0] * (maxWeight + 1)
    for i in range(wt[0], maxWeight + 1):
        prev[i] = val[0]

    for ind in range(1, n):
        for W in range(maxWeight, -1, -1):
            notTake = prev[W]
            take = float('-inf')
            if wt[ind] <= W:
                take = val[ind] + prev[W - wt[ind]]
            prev[W] = max(take, notTake)

    return prev[maxWeight]


print("0/1 Knapsack problem in Pyhton")
wt = [2, 3, 4, 5]
val = [3, 4, 5, 6]
maxWeight = 5
n = len(wt)

profit = knapsack(wt, val, n, maxWeight)
print("Maximum Profit:", profit)

# OUTPUT
# 0/1 Knapsack problem in Pyhton
# Maximum Profit: 7
