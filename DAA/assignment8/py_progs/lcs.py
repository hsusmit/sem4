# Using dynamic programming determine longest common
# subsequence of (ABCBADCDF) and (CBAF).

def lcs(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    length = dp[n][m]
    ans = [''] * (length + 1)
    i, j = n, m
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            ans[length] = s1[i - 1]
            length -= 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(ans[1:])


print("LCS in python")
s1 = 'ABCBADCDF'
s2 = 'CBAF'
lcs_str = lcs(s1, s2)
print("String 1:", s1)
print("String 2:", s2)
print("Longest common subsequence is:", lcs_str)

# OUTPUT
# LCS in python
# String 1: ABCBADCDF
# String 2: CBAF
# Longest common subsequence is: CBAF
