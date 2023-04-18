// Using dynamic programming determine longest common
// subsequence of (ABCBADCDF) and (CBAF).

#include <stdio.h>
#include <string.h>
#define MAX(x, y) ((x) > (y) ? (x) : (y))

void lcs(char s1[], char s2[])
{
    int n = strlen(s1);
    int m = strlen(s2);
    int dp[n + 1][m + 1];
    for (int i = 0; i <= n; i++)
        dp[i][0] = 0;
    for (int j = 0; j <= m; j++)
        dp[0][j] = 0;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (s1[i - 1] == s2[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = MAX(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    int len = dp[n][m];
    char ans[len + 1];
    ans[len] = '\0';
    int index = len - 1;
    int i = n, j = m;

    while (i > 0 && j > 0)
    {
        if (s1[i - 1] == s2[j - 1])
        {
            ans[index] = s1[i - 1];
            index--, i--, j--;
        }
        else if (dp[i - 1][j] > dp[i][j - 1])
            i--;
        else
            j--;
    }

    printf("String 1: %s\n", s1);
    printf("String 2: %s\n", s2);
    printf("Longest common subsequence is: %s\n", ans);
}

int main()
{
    printf("LCS in C\n");
    lcs("ABCBADCDF", "CBAF");
    return 0;
}

// OUTPUT
// LCS in C
// String 1: ABCBADCDF
// String 2: CBAF
// Longest common subsequence is: CBAF