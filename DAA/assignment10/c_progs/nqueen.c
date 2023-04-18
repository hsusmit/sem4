// N - Queens problem is to place n - queens in
// such a manner on an n x n chessboard that no
// queens attack each other by being in the same
// row, column or diagonal. Apply Backtracking to
// generate all possible valid assignment of a 4- Queen problem.

#include <stdio.h>
#include <stdbool.h>

bool isSafe(int n, int board[][n], int row, int col)
{
    int i, j;

    for (i = 0; i < col; i++)
        if (board[row][i])
            return false;

    for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return false;

    for (i = row, j = col; j >= 0 && i < n; i++, j--)
        if (board[i][j])
            return false;

    return true;
}

void printSolution(int n, int board[][n])
{
    static int solCount = 0;
    printf("Solution %d:\n", ++solCount);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%d ", board[i][j]);
        printf("\n");
    }
    printf("\n");
}

void solveNQueensUtil(int n, int board[][n], int col)
{
    if (col >= n)
    {
        printSolution(n, board);
        return;
    }

    for (int i = 0; i < n; i++)
    {
        if (isSafe(n, board, i, col))
        {
            board[i][col] = 1;
            solveNQueensUtil(n, board, col + 1);
            board[i][col] = 0;
        }
    }
}

void solveNQueens(int n)
{
    int board[n][n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            board[i][j] = 0;

    solveNQueensUtil(n, board, 0);
}

int main()
{
    int n;
    printf("N-Queen Problem in C\n");
    printf("Enter the value of N: ");
    scanf("%d", &n);
    solveNQueens(n);
    return 0;
}

// OUTPUT
// N-Queen Problem in C
// Enter the value of N: 4
// Solution 1:
// 0 0 1 0
// 1 0 0 0
// 0 0 0 1
// 0 1 0 0

// Solution 2:
// 0 1 0 0
// 0 0 0 1
// 1 0 0 0
// 0 0 1 0