# N - Queens problem is to place n - queens in
# such a manner on an n x n chessboard that no
# queens attack each other by being in the same
# row, column or diagonal. Apply Backtracking to
# generate all possible valid assignment of a 4- Queen problem.


def is_valid(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False
    return True


def solve(board, row, n, solutions):
    if row == n:
        solutions.append([row[:] for row in board])
        return
    for col in range(n):
        if is_valid(board, row, col, n):
            board[row][col] = 1
            solve(board, row+1, n, solutions)
            board[row][col] = 0


def n_queens(n):
    board = [[0] * n for _ in range(n)]
    solutions = []
    solve(board, 0, n, solutions)
    return solutions


print("N - Queens Problem in Python")
n = int(input("Enter the value of N: "))
ans = n_queens(n)
count = len(ans)
if count == 0:
    print("No solution")
else:
    count = 1
    for i in ans:
        print("Solution:", count)
        for j in i:
            print(j)
        print()
        count += 1


# N - Queens Problem in Python
# Solution: 1
# [0, 1, 0, 0]
# [0, 0, 0, 1]
# [1, 0, 0, 0]
# [0, 0, 1, 0]

# Solution: 2
# [0, 0, 1, 0]
# [1, 0, 0, 0]
# [0, 0, 0, 1]
# [0, 1, 0, 0]
