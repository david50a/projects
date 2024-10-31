#include <iostream>
using namespace std;

#define N 8

int moveX[8] = {2, 1, -1, -2, -2, -1, 1, 2};
int moveY[8] = {1, 2, 2, 1, -1, -2, -2, -1};

// Function to check if a move is valid
bool isValid(int x, int y, int table[N][N]) {
    return (x >= 0 && x < N && y >= 0 && y < N && table[x][y] == -1);
}

// Recursive function to solve the Knight's Tour problem
bool solveKnightTour(int x, int y, int moveCount, int table[N][N]) {
    if (moveCount == N * N) {
        return true;
    }

    for (int k = 0; k < 8; k++) {
        int nextX = x + moveX[k];
        int nextY = y + moveY[k];
        if (isValid(nextX, nextY, table)) {
            table[nextX][nextY] = moveCount;
            if (solveKnightTour(nextX, nextY, moveCount + 1, table)) {
                return true;
            }
            table[nextX][nextY] = -1; // Backtracking
        }
    }
    return false;
}

// Function to print the board
void printBoard(int table[N][N]) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            cout << table[x][y] << " ";
        }
        cout << endl;
    }
}

int main() {
    int table[N][N];
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            table[x][y] = -1;
        }
    }

    int startX = 0, startY = 0; // Starting position
    table[startX][startY] = 0;   // Start the knight's path

    if (solveKnightTour(startX, startY, 1, table)) {
        printBoard(table);
    } else {
        cout << "Solution does not exist" << endl;
    }

    return 0;
}
