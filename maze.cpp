#include <iostream>
#include <queue>
#include <vector>
using namespace std;

struct Cell {
    int x, y;
    Cell(int _x, int _y) : x(_x), y(_y) {}
};

bool isValid(char **grid, int i, int j, int n) {
    return i >= 0 && i < n && j >= 0 && j < n && grid[i][j] != '#' && grid[i][j] != '_';
}

void markShortestPath(char **maze, vector<vector<Cell>> &parent, int ex, int ey) {
    while (maze[ex][ey] != 'S' && maze[ex][ey]!='E') {
        maze[ex][ey] = '*';
        Cell p = parent[ex][ey];
        ex = p.x;
        ey = p.y;
    }
}

int findShortestPath(char **maze, int sx, int sy, int len) {
    queue<Cell> q;
    q.push(Cell(sx, sy));

    int move_count = 0, nodes_left_in_layer = 1, nodes_in_next_layer = 0;
    bool reach = false;

    vector<vector<bool>> visited(len, vector<bool>(len, false));
    vector<vector<Cell>> parent(len, vector<Cell>(len, Cell(-1, -1)));

    int x_sides[4] = {0, -1, 0, 1};
    int y_sides[4] = {1, 0, -1, 0};

    visited[sx][sy] = true;

    while (!q.empty()) {
        int x = q.front().x;
        int y = q.front().y;
        q.pop();

        if (maze[x][y] == 'E') {
            reach = true;
            markShortestPath(maze, parent, x, y);
            break;
        }

        for (int i = 0; i < 4; i++) {
            int new_x = x + x_sides[i];
            int new_y = y + y_sides[i];

            if (isValid(maze, new_x, new_y, len) && !visited[new_x][new_y]) {
                q.push(Cell(new_x, new_y));
                parent[new_x][new_y] = Cell(x, y);
                visited[new_x][new_y] = true;
                nodes_in_next_layer++;
            }
        }

        nodes_left_in_layer--;
        if (nodes_left_in_layer == 0) {
            nodes_left_in_layer = nodes_in_next_layer;
            nodes_in_next_layer = 0;
            move_count++;
        }
    }

    return reach ? move_count : -1;
}

bool find_path(char **maze, int len, int x, int y) {
    if (maze[x][y] == 'E') return true;
    if (!isValid(maze, x, y, len)) return false;
    maze[x][y] = '_';  
    if (find_path(maze, len, x - 1, y) || 
        find_path(maze, len, x + 1, y) || 
        find_path(maze, len, x, y + 1) || 
        find_path(maze, len, x, y - 1)) {
        return true;
    }
    maze[x][y] = '.'; 
    return false;
}

int main() {
    char table[10][10] = {
        {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#'},
        {'#', 'S', '.', '.', '.', '.', '.', '.', '.', '#'},
        {'#', '#', '#', '.', '#', '#', '.', '#', '.', '#'},
        {'#', '.', '.', '.', '#', '.', '.', '#', '.', '#'},
        {'#', '.', '#', '#', '#', '.', '#', '#', '.', '#'},
        {'#', '.', '.', '.', '.', '.', '#', '.', '.', '#'},
        {'#', '#', '#', '#', '#', '.', '#', '.', '#', '#'},
        {'#', '.', '.', '.', '#', '.', '.', '.', '#', '#'},
        {'#', '#', '.', '#', '#', '#', '#', '.', 'E', '#'},
        {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#'}
    };

    char **maze = new char *[10];
    for (int i = 0; i < 10; i++) {
        maze[i] = new char[10];
        for (int j = 0; j < 10; j++) {
            maze[i][j] = table[i][j];
        }
    }

    cout << "Maze:" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << maze[i][j] << " ";
        }
        cout << endl;
    }

    int result = findShortestPath(maze, 1, 1, 10);
    if (result == -1) {
        cout << "No path found." << endl;
    } else {
        cout << "Shortest Path Length: " << result << endl;
        cout << "Maze with Path:" << endl;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                cout << maze[i][j] << " ";
            }
            cout << endl;
        }
    }
    for (int i = 0; i < 10; i++) {
        delete[] maze[i];
    }
    delete[] maze;
    return 0;
}
