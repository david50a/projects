#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

struct position {
    int x, y, score;
};

pair<int, int> options[9];
char board[3][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'}
};

int pos, steps = 9;

bool is_winner(char player) {
    for (int i = 0; i < 3; i++) {
        if ((board[i][0] == player && board[i][1] == player && board[i][2] == player) ||
            (board[0][i] == player && board[1][i] == player && board[2][i] == player)) {
            return true;
        }
    }
    return (board[0][0] == player && board[1][1] == player && board[2][2] == player) ||
           (board[0][2] == player && board[1][1] == player && board[2][0] == player);
}

bool draw() {
    return steps == 0;
}

void displayBoard() {
    cout << "\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << " " << board[i][j] << " ";
            if (j < 2) cout << "|";
        }
        cout << "\n";
        if (i < 2) cout << "---+---+---\n";
    }
    cout << "\n";
}

position* minimax(bool is_maximizing) {
    if (is_winner('X')) return new position{0, 0, -1};
    if (is_winner('O')) return new position{0, 0, 1};
    if (draw()) return new position{0, 0, 0};

    position* best = new position{0, 0, is_maximizing ? -1000 : 1000};

    for (int i = 0; i < 9; i++) {
        int x = options[i].first, y = options[i].second;
        if (board[x][y] != 'X' && board[x][y] != 'O') {
            char original = board[x][y];
            board[x][y] = is_maximizing ? 'O' : 'X';

            position* value = minimax(!is_maximizing);
            board[x][y] = original;

            if (is_maximizing && value->score > best->score) {
                best->x = x;
                best->y = y;
                best->score = value->score;
            } 
            else if (!is_maximizing && value->score < best->score) {
                best->x = x;
                best->y = y;
                best->score = value->score;
            }
        }
    }
    return best;
}

void pc() {
    if (steps == 9) {
        int pos;
        do {
            pos = rand() % 9;
        } while (board[pos / 3][pos % 3] == 'X' || board[pos / 3][pos % 3] == 'O');
        board[pos / 3][pos % 3] = 'O';
    } else {
        position* move = minimax(true);
        board[move->x][move->y] = 'O';
    }
    steps--;
}

bool isValid() {
    return pos > 0 && pos < 10 && board[(pos - 1) / 3][(pos - 1) % 3] != 'X' && board[(pos - 1) / 3][(pos - 1) % 3] != 'O';
}

void placeMark() {
    cout << "Which place do you want to place your mark? (1-9): " << endl;
    cin >> pos;
    while (!isValid()) {
        cout << "Invalid input, please choose again (1-9): " << endl;
        cin >> pos;
    }
    board[(pos - 1) / 3][(pos - 1) % 3] = 'X';
    steps--;
}

int main() {
    srand(time(0));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            options[i * 3 + j] = {i, j};
        }
    }

    while (!draw() && !is_winner('X') && !is_winner('O')) {
        displayBoard();
        cout << "You are X, computer is O." << endl;
        placeMark();
        if (draw() || is_winner('X')) break;
        pc();
    }

    displayBoard();
    cout << (is_winner('X') ? "You are the winner!" : (is_winner('O') ? "The PC is the winner!" : "It's a tie!")) << endl;

    return 0;
}
