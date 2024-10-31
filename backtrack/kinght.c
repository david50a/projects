#include <stdio.h>
#include <stdbool.h>
#define N 8

int moveX[8] = {2, 1, -1, -2, -2, -1, 1, 2};
int moveY[8] = {1, 2, 2, 1, -1, -2, -2, -1};
bool isValid(int board[N][N],int i,int j){return board[i][j] == 0 && i >= 0 && i < N && j >= 0 && j < N;}
bool insert(int board[N][N],int i,int j,int step){
    if(step==N*N+1){return true;}
    for(int x=0;x<N;x++){
        int posY=moveY[x]+i;
        int posX=moveX[x]+j;
        if(isValid(board,posY,posX)){
            board[posY][posX]=step;
            if(insert(board,posY,posX,step+1)){return true;}
            board[posY][posX]=0;
        } 
    }return false;
}
int main(){
    int board[N][N];
    for(int i=0;i<N;i++){for(int j=0;j<N;j++){board[i][j]=0;}}
    board[0][0]=1;
    insert(board,0,0,2);
    for(int i=0;i<N;i++){for(int j=0;j<N;j++){printf("%d ",board[i][j]);}printf("\n");}
    return 0;
}