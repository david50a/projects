#include <iostream>
using namespace std;
bool diagonal_forward(char** board,int size,int col,int row,char *word){
    if(*word=='\0'){return true;}
    if(*word==board[col][row]){
        if (col<size-1 && row<size-1){return diagonal_forward(board,size,col+1,row+1,word+1);}
    }
    return false; 
}
bool diagonal_backward(char** board,int size,int col,int row,char *word){
    if(*word=='\0'){return true;}
    if(*word==board[col][row]){
        if (col>0 &&row >0)return diagonal_backward(board,size,col-1,row-1,word+1);}
    return false;
}
bool row_forward(char** board, int size,int col,int row,char *word){
    if(*word=='\0'){return true;}
    if(*word==board[col][row]){
        if (col>0 && col<size-1)return row_forward(board,size,col+1,row,word+1);}
    return false;
}
bool row_backward(char** board, int size,int col,int row,char *word){
    if(*word=='\0'){return true;}
    if(*word==board[col][row]){
        if (col>0 && col<size-1)return row_backward(board,size,col-1,row,word+1);}
    return false;
}
bool col_forward(char** board, int size,int col,int row,char *word){
    if(*word=='\0'){return true;}
    if(*word==board[col][row]){
        if (row>0 && row<size-1)return col_forward(board,size,col,row+1,word+1);}
    return false;
}bool col_backword(char** board, int size,int col,int row,char *word){
    if(*word=='\0'){return true;}
    if(*word==board[col][row]){
        if (row>0 && row<size-1)return col_backword(board,size,col,row-1,word+1);}
    return false;
}
int main(){
    char board[3][3]={{'a','s','d'},{'m','a','y'},{'d','p','n'}};
    char** matrix=new char *[3];
    char words[5][4]={{"aa"},{"eeu"},{NULL},{"naa"},{"may"}};
    char found[5][4];
    int count=0;
    for(int i=0;i<3;i++){matrix[i]=new char[3];}
    for(int i=0;i<3;i++){for(int j=0;j<3;j++){matrix[i][j]=board[i][j];}}
    for(int i=0;i<sizeof(board)/sizeof(board[0]);i++){for(int j=0;j<sizeof(board[0])/sizeof(board[0][0]);j++){for(int k=0;k<sizeof(words)/sizeof(words[0]);k++){if(board[i][j]==words[k][0]){
                    if((matrix,3,i,j,words[k])||col_forward(matrix,3,i,j,words[k])||row_backward(matrix,3,i,j,words[k])||row_forward(matrix,3,i,j,words[k])||diagonal_backward(matrix,3,i,j,words[k])||diagonal_backward(matrix,3,i,j,words[k])){
                        std::cout<<words[k]<<" have been found"<<endl;
                        for (int x = 0; x < sizeof(words[k])/sizeof(words[k][0]); x++){words[k][x]=NULL;}
                    }
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        delete[] matrix[i]; // Free each row
    }
    delete[] matrix; // Free the array of pointers
    return 0;
}