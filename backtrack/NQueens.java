import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class NQueens extends JPanel {
    private static final int BOARD_SIZE = 8;
    private static JPanel square;
    private static char[][]board;
    private Image queenImage;

    public NQueens() {
        this.board=new char[BOARD_SIZE][BOARD_SIZE];
        for(int i=0;i<BOARD_SIZE;i++){for(int j=0;j<BOARD_SIZE;j++){board[i][j]='O';}}
        setLayout(new GridLayout(BOARD_SIZE, BOARD_SIZE));
        showBoard();
        loadImages();
        initializeBoard();
    }
     private void loadImages() {
        try {
            // Change the path to the location of your queen image
            queenImage = ImageIO.read(new File("C:/Users/Owner/IdeaProjects/soduko/out/production/backtrack/q.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void initializeBoard() {
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                square = new JPanel();
                if ((row + col) % 2 == 0) {
                    square.setBackground(Color.WHITE);
                } else {
                    square.setBackground(Color.BLACK);
                }
                if (board[row][col]=='Q') {
                    JLabel pieceLabel = new JLabel(new ImageIcon(queenImage));
                    square.add(pieceLabel);}
                add(square);
            }
        }
    }
    private boolean row(int row){
        for(int i =0;i<BOARD_SIZE;i++){if (this.board[row][i]=='Q'){return false;}}
        return true;}
    private boolean column(int column){
        for(int i =0;i<BOARD_SIZE;i++){if (this.board[i][column]=='Q'){return false;}}
        return true;}
    private boolean diagonal(int row,int column){
        int i=row,j=column;
        while (i<BOARD_SIZE && j<BOARD_SIZE){
            if(this.board[i][j]=='Q'){return false;}
            i++;
            j++;
        }
        i=row;j=column;
        while (i>=0 && j>=0){
            if(this.board[i][j]=='Q'){return false;}
            i--;
            j--;
        }
        i=row;j=column;
        while (i<BOARD_SIZE && j>-1){
            if(this.board[i][j]=='Q'){return false;}
            i++;
            j--;
        }
        i=row;j=column;
        while (i>=0 && j<BOARD_SIZE){
            if(this.board[i][j]=='Q'){return false;}
            i--;
            j++;
        }
        return true;
    }
    private boolean check(int i,int j){return diagonal(i,j)&& row(i)&&column(j);}
    private boolean solver(int count){
        if (count==BOARD_SIZE){return true;}
        for(int i=0;i<BOARD_SIZE;i++){
            for (int j=0;j< BOARD_SIZE;j++){
             if(check(i,j)){
                 board[i][j]='Q';
                 if (solver(count+1)){return true;}
                 board[i][j]='O';
             }
            }
        }return false;
    }
    private void showBoard(){
        solver(0);
        for(int i=0;i<BOARD_SIZE;i++){System.out.println(board[i]);}
    }
    private static void createAndShowGUI() {
        JFrame frame = new JFrame("Chess Board");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new NQueens());
        frame.setSize(800, 800);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }
}
