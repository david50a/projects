import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class soduko {
    public int [][] matrix=new int[9][9];
    public soduko(){
        for(int i=0;i<9;i++){
            for (int j=0;j<9;j++){this.matrix[i][j]=0;}
        }
    }

    private JTextField[][] createAndShowGUI() {
        // Create the frame
        JFrame frame = new JFrame("soduko solver");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(600, 600);

        // Create a panel with a flow layout
          // Create a JPanel with a GridLayout
        JPanel mainPanel = new JPanel(new GridLayout(2, 1));
        JPanel panel = new JPanel(new GridLayout(9, 9));

        // Create a 2D array of JTextFields
        JTextField[][] buttons = new JTextField[9][9];
        Font textFieldFont = new Font("Arial", Font.PLAIN, 20);
        // Initialize and add JTextFields to the panel
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                buttons[i][j] = new JTextField();
                buttons[i][j].setPreferredSize(new Dimension(50, 50));
                buttons[i][j].setFont(textFieldFont);
                buttons[i][j].setHorizontalAlignment(JTextField.CENTER);
                panel.add(buttons[i][j]);
            }
        }
        // Add the panel to the frame

        JPanel panelButton =new JPanel();
        JButton button = new JButton("Solve");
        button.setPreferredSize(new Dimension(150, 50)); // Set the button size to 150x50 pixels
        button.setBounds(120, 150, 100, 50); // x, y, width, height
        button.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int[][] matrix=getValuesFromEntries(buttons);
                if (solver(matrix)){
                    for (int i=0;i<9;i++) {
                        for(int j=0;j<9;j++){
                         buttons[i][j].setText(String.valueOf(matrix[i][j]));
                        }
            }}
        }
        });
        panelButton.add(button);
        mainPanel.add(panel);
        mainPanel.add(panelButton);
        frame.getContentPane().add(mainPanel);
        // Display the window
        frame.pack(); // Adjusts the frame size to fit the preferred size of the button and text field
        frame.setVisible(true);
        return buttons;
    }
    public int[][] getValuesFromEntries(JTextField[][] e) {
    int[][] m = new int[9][9];
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            try {
                String text = e[i][j].getText();
                if (text != null && !text.equals("")) {
                    int value = Integer.parseInt(text);
                    m[i][j] = value;
                } else {
                    m[i][j] = 0;
                }
            } catch (NumberFormatException ex) {
                // Handle the case where the text is not a valid integer
                System.err.println("Invalid number format at e[" + i + "][" + j + "]: " + e[i][j].getText());
                m[i][j] = 0; // You can choose to handle this differently
            }
        }
    }
    return m;
    }

    public boolean solver(int[][]matrix){
        int[]find=checkZeros(matrix);
        if (find==null){return true;}
        int i=find[0],j=find[1];
        for(int number=1;number<10;number++){
            if (checkColumn(matrix,i,number)&&checkSqure(matrix,i,j,number)&&checkRow(matrix,j,number)){
                matrix[i][j]=number;
                if (solver(matrix)){return true;}
                matrix[i][j]=0;
        }
    }
        return false;
    }

    public boolean checkSqure(int[][]matrix,int column,int row, int num){
            int c=(column/3)*3,r=(row/3)*3;
            for(int i=c;i<c+3;i++){
                for (int j=r;j<3+r;j++){
                    if (matrix[i][j]==num){return false;}

                }
            }
            return true;
        }
    public boolean checkRow(int[][]matrix,int row, int num){
            for(int i=0;i<9;i++){
                    if (matrix[i][row]==num){return false;}
            }
            return true;
        }
    public boolean checkColumn(int[][]matrix,int column, int num){
            for(int i=0;i<9;i++){
                    if (matrix[column][i]==num){return false;}
            }
            return true;
        }
    public int[] checkZeros(int[][]matrix){
        int[]position=new int[2];
        for(int i=0;i<9;i++){
            for (int j=0;j<9;j++){
                if (matrix[i][j]==0){
                    position[0]=i;
                    position[1]=j;
                    return position;
                  }
            }
        }
        return null;
    }
    public static void main(String[] args) {
        // Schedule a job for the event-dispatching thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                soduko s=new soduko();
                s.createAndShowGUI();
            }
        });
    }
}
