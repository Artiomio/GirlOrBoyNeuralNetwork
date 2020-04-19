import java.awt.event.KeyListener;
import java.awt.event.KeyEvent;
import javax.swing.JFrame;
import javax.swing.JTextArea;
import javax.swing.JPanel;
import java.awt.Font;
import java.awt.Color;

public class ArtConsole implements KeyListener {

    private int numberOfOccurences(String s, String subStr) {
    	return s.length() - s.replace(subStr,  "").length();
    }

    
    public JTextArea textArea;
    
    public void keyTyped(KeyEvent e) {
    }

    public void keyReleased(KeyEvent e) {
    }
    
    public JFrame keyboardReadingFrame;

    public volatile boolean keyPressed = false;
    public int keyCode;
    
 
    public int getKeyCode() {
        return keyCode; 

    }

    public void keyPressed(KeyEvent e) {
        this.keyPressed = true; 
        this.keyCode = e.getKeyCode();
    }


  
    public void clearKey() {
        keyPressed = false;
    }


    public int numberOfLines;
    public int numberOfColumns;

    public ArtConsole(int lines, int columns) {
        numberOfLines = lines;
        numberOfColumns = columns;
        keyboardReadingFrame = new JFrame();
        keyboardReadingFrame.setVisible(true);
        keyboardReadingFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        keyboardReadingFrame.addKeyListener(this);

        textArea = new JTextArea("Welcome!\n", lines, columns);
		/* textArea.setLineWrap(true); // Как это примирить с переносом строк? */

        /* textArea.setFont(new Font("monospaced", Font.PLAIN, 17)); */
        textArea.setFont(textArea.getFont().deriveFont(20f));

        textArea.setBackground(new Color(20, 50, 71));
        textArea.setForeground(new Color(170, 200, 170));
        textArea.addKeyListener(this);
        textArea.setEditable(false);

        keyboardReadingFrame.add(textArea);
        keyboardReadingFrame.pack();

    }

    public ArtConsole() {
    	this(25, 80);
    }
    
    
    public void print(String s) {
        textArea.append(s);

        if (numberOfOccurences(textArea.getText(), "\n") > numberOfLines) {
        	int firstEndOfLinePos =  textArea.getText().indexOf("\n");
           	textArea.replaceRange("", 0, firstEndOfLinePos + 1);

        }
    }

    public void println(String s) {
        print(s + "\n");
    }


    public void print(int s) {
        print("" + s);
    }

    public void println(int s) {
        print("" + s + "\n");
    }

    public void print(double s) {
        print("" + s);
    }

    public void println(double s) {
        print("" + s + "\n");
    }










    public void clearScreen() {
        textArea.setText(null);
    }



    public static void main(String[] args) {
        ArtConsole keyb = new ArtConsole(20, 50);

        while (true) {
            if (keyb.keyPressed) {
               keyb.clearKey();
               System.out.println("You pressed a key. Code: " + keyb.keyCode);
               keyb.println("You pressed a key. Code: " + keyb.keyCode);
            }
            
        }
    }

}
