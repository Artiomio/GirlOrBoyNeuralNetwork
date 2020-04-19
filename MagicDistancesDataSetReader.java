import java.io.*;
import java.util.Scanner;
import java.util.Random;
import java.util.Arrays;

public class MagicDistancesDataSetReader {
	final int NUMBER_OF_MALE_FACES = 30000;
	final int NUMBER_OF_FEMALE_FACES = 30000;
	
	int TOTAL_NUMBER_OF_FACES =  NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES; 
	
	final int NUMBER_OF_DISTANCES = 136;

	static double[] girlLabel = {0};  // -- {0, 0}; /* ВНИМАНИЕ! Изначально здесь были {0} и {1}*/
	static double[] guyLabel  = {1};  // -- {0, 1};

	double[][] magicDistances = new double[NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES][NUMBER_OF_DISTANCES];
	double[][] desiredOutputs = new double[NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES][];
	

	/* Массив с номерами полов: 0 - девушка, 1 - парень - номер бита, который должен "зажечься" */
	/* Не забыть учесть их при перемешивании! */
	int[] bitNumber = new int[NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES];
	
	private void readDoublesTo2DArrayFromFile(double[][] arr, int arrayOffset, String fileName, int n) {
	        try {
	            FileReader fileReader = new FileReader(fileName);
	            BufferedReader reader = new BufferedReader(fileReader);
	            Scanner scanner = new Scanner(reader);
	            
	            for (int face=0; face < n; face++) {
	            	if (face % ( n / 100) == 0)
	            		System.out.print("\rReady: " + (100 * face / n) + "%");
	            	for (int i=0; i < NUMBER_OF_DISTANCES; i++) {
	            		double x = scanner.nextDouble();
	            		arr[face + arrayOffset][i] = x;
	            	}
	            	
	            }
	            System.out.println("\rReady        ");

	        }
        	catch (IOException e) {
            	System.out.println("Error! ");
            	e.printStackTrace();
        	}


	}

	

	
	
	
	public void readMagicDistancesFromTextFile() {

		readDoublesTo2DArrayFromFile(magicDistances, 0, "magic_distances_girls", NUMBER_OF_FEMALE_FACES);
		readDoublesTo2DArrayFromFile(magicDistances, NUMBER_OF_FEMALE_FACES,  "magic_distances_guys", NUMBER_OF_MALE_FACES);



	}
	
	
	public void serializeMagicDistancesAndLabels(String distancesFileName /*, String labelsFileName*/ ) {
        try {
			ObjectOutputStream osDistances = new ObjectOutputStream(new FileOutputStream(distancesFileName));
			osDistances.writeObject(magicDistances);
        }
        catch(IOException e) {
            System.out.println("Wow there's something wrong with the file!");
        }
	}
	
	
	public void loadSerializedMagicDistancesAndLabels(String distancesFileName) {
		ObjectInputStream isDistances;
		
		try {

			isDistances = new ObjectInputStream(new FileInputStream(distancesFileName));
		
			
			try {
				magicDistances = (double[][]) isDistances.readObject();
			}
			catch(Exception e) {
				System.out.println("Error while deserializing the magicDistances array!");
			}

        }
        catch(IOException e) {
            System.out.println("Wow there's something wrong with the file!");
        }




	}
	
	

	
	
	
	
	public MagicDistancesDataSetReader()  throws IOException {

				long startTime = System.currentTimeMillis();
                // readMagicDistancesFromTextFile();
				System.out.println("Reading from the text files took " + (System.currentTimeMillis() - startTime));




				startTime = System.currentTimeMillis();

				/* Должен ли data set reader предоставлять доступ к чему-то, кроме значений входов и выходов */
				/* нейронной сети - массив массивов входов и массив массивов выходов и не включать в себя 
				/* информацию о том,  что кодируют входы и выходы - т.е., например, возможно, не стоит 
				/* включать информация с целочисленными ярлыками цифры MNIST */
				
				System.out.println("Loading data");
				loadSerializedMagicDistancesAndLabels("distances.serialized" /*, "labels.serialized"*/ );
				//serializeMagicDistancesAndLabels("distances.serialized" /*, "labels.serialized"*/ );

				System.out.println("Loaded!");

				System.out.println("Reading from the text files took " + (System.currentTimeMillis() - startTime));
            
				/* Формируем ярлыки */
				for (int i=0; i < NUMBER_OF_FEMALE_FACES; i++) {
					desiredOutputs[i] = girlLabel;
					bitNumber[i] = 0;
					
				}
		

				for (int i=NUMBER_OF_FEMALE_FACES; i < NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES; i++) {
					desiredOutputs[i] = guyLabel;
					bitNumber[i] = 1;					
				}
				/* ---------------- */
				
				

			     shuffleDataSet((NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES) * 200, 1);

/*                for (int i=0; i < NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES; i++) {
                    System.out.print("" + bitNumber[i] + Arrays.toString(desiredOutputs[i]) + " : ");
                    System.out.println("" + Arrays.toString(magicDistances[i]));
                }
*/                

                

	}


        public void shuffleDataSet(long n, long seed) {
        	
            System.out.println("\nShuffling...");
            Random random = new Random(seed);
            for (long i=0; i < n; i++) {
                int x = random.nextInt(NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES);
                int y = random.nextInt(NUMBER_OF_FEMALE_FACES + NUMBER_OF_MALE_FACES);
                if (x != y) {
                    double[] tmpDistances = magicDistances[x];
                    double[] tmpLabel = desiredOutputs[x];
                    
                    magicDistances[x] = magicDistances[y];
                    desiredOutputs[x] = desiredOutputs[y];
                    
                    
                    magicDistances[y] = tmpDistances;
                    desiredOutputs[y] = tmpLabel;
                    
                    
                    
                    int tmp = bitNumber[x];
                    bitNumber[x] = bitNumber[y];
                    bitNumber[y] = tmp;
                    
                }
                if (i % ( n / 100) == 0)
                    System.out.print("\rReady: " + (100 * i / n) + "%");
                
            }
            
            System.out.println();
            
        }
	
	
	public static void main(String[] args) {
		try {
			MagicDistancesDataSetReader dataSetReader = new MagicDistancesDataSetReader();
		}
		catch (IOException e) {
		}

		


	}
    
}
