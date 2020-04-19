import java.util.Random;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;


class ArtiomArrayUtils {
	public static double[][][] empty3DArrayLike(double[][][] src) {
		double[][][] dest = new double[src.length][][];
		for (int i=0; i < src.length; i++)
			dest[i] = empty2DArrayLike(src[i]);
		
		return dest;
	}


	public static double[][] empty2DArrayLike(double[][] src) {
		double[][] dest = new double[src.length][];
		for (int i=0;  i < src.length; i++) {
			dest[i] = new double[src[i].length];
		}

		return dest;
	}


	public static void zeroFill3DArray(double[][][] arr) {
		for (int i=0;  i < arr.length; i++) 
			for (int j=0;  j < arr[i].length; j++) 
				for (int k=0;  k < arr[i][j].length; k++)
					arr[i][j][k] =	0;	

	}


	public static void multiplyArrayByScalar(double[][][] arr, double scalar) {
		for (int i=0;  i < arr.length; i++) 
			for (int j=0;  j < arr[i].length; j++) 
				for (int k=0;  k < arr[i][j].length; k++)
					arr[i][j][k] = arr[i][j][k] * scalar ;	

	}



	public static void FromArraySubtractArrayTimesAlpha(double[][][] arr1, double[][][] arr2, double alpha) {
		for (int i=0;  i < arr1.length; i++) 
			for (int j=0;  j < arr1[i].length; j++) 
				for (int k=0;  k < arr1[i][j].length; k++)
					arr1[i][j][k] -= alpha * arr2[i][j][k];
	}

	/**
	 * Print a 3D array 
	 * @param a - array to print
	 * @param labels[] - a String array containing dimensions text annotations, e.g. labels = {"Z", "Y", "X"} or {"Layer", "Neuron", "Weight"} 
	 */
	public static void print3DArray(double[][][] a, String[] labels) {
		for (int i=0;  i < a.length; i++) {
				System.out.println(labels[0] + " " + i + " :");
				for (int j=0;  j < a[i].length; j++){ 
					System.out.println("    " + labels[1] + " " + j + " :");
					for (int k=0;  k < a[i][j].length; k++){
						System.out.printf("        %s %d  :  %.7f\n", labels[2], k, a[i][j][k]);
				}
			}
		}	
	}



	public static int maxIndexInArray(double[] a) {
		double max = a[0];
		int maxIndex = 0;
		for (int i=1; i < a.length; i++) {
			if (max < a[i]) {
				max = a[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}




	public static double maxValueInArray(double[] a) {
		double max = a[0];
		int maxIndex = 0;
		for (int i=1; i < a.length; i++) {
			if (max < a[i]) {
				max = a[i];
				maxIndex = i;
			}
		}
		return max;
	}

	public static int closestToOneIndex(double[] a) {
		double closestDiff = Math.abs(a[0] - 1);
		int closestIndex = 0;
		for (int i=1; i < a.length; i++) {
			double currentDiff = Math.abs(a[i] - 1);	
			if (closestDiff > currentDiff) {
				closestDiff = currentDiff;
				closestIndex = i;
			}
		}
		return closestIndex;
	}


	public static double abs(double[][][] a) {
		double s = 0;
		for (double[][] l : a) {
			for (double[] n : l) {
				for (double w : n) {
					s += w * w;
				}
			}
		}
		return Math.sqrt(s);
	}
	
	public static void print2DArray(int[][] a) {
		String[] labels = {"Row", "Column"};
		for (int j=0;  j < a.length; j++) { 
	    	System.out.println("    Row " + j + " :");
	        for (int k=0;  k < a[j].length; k++) {
	        	System.out.printf("        %s %d  :  %d\n", "Column", k, a[j][k]);
			}
		}
			
	}


	public static void print2DArray(double[][] a) {
		String[] labels = {"Row", "Column"};
		for (int j=0;  j < a.length; j++) { 
	    	System.out.println("    Row " + j + " :");
	        for (int k=0;  k < a[j].length; k++) {
	        	System.out.printf("        %s %d  :  %f\n", "Column", k, a[j][k]);
			}
		}
			
	}
	
	
	public static void serialize3DArray(double[][][] arr, String filename) {
        try {
			ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(filename));
			os.writeObject(arr);
        }
        catch(IOException e) {
            System.out.println("Error while trying to serialize array!");
			throw new ArrayIndexOutOfBoundsException();
        }
		
	}

	public static double[][][] loadSerialized3DArray(String filename) {
			ObjectInputStream is;
			double[][][] arr = null;
			
			try {
				is = new ObjectInputStream(new FileInputStream(filename));
				try {
					arr = (double[][][]) is.readObject();
				}
				catch(Exception e) {
					System.out.println("Error while deserializing array!");
				}

	        }
	        catch(IOException e) {
    	        System.out.println("Wow there's something wrong with the file!");
	        }

			return arr;

	}




	

	public static String convert3DArrayToPythonStyle(double[][][] a) {
		StringBuffer arrString = new StringBuffer("[");
		for (int i=0;  i < a.length; i++) {
		    arrString.append("[");
			for (int j=0;  j < a[i].length; j++){ 
			    arrString.append("[");
				for (int k=0;  k < a[i][j].length; k++){
					arrString.append("" + a[i][j][k]);
					if (k != a[i][j].length - 1) arrString.append(", ");
				}
			    arrString.append("]");
				if (j != a[i].length - 1) arrString.append(", ");
			}
		    arrString.append("]");
			if (i != a.length - 1) arrString.append(", ");
		}	
	    arrString.append("]");
   	    return arrString.toString();
		
	}

	public static void write3DArrayToFilePythonStyle(double[][][] a, String filename) {
        try {
            String text = convert3DArrayToPythonStyle(a);
            Files.write(Paths.get(filename), text.getBytes());
        }
        catch (Exception e) {
            System.out.println("Something went wrong!");
        }
	}


	public static Integer[] toObject(int[] a) {
		int n = a.length;
		Integer[] objArr = new Integer[n];
		for (int i=0; i < n; i++) {
			objArr[i] = Integer.valueOf(a[i]);
		}
		return objArr;
	}
	
	
	
	
	public static void shuffleBound(Object[] a, Object[] b, Object[] c, long n) {
		shuffleBound(a, b, c, n, 1);
	}

	public static void shuffleBound(Object[] a, Object[] b, Object[] c, long n, long seed) {
		Random random = new Random(seed);
		if (!(a.length == b.length && b.length == c.length))
			throw new ArrayIndexOutOfBoundsException();

		for (long i=0; i < n; i++) {
        	int x = random.nextInt(a.length);
			int y = random.nextInt(a.length);
			if (x != y) {
				Object tmpA = a[x];
				Object tmpB = b[x];
				Object tmpC = c[x];
            
				a[x] = a[y];
				b[x] = b[y];
				c[x] = c[y];

				a[y] = tmpA;
				b[y] = tmpB;
				c[y] = tmpC;
			}
		}

	}

	public static void main(String[] args) {

	}

}
