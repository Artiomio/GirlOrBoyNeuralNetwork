/** 
 * Класс для валидации работы ванильной сети с одним выходным нейроном, который
 * разделяем входные сигналы на два множества - 1 и 0
 *
 */
public class NeuralOneOrZeroValidator {
    NetworkOne network;
    double[][] validationInputs;
    int[] desiredNeuronState; /* Zero or One */
	private double alpha = 0; /* Смещение центра валидации */

    public NeuralOneOrZeroValidator(NetworkOne network, double[][] validationInputs, int[] desiredNeuronState) {
        
        if (validationInputs.length != desiredNeuronState.length) {
            System.out.println("Error in validator constructor: validationInputs and desiredNeuronState have different lengths!");
            throw new ArrayIndexOutOfBoundsException();
        }

        this.network = network;
        this.validationInputs = validationInputs;
        this.desiredNeuronState = desiredNeuronState;
    }

    public int validate(int startPos, int endPos) {
        int positives = 0;

        for (int i=startPos; i <= endPos; i++) {
            network.feedForward(validationInputs[i]);
			int res;
			if (Math.abs(network.output[0] - 0) < Math.abs(network.output[0] - 1 - alpha))
				res = 0;
			else
				res = 1;
				
				
            if (res == desiredNeuronState[i]) {
                positives++;
            }
				
            

            
        }
        return positives;
    }



    public boolean validate(int n) {
        if (validate(n, n) == 1)
        	return true;
     	else
     		return false;
     			
    }



    public int validateAndPrintTemp(int startPos, int endPos) {
        int positives = 0;

        for (int i=startPos; i <= endPos; i++) {
            network.feedForward(validationInputs[i]);
            int closestToOne = ArtiomArrayUtils.closestToOneIndex(network.output);
            if (closestToOne == desiredNeuronState[i]) {
                positives++;
            } else
            System.out.print(" " + i);

            
        }
        return positives;
    }


}