import java.util.Random;

public class NetworkOneTanh extends NetworkOne {
     
    /**
     * Создание нейронной сети
     * @param numberOfInputs - количество входов
     * @param numberOfNeuronsInEachLayer - массив с количеством нейронов в каждом слое
     */
    public NetworkOneTanh(int numberOfInputs,
               int[] numberOfNeuronsInEachLayer) {
            
    	super(numberOfInputs, numberOfNeuronsInEachLayer);
    } /* Конец конструктора */



    protected double activation(double arg) {
        /* Гиперболический тангенс */
        return Math.tanh(arg); /* Надо обязательно поэкспериментировать с linear rectified activation function */

    
        /* Сигмоида */
        // return 1.0 / (1 + Math.exp(-arg));

        /* функция активации relu */
        /*if (arg > 0)
        	return arg;
        else
        	return 0;
      	*/

    } 


    /* Производная функции активации */
    /* Нужна в частности при определении dE/dz[i][j] */
    /* Что интересно - для случая с гиперболическим тангенсом можно срезать и порядочно сэкономить на вычислениях, */
    /* используя тот факт, что его производная tanh выражается через его квадрат: [tahn(x)]' = 1 - tanh(x)^2  */
    /* Т.е. можно сохранять массив функцией активации от каждого z при проходе feedforward */
    /* Однако, в случае использования rectified linear unit - такого рода оптимизация уже не будет возможна */
    protected double activationDerivative(double arg) {
        /*  Производная гиперболического тангенса */
        
        double tanh = Math.tanh(arg);
    	return 1 - tanh * tanh;
    	


    	/* Производная сигмоиды */
    	// double sigmoid = 1.0 / (1.0 + Math.exp(-arg));
    	// return sigmoid * (1.0 - sigmoid);

    	/* Производная relu */
    	/*
    	if (arg < 0)
    		return 0;
    	else
    		return 1;
   		*/

    }



   
    
    public static void main(String[] args) {

    }
}