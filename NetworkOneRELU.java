import java.util.Random;

public class NetworkOneRELU extends NetworkOne {
     
    /**
     * Создание нейронной сети
     * @param numberOfInputs - количество входов
     * @param numberOfNeuronsInEachLayer - массив с количеством нейронов в каждом слое
     */
    public NetworkOneRELU(int numberOfInputs,
               int[] numberOfNeuronsInEachLayer) {
            
        super(numberOfInputs, numberOfNeuronsInEachLayer);
    } /* Конец конструктора */



    protected double activation(double arg) {
        /* функция активации relu */
        if (arg > 0)
            return arg;
        else
            return 0; 

    } 


    /* Производная функции активации */
    /* Нужна в частности при определении dE/dz[i][j] */
    /* Что интересно - для случая с гиперболическим тангенсом можно срезать и порядочно сэкономить на вычислениях, */
    /* используя тот факт, что его производная tanh выражается через его квадрат: [tahn(x)]' = 1 - tanh(x)^2  */
    /* Т.е. можно сохранять массив функцией активации от каждого z при проходе feedforward */
    /* Однако, в случае использования rectified linear unit - такого рода оптимизация уже не будет возможна */
    protected double activationDerivative(double arg) {

        
        /* Производная relu */

        if (arg < 0)
            return 0;
        else
            return 1; 


    }





    /**
     * Заполняет массив весов случайными числами (предположительно - для инициализации)
     * так как здесь используется функция активации relu - числа выбираются положительными
     */
    public void initializeWeightsWithRandomNumbers() {
        System.out.println("Initializing weights with uniformly distributed POSITIVE random numbers (so the training is not stuck in the zero area of the relu-function). \nProbably you should consider using the initializeWeightsWithNormalRandomNumbers() method");
        float range = 1;
        for (int currentLayer = 0; currentLayer < weight.length; currentLayer++) {
            for (int currentNeuron = 0; currentNeuron < weight[currentLayer].length; currentNeuron++){
                for (int currentWeight = 0; currentWeight < weight[currentLayer][currentNeuron].length; currentWeight++) {
                    weight[currentLayer][currentNeuron][currentWeight] = range * randomGenerator.nextDouble();
                }
            }
        }

    }




    /**
     * Заполняет массив весов случайными распределёнными по Гауссу числами (для инициализации)
     * 
     */
    public void initializeWeightsWithNormalRandomNumbers(double d, double addition, double minWeight) {
        System.out.println("Initializing weights with normal randoms");
        for (int currentLayer = 0; currentLayer < weight.length; currentLayer++) {
            for (int currentNeuron = 0; currentNeuron < weight[currentLayer].length; currentNeuron++) {
                for (int currentWeight = 0; currentWeight < weight[currentLayer][currentNeuron].length; currentWeight++) {
                   double randomWeight = d * randomGenerator.nextGaussian() + addition;
                   weight[currentLayer][currentNeuron][currentWeight] = randomWeight > 0 ? randomWeight : minWeight;
                }
            }
        }

    }


    public void initializeWeightsWithNormalRandomNumbers() {
        throw new RuntimeException("You can't use this kind of initialization with the RELU activation function");
    }

    public void initializeWeightsWithNormalRandomNumbers(double x) {
        throw new RuntimeException("You can't use this kind of initialization with the RELU activation function");
    }







   
    
    public static void main(String[] args) {

    }
}