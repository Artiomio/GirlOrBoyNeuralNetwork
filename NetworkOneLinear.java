import java.util.Random;

/**
 * Класс для тестирования: активационная функция сети тут f(x) = x
 * Возможно, может пригодиться для проверки того, нельзя ли конкретную нейронную сеть заменить линейным классификатором.
 */
public class NetworkOneLinear extends NetworkOne {
     
    /**
     * Создание нейронной сети
     * @param numberOfInputs - количество входов
     * @param numberOfNeuronsInEachLayer - массив с количеством нейронов в каждом слое
     */
    public NetworkOneLinear(int numberOfInputs,
               int[] numberOfNeuronsInEachLayer) {
            
    	super(numberOfInputs, numberOfNeuronsInEachLayer);
    } /* Конец конструктора */



    protected double activation(double arg) {
      	return arg;
    } 


    /* Производная функции активации */
    /* Нужна в частности при определении dE/dz[i][j] */
    /* Что интересно - для случая с гиперболическим тангенсом можно срезать и порядочно сэкономить на вычислениях, */
    /* используя тот факт, что его производная tanh выражается через его квадрат: [tahn(x)]' = 1 - tanh(x)^2  */
    /* Т.е. можно сохранять массив функцией активации от каждого z при проходе feedforward */
    /* Однако, в случае использования rectified linear unit - такого рода оптимизация уже не будет возможна */
    protected double activationDerivative(double arg) {
   		return 1; 
    }



   
    
    public static void main(String[] args) {

    }
}