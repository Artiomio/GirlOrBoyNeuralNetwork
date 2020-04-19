import java.util.Random;

public class NetworkOne {
     
    /*  Константа DELTA_W определяет шаг градиентного спуска
        Uспользуется в частности в методе calculateSlowGradient() при вычислении
        разницы между двумя значениями функции ошибки (cost function) (f(x + DELTA_W) - f(x))
        Uменно эта константа DELTA_W играет основную роль в   т о ч н о с т и  определения минимума.  
        
        ---комментарий с кодом IEJ472kH  
        Что очень интересно - в этой константе нет необходимости при просчитывании градиентов 
        с помощью метода calculateBackpropagationGradient, так как в сущности мы вычисляем почти
        аналитически производную. Смотреть на это можно так: мы придаём некоторое приращение рассматриваемому
        весу DELTA_W и вычисляем соответственное изменение значения функции ошибки (cost function).
        При этом в это выражение deltaCostFunction будет входить множителем то самое DELTA_W. При вычислении же
        градиента мы, естественно, делим вектор приращений на это самое DELTA_W. Поэтому этот чистло иллюстративный
        шаг с добавлением приращения можно опустить - и не домножать всё выражение на DELTA_W. Т.е. перейти от
        терминов дифференциалов и конечных приращений к - производным - причем вычисленных не в конечных разностях.
          Тут напрашивается ещё идея с генерацией аналитически выраженных - конкретных - алгебраических
        выражений (например, на Си), для просчета feedforward и backpropagation. Видимо, на Theano принято делать
        что-то подобное. Эта идея мне пришла еще летом 2016.


        */  
    public double DELTA_W = 0.0001;


    public double regularizationL2 = 0.00000001;

    /*  Константа LEARNING_RATE используется при градиентном спуске в частоности в методе  slowGradientDescent
        Определяет "скорость спуска": при корректировании весов (т.е. обучении) из них вычитается градиент, умноженный на эту
        константу.  */
    public double LEARNING_RATE = 10;


    public double randomWeightAdditionRange = 3;


    /* Массив весов:
            первый индекс - номер слоя
            второй - номер нейрона на этом слое
            третий  - номер веса этого нейрона
            
       Каждый нейрон n-го слоя связан с каждым из нейронов (n - 1)-го слоя.
       При этом дополнительный - последний вес каждого нейрона - это смещение (bias)


       Этот массив можно было бы организовать как двумерный, а не трехмерный, так как в случае, если
       каждый нейрон следующего слоя связан с каждым нейроном предыдущего слоя - каждый вес конкретного
       слоя однозначно идентифицируется. Но наглядность будет теряться и при расчете того, к какому нейрону
       принадлежит данный вес, придется использовать операцию целочисленного деления. Возможно, это может ускорить
       работу сети - но скорее всего то, как устроено сейчас, - по скорости близко к оптимуму.

    */
    public double[][][] weight;
    public double[][][] costGradient; /* Здесь будет находиться градиент функции-ошибки (cost function) */

    /*     При этом на данный момент подразумевается, что при вызове методов обучения будет
       передаваться не  весь тренировочный материал, а только некоторый фрагмент - minibatch
    */ 



    public double[][] charge; /* Значения выходных "зарядов" нейронов: charge[номер_слоя][номер_нейрона] */
    public double[][] z;      /* Массив z для каждого нейрона $z^{l}_{n} = \sum_{i} a^{l-1}_{i} w_{n,i}^{l}$ */
                       /* z[l][n] = a[l-1][0] * w[l][n][0] + a[l-1][1] * w[l][n][1] + ...            */ 


    public double[][] dE_over_dz; /* Производная функции ошибки (cost function) по z[l][n] с
                              соответствующими индексами слоя и номера нейрона 
                              У каждого нейрона это значение своё. Умножив его на некоторое delta_z,
                              можно получить соответствующее этому изменению изменение функции ошибки (cost function)
                              */

    public double[] output;   /* Массив, который ссылается на последний выходной слой массива charge */

    // double[] input; /* Пока закомментировано - см. комментарий ниже*/


    protected int numberOfLayers;


    public double lastChangedWeightUnchanged;
    public int lastChangedLayerNumber;
    public int lastChangedNeuronNumber;
    public int lastChangedWeightNumber;

    protected int numberOfInputs;
    protected int numberOfOutputNeurons;


    /* Uспользуется для выделения случайного слоя из сети:
       чем больше весов в слое - тем чаще он будет "выпадать".
       Задействован в функции addRandomNumberToRandomWeight()
    */
    protected WeightedRandom weightedRandomGenerator;


    /* Uспользуется для выделения случайного нейрона и случайного веса из
       определенного слоя. Тоже задействован в функции addRandomNumberToRandomWeight()
    */
    Random randomGenerator = new Random(2023);
    
    
    /* trainingInputs - значения  в х о д н ы х  "контактов" учебных данных для нейронной сети
       desiredOutputs - значения  в Ы х о д н ы х "контактов" учебных данных, которые должен
         стремиться получить процесс обучения, когда на вход подаются образцы из trainingInputs.

        Uспользуемая для обучения часть данных может быть сужена до т.н. мини-пакета (mini-batch) -
        границы определяются переменными miniBatchStartIndex и miniBatchEndIndex. Граничные значения
        включаются в промежуток.
    */

    protected double[][] trainingInputs, trainingOutputs;
    protected int miniBatchStartIndex = 0;
    protected int miniBatchEndIndex = 0;





    /**
     * Создание нейронной сети
     * @param numberOfInputs - количество входов
     * @param numberOfNeuronsInEachLayer - массив с количеством нейронов в каждом слое
     */
    public NetworkOne(int numberOfInputs,
               int[] numberOfNeuronsInEachLayer) {
            
        System.out.println("NetworkOne constructor has been run!");
        this.numberOfLayers = numberOfNeuronsInEachLayer.length;
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputNeurons = numberOfNeuronsInEachLayer[numberOfNeuronsInEachLayer.length - 1];
        System.out.println("Number of neurons in the output layer: " + this.numberOfOutputNeurons);
 
        /* Сначала создаём массив выходных контактов с входами первого слоя                 */
        /* Пока что необходимость хранить массив "входных контактов" нейронной сети         */
        /* отпала - входные параметры передаются в качестве аргументов функции feedForward, */ 
        /* но, возможно, ее стоит вернуть позже */
        //input = new double[numberOfInputs];




        /* Теперь создаём массив выходов нейронов - там будут храниться "выходные заряды"
           Т.к. кол-во нейронов разное в разных слоях - делаем это в цикле:
           После этого, чтобы получить выходной заряд на i-м нейроне
           из j-го слоя, нужно обратиться к элементу charge[j][i].
           
           При этом, если последний - выходной нейрон - один, то это должно
           быть отражено в массиве.

           */
        
        charge = new double[numberOfLayers][];
        for (int i = 0; i < numberOfLayers; i++) {
            charge[i] = new double[numberOfNeuronsInEachLayer[i]];    
        }


        /* Создаём массив z для каждого нейрона */
        /* $z^{l}_{n} = \sum_{i} a^{l-1}_{i} w_{n,i}^{l}$ */
        z = ArtiomArrayUtils.empty2DArrayLike(charge);
 
        /* Создаём массив dE_over_dz - производной функции ошибки по различным z  - элемент dE_over_dz[i][j] определяет то, как изменится функция ошибки при изменении z на (i,j)-м нейроне */
        dE_over_dz = ArtiomArrayUtils.empty2DArrayLike(charge);



        /* Теперь каждому нейрону создаём массив весов
           Сначала создаем элементы первое измерение - индекс слоёв.
        */
        weight = new double[numberOfLayers][][];


        /*   Во избежание путаницы - массив весов для первого слоя (связь со входами) -
           инициализируем отдельно.
        */
        
        int numberOfNeuronsInTheFirstLayer = numberOfNeuronsInEachLayer[0];
        weight[0] = new double[numberOfNeuronsInTheFirstLayer][];
        for (int currentNeuron = 0; currentNeuron < numberOfNeuronsInTheFirstLayer; currentNeuron++) {
            /* У каждого нейрона в первом слое количество весов равно количеству входов.
               При этом считаем, что первый вес - это связь с первым входом, второй - со втором и т.д. */
            weight[0][currentNeuron] = new double[numberOfInputs + 1];
            /*  Добавочная единица в строке выше - служит для хранения сдвигов (biases) */
        } 
 

       
        /* Теперь переходим ко второму слою, который уже типичен, в отличие от первого, каждый нейрон которого
           был связан не с нейроном, а со входом.
           Теперь у каждого нейрона количество весов (и входов) равно числу нейронов в предыдущем слое. 
        */        
        for (int currentLayer = 1; currentLayer < numberOfLayers; currentLayer++) {
            /* Каждый weight[currentLayer] - это двумерный массив, который состоит из нескольких одномерных.
               В нашем случае кол-во таких одномерных массивов - numberOfNeuronsInEachLayer[currentLayer]
               При этом кол-во элементов в них уже всегда одинаковое - numberOfNeuronsInEachLayer[currentLayer - 1]
            */
            
            /* As far as I understand the line below and the for cycle below could be replaced with one line:
                weight[currentLayer] = new double[numberOfNeuronsInEachLayer[currentLayer]][numberOfNeuronsInEachLayer[currentLayer - 1]]
                Or a little more verbose:
                   int numberOfNeuronsInThisLayer = numberOfNeuronsInEachLayer[currentLayer];
                   int numberOfNeuronsInPreviousLayer = numberOfNeuronsInEachLayer[currentLayer - 1];
                   weight[currentLayer] = new double[numberOfNeuronsInThisLayer][numberOfNeuronsInPreviousLayer]

                   вместо изначальных строчек:
                        weight[currentLayer] = new double[numberOfNeuronsInEachLayer[currentLayer]][];
                        for(int currentNeuron = 0; currentNeuron < numberOfNeuronsInEachLayer[currentLayer]; currentNeuron++) {
                            weight[currentLayer][currentNeuron] = new double[numberOfNeuronsInEachLayer[currentLayer - 1]];
                        }

                */


            int numberOfNeuronsInThisLayer = numberOfNeuronsInEachLayer[currentLayer];
            int numberOfNeuronsInPreviousLayer = numberOfNeuronsInEachLayer[currentLayer - 1];
            weight[currentLayer] = new double[numberOfNeuronsInThisLayer][numberOfNeuronsInPreviousLayer + 1];
            /* Добавочная единица в строчке выше - нужна для хранения сдвигов (biases) */


        } /* currentLayerCycle */


        output = charge[numberOfLayers - 1];

        /*
        Теперь чтобы обратиться к весу с номером x - нейрона с номером y - из слоя с номером z - нужно
        обратиться к элементу weight[z][y][x]                                                        
      
      
        Создаём генератор случайных чисел, который будет возвращать 0, 1, 2,.., вплоть до номера       
        последнего слоя (отсчёт нуля)                                                                  
        При этом относительные частоты будут соответствовать количеству весов нейронов в каждом слое:  
            номер слоя с бОльшим кол-вом весов нейронов будет "выпадать" чаще: тем чаще - чем больше
            в нём нейронных связей.                         
       
        Для начала сохраняем в массив количество numberOfWeightsInEachLayer[] количество всех весов в  
        каждом отдельном слое.
        */

        int[] numberOfWeightsInEachLayer = new int[numberOfLayers];
        for (int currLayer=0; currLayer < numberOfLayers; currLayer++) {
            numberOfWeightsInEachLayer[currLayer] = numberOfNeuronsInEachLayer[currLayer] * weight[currLayer][0].length;
        }

        /* Создаём генератор случайных чисел с вероятностными(!) весами. В качестве весов передаём созданный только что массив */
        weightedRandomGenerator = new WeightedRandom(numberOfWeightsInEachLayer, randomGenerator);


        /*  Создаём массив градиента, имеющий те же размеры что и массив весов */
        /*  Каждому весу - в соответствие - приращение (~производная) функции ошибки (cost function) */
        costGradient = ArtiomArrayUtils.empty3DArrayLike(weight);


 
    } /* Конец конструктора */




    /**
     * Заполняет массив весов случайными числами (предположительно - для инициализации)
     */
    public void initializeWeightsWithRandomNumbers() {

        System.out.println("Initializing weights with uniformly distributed random numbers.\nProbably you should consider using the initializeWeightsWithNormalRandomNumbers() method");

        float range = 1;

        for (int currentLayer = 0; currentLayer < weight.length; currentLayer++) {
            for (int currentNeuron = 0; currentNeuron < weight[currentLayer].length; currentNeuron++){
                for (int currentWeight = 0; currentWeight < weight[currentLayer][currentNeuron].length; currentWeight++) {
                    weight[currentLayer][currentNeuron][currentWeight] = range * randomGenerator.nextDouble() - (range / 2);
                }
            }
        }

    }




    /**
     * Заполняет массив весов случайными распределёнными по Гауссу числами (для инициализации)
     * 
     */
    public void initializeWeightsWithNormalRandomNumbers(double d) {
        System.out.println("Initializing weights with normal randoms");
        for (int currentLayer = 0; currentLayer < weight.length; currentLayer++) {
            for (int currentNeuron = 0; currentNeuron < weight[currentLayer].length; currentNeuron++) {
                for (int currentWeight = 0; currentWeight < weight[currentLayer][currentNeuron].length; currentWeight++) {
                    weight[currentLayer][currentNeuron][currentWeight] = d * randomGenerator.nextGaussian();
                }
            }
        }

    }


    public void initializeWeightsWithNormalRandomNumbers() {
        initializeWeightsWithNormalRandomNumbers(1);
    }


    /**
     * Добавляет случайное число случайно выбранному весу
     */
    protected void addRandomNumberToRandomWeight() {
        int layerNumber = weightedRandomGenerator.getRndResultNumber();
        int neuronNumber = randomGenerator.nextInt(weight[layerNumber].length);
        int weightNumber = randomGenerator.nextInt(weight[layerNumber][0].length);

        lastChangedWeightUnchanged = weight[layerNumber][neuronNumber][weightNumber];
        lastChangedLayerNumber = layerNumber;
        lastChangedNeuronNumber = neuronNumber;
        lastChangedWeightNumber = weightNumber;

        double additive = (this.randomWeightAdditionRange * randomGenerator.nextDouble() - (this.randomWeightAdditionRange / 2));
        //double additive = (randomGenerator.nextGaussian());

        weight[layerNumber][neuronNumber][weightNumber] += additive;
    }




    /**
     * Отменяет последнее добавление случайного числа к случайно выбранному весу
     */
    protected void undoAddingRandomNumberToRandomWeight() {
        weight[lastChangedLayerNumber][lastChangedNeuronNumber][lastChangedWeightNumber] = lastChangedWeightUnchanged;
    }




    /**
     * Производит заданное количество сотрясаний так, чтобы ошибка
     * на переданных данных и желаемых результатах не увеличилась, а при
     * хороших обстоятельствах - и уменьшилась
     */
    public void teachByShakingWeights(int numberOfShakes) {
        
        for (int i=0; i < numberOfShakes; i++) {
            // Вычисляем первую ошибку
            double error_1 = getMeanSquareError(); /* Скорость работы этой фунции сильно зависит от количества нейронов сети */

            // Добавляем случайное число случайному весу
            addRandomNumberToRandomWeight();

            // Вычисляем ошибку для получившейся сети
            double error_2 = getMeanSquareError();

            // Если новая ошибка больше, чем предыдущая, то отменяем изменение
            if (error_1 < error_2) {
                undoAddingRandomNumberToRandomWeight();
            }

        }
    }




    /**
     * Прямой проход нейронной сети 
     * @param inputValue - значение входных нейроннов
     *      NB: на данный момент значение входных сигналов не сохраняется как состояние сети после прямого прохода, 
     *   в отличие от зарядов charge[][] и взвешенных сумм z[][]
     */

    public void feedForward(double[] inputValue) throws ArrayIndexOutOfBoundsException {

        /* Проверяем соответствие размерностей */
        if (numberOfInputs != inputValue.length){
            System.err.println("feedForward(): Количество входных датчиков не соответствует количеству переданных данных:");
            System.err.println("Количество входных датчиков: " + numberOfInputs);
            System.err.println("Количество входных данных: " + inputValue.length);
            throw new ArrayIndexOutOfBoundsException();
        }

        /* Чтобы избежать неоднозначностей, сначала вычислим заряды на первом слое нейронов */
        
        for (int currentNeuron = 0; currentNeuron < weight[0].length; currentNeuron++){
            double s = 0;
            for (int currentWeight = 0; currentWeight < weight[0][currentNeuron].length - 1; currentWeight++){
                /* В строке выше ("..].length - 1;...") ВЫЧUТАНUНUЕ единицы нужно, чтобы исключить смещение (bias) из цикла по нейронам предыдущего слоя */
                s = s + weight[0][currentNeuron][currentWeight] * inputValue[currentWeight];
            }
            
            /* Прибавляем смещение (bias) */
            s = s + weight[0][currentNeuron][weight[0][currentNeuron].length - 1];
            
            /* Сохраняем в z линейную комбинацию  */
            z[0][currentNeuron] = s;    

            /* Вычисляем значение активационной функции для получившейся линейной комбинации */
            charge[0][currentNeuron] = activation(s);                          
        }

        /* Теперь - остальное */
        for (int currentLayer = 1; currentLayer < weight.length; currentLayer++){
            for (int currentNeuron = 0; currentNeuron < weight[currentLayer].length; currentNeuron++){
                double s = 0;
                for (int currentWeight = 0; currentWeight < weight[currentLayer][currentNeuron].length - 1; currentWeight++){
                    /* В строке выше ("..].length - 1;...") вычитание единицы нужно, чтобы исключить смещение (bias) из цикла по нейронам предыдущего слоя */
                    s = s + weight[currentLayer][currentNeuron][currentWeight] * charge[currentLayer - 1][currentWeight];
                }

                 /* Прибавляем смещение (bias) */
                s = s + weight[currentLayer][currentNeuron][weight[currentLayer][currentNeuron].length - 1];

                /* Сохраняем в z линейную комбинацию  */
                z[currentLayer][currentNeuron] = s;

                /* Вычисляем значение активационной функции для получившейся линейной комбинации */
                charge[currentLayer][currentNeuron] = activation(s);            
            }
        }
    } /* End of feedForward */


    /* Почему я решил сделать эту функцию protected? */
    /* Функция активации (sigma) */ 
    protected double activation(double arg) {
        /* Гиперболический тангенс */
        //return Math.tanh(arg); /* Надо обязательно поэкспериментировать с linear rectified activation function */

    
        /* Сигмоида */
        return 1.0 / (1 + Math.exp(-arg));

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
        
        //double tanh = Math.tanh(arg);
        //return 1 - tanh * tanh;
        


        /* Производная сигмоиды */
        double sigmoid = 1.0 / (1.0 + Math.exp(-arg));
        return sigmoid * (1.0 - sigmoid);

        /* Производная relu */
        /*
        if (arg < 0)
            return 0;
        else
            return 1;
        */

    }


    /* Функция ошибки - Error function - Cost function */
    /* Зависит от массива вещественных чисел - в сущности - выходных зарядов последнего слоя */
    /*                                                                                       */ 
    /*                                                                                       */ 
    /*   На данный момент в некотором роде есть д у б л и р о в а н и е - в части программы, */
    /* которая отвечает за обучение случайным блужданием в т.ч. в getMeanSquareError()       */
    /*   !UЛU ОНА вообще тут не нужна пока и нужна только ее производная? */
    protected double errorFunction(double[] lastCharge) {
        return 0;
    }


    
    /* ПРОUЗВОДНАЯ функции ошибки по одному из аргументов - error function derivative- Cost function derivative */
    /* Производная берётся по аргументу с номером numberOfArgument */
    /* Отсчет ведётся от нуля */
    
    protected double errorFunctionDerivative(double[] outputLayerCharge, double[] desiredOutputs, int numberOfArgument) {
        int i = numberOfArgument;
        return 2 * (outputLayerCharge[i] - desiredOutputs[i]) / outputLayerCharge.length; /* Оптимизация: деление на outputLayerCharge.length можно вынести туда же, где градиент умножается на LEARNING_RATE */
            
    }
    


 
    
    /**
     * Устанавлавает набор тренировочных данных, относительно которых впоследствии определяется ошибка,
     * которую мы стараемся минимизировать обучением, изменяя веса
     * @param trainingInputs - значения входных "контактов" учебных данных для нейронной сети
     * @param desiredOutputs - значения вЫходных "контактов" учебных данных, которые
     * должен стремиться получить процесс обучения, когда на вход подаются образцы из trainingInputs
     */
    public void setTrainingSet(double[][] trainingInputs, double[][] desiredOutputs) {
        System.out.println("\nsetTrainingSet: setting data set");

        if (trainingInputs.length != desiredOutputs.length) {
            System.out.println("Error in setTrainingSet():");
            System.out.println("The training examples and the desired outputs array have different lengths:");
            System.out.printf("Number of training examples : %d\nNumber of \"correct answers\" to them: %d\n", trainingInputs.length, desiredOutputs.length);           
            throw new ArrayIndexOutOfBoundsException();
        }

        System.out.printf("Number of training examples : %d\n", trainingInputs.length);

        this.trainingInputs = trainingInputs;
        this.trainingOutputs = desiredOutputs;

        this.miniBatchStartIndex = 0;
        this.miniBatchEndIndex = trainingInputs.length - 1;
        System.out.println("\nCaution: Default mini-batch covers the whole data set\nThe method setTrainingSet() is supposed to run once");
    }


    public void setCurrentMiniBatchRange(int left, int right) {
        this.miniBatchStartIndex = left;
        this.miniBatchEndIndex = right;
    }


    /**
     *    Тестовый заменитель функции ошибки - возвращающий сумму квадратов всех весов (с некоторыми
     *  вносимыми вариациями).
     *
     *    Может быть полезна для проверки метода поиска локального минимума в т.ч. градиентного спуска,
     *  т.к. функция ошибки (cost function) в этом случае будет представлять собой выпуклую функцию с минимумом
     *  в известной точке. При правильной работе алгоритма, набор весов примет опредённые значения.
     */
    protected double fakeCostFunction() {
        double s = 0;
        for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++){
            for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){
                for (int weightNumber = 0; weightNumber < weight[layerNumber][neuronNumber].length; weightNumber++){
                    s += weight[layerNumber][neuronNumber][weightNumber] * weight[layerNumber][neuronNumber][weightNumber];
                }    
            }
        }
        return s;

    }
   


    
    /**
     * Cost function - функция ошибки 
     * Возвращает среднеквадратичную ошибку предсказаний нейронной сети относительно
     * набора переданных данных. 
     * 
     * В отличие от синонимичного перегруженного метода, тут ошибка вычисляется относительно
     * данных, переданных в аргументах функции. В перегруженном же методе - в качестве данных
     * используется текущий мини-пакет (mini-batch).
     * Первый параметр - передаются массив массивов входных данных:
     *    - первый индекс первого параметра - это номер образца.
     *    - второй индекс первого параметра - номер "входного контакта" (e.g. 0 и 1, если входные - только x и y)
     *    
     * Второй параметр:
     *    - первый индекс - номер образца
     *    - второй индекс - номер выходного нейрона, желаемый заряд которого и передаётся в массиве
     *
     *             
     */
    public double getMeanSquareError(double[][] inputs, double[][] desiredOutputs) throws ArrayIndexOutOfBoundsException {
        
        /* Удостоверяемся в совпадении количества входных образцов и выходных эталонных данных */
        if (inputs.length != desiredOutputs.length){
            System.err.println("getMeanSquareError: Количество входных и выходных образцов не совпадает");
            throw new ArrayIndexOutOfBoundsException();
        }
 
        /* Удостоверяемся в том, что каждый образец содержит данные именно для необходимого числа "входных контактов" */ 
        for (int i=0; i < inputs.length ; i++){
            if (inputs[i].length != numberOfInputs){
                System.err.println("getMeanSquareError: Количество входных данных в образце с номером " + i + " не совпадает с количеством входных контактов нейронной сети");
                throw new ArrayIndexOutOfBoundsException();
            }
        }
    
        /* Удостоверяемся в том, что каждый эталон содержит данные именно для необходимого числа "вЫходных контактов" */ 
        
        for (int i=0; i < inputs.length ; i++){
            if (desiredOutputs[i].length != numberOfOutputNeurons){
                System.err.println("getMeanSquareError: Количество выходных данных в образце с номером " + i + " не совпадает с количеством вЫходных контактов нейронной сети");
                throw new ArrayIndexOutOfBoundsException();
            }
        }
        
        /* Здесь заканчиваются проверки размеров массивов */
        /* Конечно, для ускорения программы эти проверки можно исключить */
       
        double sumError = 0;
        double sampleSumError;
 
        for (int currentSample = miniBatchStartIndex; currentSample <= miniBatchEndIndex; currentSample++){
            /* Сначала прогоняем входные данные через нейронную сеть, после чего */
            /* результат окажется на выходном слое нейронов */
            feedForward(inputs[currentSample]);
 
            /* Суммируем */
            sampleSumError = 0;
            for (int currentNeuron = 0; currentNeuron < numberOfOutputNeurons; currentNeuron++){
                sampleSumError = sampleSumError + /* Вычисления ниже явно можно оптимизировать */
                                  (charge[numberOfLayers-1][currentNeuron] - desiredOutputs[currentSample][currentNeuron]) * 
                                  (charge[numberOfLayers-1][currentNeuron] - desiredOutputs[currentSample][currentNeuron]);
            }

            
            /* Усреднённая ошибка для образца с номером currentSample */
            /* Суммируем ошибку, получившующся на каждом отдельно образце */
            /* Возможно, множитель 1 / numberOfOutputNeurons стоит убрать */
            sumError = sumError + 1 * sampleSumError / numberOfOutputNeurons;
        }
 
        /* Делим на количество образцов */
        sumError = sumError / (miniBatchEndIndex - miniBatchStartIndex + 1);
    
        /* System.out.println("Mean Square Error = " + sumError); */
        
         /* ATTENTION! Здесь временно используется fakeCostFunction для отладки градиентного спуска */
        //return fakeCostFunction();
        return sumError;

    }




    /**
     *  Перегруженный метод getMeanSquareError(double[][] inputs, double[][] desiredOutputs) см. выше
     *  Тут, в отличие от аналогичного метода, не передаётся в аргументах набор тренировочных
     *  данных, а используются данные, предварительно устанавленные методом setTrainingSet()
     *  и пределах текущих границ мини-пакета (mini-batch) [miniBatchStartIndex, miniBatchEndIndex]
     */
    public double getMeanSquareError() throws ArrayIndexOutOfBoundsException {
        return getMeanSquareError(trainingInputs, trainingOutputs);

    }


    


    /*  Заполняет массив costGradient градиентом функции ошибки (cost function) 
        относительно данных, хранящихся в trainingInputs и trainingOutputs
        Расчет происходит по грубому алгоритму: к каждому весу прибавляется некоторое delta_w, после чего
        вычисляется значение функции ошибки. Разность между получившимся значением функиии ошибки и
        изначальным, помноженное на некоторый коэффициент в сущности и есть тот самый градиент.
        В отличие от алгоритма обратного распространения, этот метод очень затратный с точки зрения
        вычислений, т.к. прямой проход сети запускается для каждого из весов, в то время как при работе 
        обратного распространения требуется в сущности только один проход - обратныйю - а он требует
        не намного больше вычислений, чем прямой проход).
    */
    public void calculateSlowGradient() {
    
        /* Uзначальная функция ошибки (cost function), определённая на тренировочных данных данных trainingInputs и trainingOutputs */
        double initialError = getMeanSquareError();

        for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++){
            for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){
                for (int weightNumber = 0; weightNumber < weight[layerNumber][neuronNumber].length; weightNumber++){

                    /* Сохраняем вес, которому будем придавать приращение  */
                    double weightBeforeChange = weight[layerNumber][neuronNumber][weightNumber];

                    /* Придаём приращение delta_w */
                    weight[layerNumber][neuronNumber][weightNumber] += DELTA_W;

                    /* Записываем условный градиент - точнее - приращение функции ошибки (cost function) */ 
                    costGradient[layerNumber][neuronNumber][weightNumber] = (getMeanSquareError() - initialError) / DELTA_W;

                    /* Восстанавливаем первоначальное значение текущего веса */
                    weight[layerNumber][neuronNumber][weightNumber] = weightBeforeChange;

                }
            }
        }
        /* массив, содержащий градиент, сформирован */
    } 




    /**
     * Медленный градентный поиск, использующий функцию calculateSlowGradient, записывающую
     * в массив весов разницу значения функции ошибки при добавлении к соответствующему 
     * веса величины DELTA_W - константы, определяющй шаг градиентного спуска.
     * При этом после каждого добавления DELTA_W заново вычисляется функция ошибки (cost function)
     *
     * @param numberOfIterations - количество шагов градиентного поиска
     *
     * Uменно эта константа DELTA_W играет основную роль в   т о ч н о с т и  определения минимума.


     */
    public void slowGradientDescent(int numberOfIterations) {
        for (int i=0; i < numberOfIterations; i++) {
            calculateSlowGradient(); /* Сосчитали градиент */       
            /* Теперь поэлементно вычитаем его от массива весов (предварительно умножив на LEARNING_RATE) */
            for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++){
                for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){
                    for (int weightNumber = 0; weightNumber < weight[layerNumber][neuronNumber].length; weightNumber++){
                        weight[layerNumber][neuronNumber][weightNumber] -=  (LEARNING_RATE) * costGradient[layerNumber][neuronNumber][weightNumber];
                    }
                }
            }


        }

    }



    /**
     *
     **/
    public void calculateBackpropagationGradient(double[] trainingInputs, double[] desiredOutputs) {

        /* Проводим прямой проход сети - feedforward */
        /* После этого в каждом из z[][] сохранится линейная комбинация взвешенных сигналов соответствующего нейрона */
        /* U, конечно же, результат просчета - на выходных нейронах*/
        feedForward(trainingInputs);


        // Сначала заполним первую - самую правую строку dE_over_dz
        // как произведение производных функции активации и функции ошибки
        int layerNumber = numberOfLayers - 1; // Последний слой
        
        for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){
            dE_over_dz[layerNumber][neuronNumber] = 
                activationDerivative(z[layerNumber][neuronNumber])
                * errorFunctionDerivative(charge[layerNumber], desiredOutputs, neuronNumber);
        }



        /* Теперь отдельно находим dE для каждого из весов нейронов  п о с л е д н е г о  слоя - он нетипичен для рекуррентности, */
        /* которую мы будем использовать с остальными слоями */
        /* NB Предполагается, что есть по крайней мере еще один слой, кроме последнего и обращение к charge[layerNumber - 1] имеет смысл; */
        for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){
            int weightNumber;        
            /*                                                                     исключаем смещения (bias), которые хранятся в последнем весе каждого нейрона */                      
            for (weightNumber = 0; weightNumber < weight[layerNumber][neuronNumber].length - 1; weightNumber++){
                costGradient[layerNumber][neuronNumber][weightNumber] += /* DELTA_W * */ charge[layerNumber - 1][weightNumber] * dE_over_dz[layerNumber][neuronNumber]; /* Относительно закомментированного DELTA_W см. сноску IEJ472kH */
            }
            /* Теперь для смещения: вместе заряда нейрона тут уже просто единица */
            weightNumber = weight[layerNumber][neuronNumber].length - 1;
            costGradient[layerNumber][neuronNumber][weightNumber] += /* DELTA_W * */ 1 * dE_over_dz[layerNumber][neuronNumber]; /* Относительно закомментированного DELTA_W см. сноску IEJ472kH */
        }


        // Теперь итерационная процедура для всех остальных слоёв,  к р о м е  самого первого,
        // нетипичность которого заключается в том - что его веса присоединены не к нейронам, а к входным контактам
        // Для первого (с индексом 0) слоя цикл по весам -  о т д е л ь н ы й
        // Цикл по слоям тут будет уже обратным
        for (layerNumber = numberOfLayers - 2; layerNumber >= 1; layerNumber--){
            for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){

                // Для данного нейрона сначала сосчитаем dE/dz
                double current_de_over_dz = 0;
                int prevLayer = layerNumber + 1;
                /* Пробегаемся по всем нейронам слоя на один правее текущего */
                /* Точнее - по весам c номером, совпадающим с номером текущего нейрона - */
                /* то есть - по весам, просоединённым к текущему нейрону */
                for (int neuronNextLayer = 0; neuronNextLayer < weight[prevLayer].length; neuronNextLayer++){
                    current_de_over_dz += weight[prevLayer][neuronNextLayer][neuronNumber] * dE_over_dz[prevLayer][neuronNextLayer];
                }

                current_de_over_dz *= activationDerivative(z[layerNumber][neuronNumber]); /* Теперь домножаем на sigma' */
                /* То есть теперь у нас есть de/dz для данного нейрона */ 
                dE_over_dz[layerNumber][neuronNumber] = current_de_over_dz;


                int weightNumber;
                /*                                                                     исключаем смещения (bias), которые хранятся в последнем весе каждого нейрона */                      
                for (weightNumber = 0; weightNumber < weight[layerNumber][neuronNumber].length - 1; weightNumber++){
                    costGradient[layerNumber][neuronNumber][weightNumber] += /* DELTA_W * */ charge[layerNumber - 1][weightNumber] * current_de_over_dz; /* Относительно закомментированного DELTA_W см. сноску IEJ472kH */
                }
                /* Теперь для смещения: вместо заряда нейрона тут уже просто единица */
                weightNumber = weight[layerNumber][neuronNumber].length - 1;
                costGradient[layerNumber][neuronNumber][weightNumber] += /* DELTA_W * */ 1 * current_de_over_dz; /* Относительно закомментированного DELTA_W см. сноску IEJ472kH */


            }
        }

       
        layerNumber = 0;
        /* Теперь - самый первый слой - который связан уже не с нейронами, а с входными "датчиками", значения которых */
        /* хранятся в массиве trainingInputs[] */
        for (int neuronNumber = 0; neuronNumber < weight[layerNumber].length; neuronNumber++){

            // Для данного нейрона сначала сосчитаем dE/dz
            double current_de_over_dz = 0;
            int prevLayer = layerNumber + 1;
            /* Пробегаемся по всем нейронам слоя на один правее текущего */
            /* Точнее - по весам c номером, совпадающим с номером текущего нейрона - */
            /* то есть - по весам, просоединённым к текущему нейрону */
            for (int neuronNextLayer = 0; neuronNextLayer < weight[prevLayer].length; neuronNextLayer++){
                current_de_over_dz += weight[prevLayer][neuronNextLayer][neuronNumber] * dE_over_dz[prevLayer][neuronNextLayer];
            }

            current_de_over_dz *= activationDerivative(z[layerNumber][neuronNumber]); /* Теперь домножаем на sigma' */

            /* То есть теперь у нас есть de/dz для данного нейрона */ 
            dE_over_dz[layerNumber][neuronNumber] = current_de_over_dz;


            int weightNumber;
            /*                                                                     исключаем смещения (bias), которые хранятся в последнем весе каждого нейрона */                      
            for (weightNumber = 0; weightNumber < weight[layerNumber][neuronNumber].length - 1; weightNumber++){
                costGradient[layerNumber][neuronNumber][weightNumber] += /* DELTA_W **/ trainingInputs[weightNumber] * current_de_over_dz;  /* Относительно закомментированного DELTA_W см. сноску IEJ472kH */
            }
            /* Теперь для смещения: вместе заряда нейрона тут уже просто единица */
            weightNumber = weight[layerNumber][neuronNumber].length - 1;
            costGradient[layerNumber][neuronNumber][weightNumber] += /* DELTA_W * */ 1 * current_de_over_dz;    /* Относительно закомментированного DELTA_W см. сноску IEJ472kH */
        }


    }




    /**
     *  Обучение обратным распространением
     *  в пределах текущего мини-пакета, включающего себя образцы и эталонные ответы
     *  в this.trainingInputs и this.trainingOutputs в промежутке от [miniBatchStartIndex, miniBatchEndIndex] 
     */
    public void teachWithBackpropagation() /* Current minibatch */ {
        /* Тренировочные данные хранятся в this.trainingInputs и this.trainingOutputs */
        /* Обучение проводится в пределах [miniBatchStartIndex, miniBatchEndIndex] (границы включены) */
        /*  Это и есть текущий mini-batch */

        /* Надо обнулить массив градиентов: именно в него будут добавляться части градиента,
         составляющего градиент функции ошибки на всём мини-пакете */
        ArtiomArrayUtils.zeroFill3DArray(costGradient);

        /* Формируем градиент весов и смещений относительно функции ошибки, */ 
        /* построенной на текущем мини-пакете (данные хранятся в trainingInputs[] и trainingOutputs[])  */
        for (int trainingItemNumber = miniBatchStartIndex; trainingItemNumber <= miniBatchEndIndex; trainingItemNumber++) {
            /* При каждом таком вызове метода calculateBackpropagationGradient к массиву градиента будет
               прибавляться (поэлементное суммирование) градиент функции ошибки данного образца
            */
            calculateBackpropagationGradient(trainingInputs[trainingItemNumber], trainingOutputs[trainingItemNumber]);
        }

        
        /* Сформировали массив градиента */ 
        /* Теперь нужно из массива весов вычесть */
        /* NB: здесь мы делим на кол-во элементов элементов. Вообще, это деление должно быть в цикле метода calculateBackpropagationGradient() */
        /* Но с целью оптимизации я перенёс это сюда - чтобы не повторять деление в цикле */ 
        /* Поэтому в массиве costGradient хранится не совсем градиент функции ошибки - различие во множителе trainingInputs.length */
        ArtiomArrayUtils.FromArraySubtractArrayTimesAlpha(weight, costGradient,  LEARNING_RATE  / (miniBatchEndIndex - miniBatchStartIndex + 1));

    }




    /**
     *  Обучение обратным распространением С ФUЛЬТРОМ
     *  в пределах текущего мини-пакета, включающего себя образцы и эталонные ответы
     *  в this.trainingInputs и this.trainingOutputs в промежутке от [miniBatchStartIndex, miniBatchEndIndex],
     *  при этом учитываться будут  Т О Л Ь К О  образцы с номерами i,  Д Л Я   К О Т О Р Ы Х  mask[i] равно true 
     */
    public void teachWithBackpropagationWithMaskFilter(boolean[] mask) /* Current minibatch */ {
        /* Тренировочные данные хранятся в this.trainingInputs и this.trainingOutputs */
        /* Обучение проводится в пределах [miniBatchStartIndex, miniBatchEndIndex] (границы включены) */
        /* При этом учитываться будут  Т О Л Ь К О  образцы с номерами i,  Д Л Я   К О Т О Р Ы Х  mask[i] равно true */
        /*  Это и есть текущий mini-batch */

        /* Надо обнулить массив градиентов: именно в него будут добавляться части градиента,
         составляющего градиент функции ошибки на всём мини-пакете */
        ArtiomArrayUtils.zeroFill3DArray(costGradient);

        /* Формируем градиент весов и смещений относительно функции ошибки, */ 
        /* построенной на текущем мини-пакете (данные хранятся в trainingInputs[] и trainingOutputs[])  */
        for (int trainingItemNumber = miniBatchStartIndex; trainingItemNumber <= miniBatchEndIndex; trainingItemNumber++) {
            /* При каждом таком вызове метода calculateBackpropagationGradient к массиву градиента будет
               прибавляться (поэлементное суммирование) градиент функции ошибки данного образца
            */
            if (mask[trainingItemNumber]) {
                calculateBackpropagationGradient(trainingInputs[trainingItemNumber], trainingOutputs[trainingItemNumber]);
            }
        }

        
        /* Сформировали массив градиента */
        /* Теперь нужно из массива весов вычесть */
        /* NB: здесь мы делим на кол-во элементов элементов. Вообще, это деление должно быть в цикле метода calculateBackpropagationGradient() */
        /* Но с целью оптимизации я перенёс это сюда - чтобы не повторять деление в цикле */ 
        /* Поэтому в массиве costGradient хранится не совсем градиент функции ошибки - различие во множителе trainingInputs.length */
        ArtiomArrayUtils.FromArraySubtractArrayTimesAlpha(weight, costGradient,  LEARNING_RATE  / (miniBatchEndIndex - miniBatchStartIndex + 1));

    }





    public void teachWithBackpropagation(int n) {
        for (int i=0; i < n; i++) {
            teachWithBackpropagation();
        }
    }




    /**
     *  Обучение обратным распространением с L2-регуляризацией    
     *
     */
	public void teachWithBackpropagationL2() {
        ArtiomArrayUtils.zeroFill3DArray(costGradient); /* Обнуляем массив градиентов */
        for (int trainingItemNumber = miniBatchStartIndex; trainingItemNumber <= miniBatchEndIndex; trainingItemNumber++) {
            /* При каждом таком вызове метода calculateBackpropagationGradient к массиву градиента будет
			   прибавляться (поэлементное суммирование) градиент функции ошибки данного образца  */
			calculateBackpropagationGradient(trainingInputs[trainingItemNumber], trainingOutputs[trainingItemNumber]);
        }


        		

		/* От не изменённых еще весов вычитаем $ \lambda w^{l}_{n, k}$ */
		ArtiomArrayUtils.multiplyArrayByScalar(weight, 1.0 - LEARNING_RATE * regularizationL2);


        /* Массив градиента уже сформирован */
        /* Теперь нужно из массива весов вычесть */
        /* NB: здесь мы делим на кол-во элементов элементов. Вообще, это деление должно быть в цикле метода calculateBackpropagationGradient() */
        /* Но с целью оптимизации я перенёс это сюда - чтобы не повторять деление в цикле */ 
        /* Поэтому в массиве costGradient хранится не совсем градиент функции ошибки - различие во множителе trainingInputs.length */

        ArtiomArrayUtils.FromArraySubtractArrayTimesAlpha(weight, costGradient,  LEARNING_RATE  / (miniBatchEndIndex - miniBatchStartIndex + 1));

    }


   

    
    public static void main(String[] args) {

    }
}