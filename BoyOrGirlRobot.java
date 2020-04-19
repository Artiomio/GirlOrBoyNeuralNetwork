import java.io.IOException;
import static java.awt.event.KeyEvent.*;
import java.util.Arrays;
import java.util.Random;

class BoyOrGirlRobot {

    public static void main(String[] args) {

        try {

            /* создаём экземпляр считывателя базы данных MagicDistancesDataSetReader */
            MagicDistancesDataSetReader dataSetReader = new MagicDistancesDataSetReader();
            dataSetReader.shuffleDataSet(1000000, System.currentTimeMillis()); /* Перемешивание рандомизировано таймером! */
            
            final int TOTAL_NUMBER_OF_FACES = dataSetReader.TOTAL_NUMBER_OF_FACES;

            /* теперь создаём экзмепляр нейронную сеть */
            NetworkOne network = new NetworkOneTanh(2 * 68, /* Количество входных нейронов */
                                               new int[] {10, 10, 1} /* Первый и второй (последний) слои */);

            /* инициализируем параметры обучения */
            network.LEARNING_RATE = 0.003;
            int miniBatchSize = 3;
            network.regularizationL2 = 0.000001;
            network.initializeWeightsWithNormalRandomNumbers(0.1);


            int numberOfTrainingItems = 55000; /* Количество образцов, используемых для ОБУЧЕНUЯ */
            int numberOfTestingItems = 5000; /* Количество последних (отсчёт справа) образцов для ВАЛИДАЦUU */

            
            
            
            network.DELTA_W = 0.01;
            network.randomWeightAdditionRange = 0.01;
            

            /* создаём валидатор */
            NeuralOneOrZeroValidator validator = new NeuralOneOrZeroValidator(network, dataSetReader.magicDistances, dataSetReader.bitNumber);
            
            /* устанавливаем набор данных обучения */
            network.setTrainingSet(dataSetReader.magicDistances, dataSetReader.desiredOutputs);

          

            ArtConsole console = new ArtConsole(30, 35);
            int keyCode = 0;


            int preliminaryValidation =  validator.validate(0, 1000 - 1);
            console.println("Preliminary validation (must be ~50%): " + preliminaryValidation + " / " + 1000 + " (" + (preliminaryValidation / 1000) +"%)");




            
            double error = 0;

            boolean backpropagationAllowed = true;
            int numberOfBackpropCycles = 50;

            boolean randomWalkAllowed = false;
            int numberOfRandomWalkCycles = 20;



            console.println("Calculating mean square error");
            network.setCurrentMiniBatchRange(0, dataSetReader.TOTAL_NUMBER_OF_FACES - 1);
            error = network.getMeanSquareError();
            console.println("Ready");            
            System.err.println(error);

            int epoch = 0;
            
            /* Сохраняем время запуска программы */
            long programStartTime = System.currentTimeMillis();


            int maxValidation = 0;
            double maxValidationCostFunction = 0;

            /* Чтобы следить за ростом */
            double lastError = 0;

            while (true) {
                if (console.keyPressed) { /* Обработка нажатий клавиш */
                    console.clearKey();
                    keyCode = console.keyCode;

                    if (keyCode == VK_ADD) {
                        network.LEARNING_RATE *= 1.1;
                        console.println("Increasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                    }

                    if (keyCode == VK_SUBTRACT) {
                        network.LEARNING_RATE /= 1.1;
                        console.println("Decreasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                    }

                    if (keyCode == VK_MULTIPLY) {
                        network.LEARNING_RATE *= 10;
                        console.println("Increasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                    }

                    if (keyCode == VK_DIVIDE) {
                        network.LEARNING_RATE /= 10;
                        console.println("Decreasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                    }


                    if (keyCode == VK_PAGE_UP) {
                        network.randomWeightAdditionRange *= 2;
                        console.println("Increasing Random step max size! Now it is " + network.randomWeightAdditionRange);
                    }

                    if (keyCode == VK_PAGE_DOWN) {
                        network.randomWeightAdditionRange /= 2;
                        console.println("Decreasing random step max size! Now it is " + network.randomWeightAdditionRange);
                    }

                    
                    if (keyCode == VK_R) {
                        console.println("Reinitializing weights");
                        network.initializeWeightsWithNormalRandomNumbers(0.01);
                        programStartTime = System.currentTimeMillis();
                    }

                    if (keyCode == VK_Z) {
                        console.println("Zeroing weights");
                        ArtiomArrayUtils.zeroFill3DArray(network.weight);
                        programStartTime = System.currentTimeMillis();
                    }

                    
                    if (keyCode == VK_B)
                        backpropagationAllowed = !backpropagationAllowed;

                    if (keyCode == VK_F)
                        randomWalkAllowed = !randomWalkAllowed;


                    if (keyCode == VK_H) {
                        console.println("Wow! We're gonna fantasize big!");

                        int miniBatchSize2 = 50000;
                        for (int i=0; i + miniBatchSize2 - 1 <= numberOfTrainingItems; i += miniBatchSize2) {

                            /* Устанавливаем границы мини-пакета в цикле по тренировочным данным */
                            network.setCurrentMiniBatchRange(i, i + miniBatchSize2 - 1);


                            network.teachByShakingWeights(30);

                            if (i % 2000 == 0) {
                                console.print("☻");
                            }
                        }

                
                    }

                    
                    if (keyCode == VK_S) { /* Сохраняем веса */
                        ArtiomArrayUtils.serialize3DArray(network.weight, "neural_weights.serialized");
                    }

                    if (keyCode == VK_L) { /* Восстанавливаем веса */
                        network.weight = ArtiomArrayUtils.loadSerialized3DArray("neural_weights.serialized");
                        ArtiomArrayUtils.write3DArrayToFilePythonStyle(network.weight, "neural_weights.python");
                    }


                    if (keyCode == VK_W) { /* Перемешиваем набор данных */
                        dataSetReader.shuffleDataSet(1000000, System.currentTimeMillis());
                        console.println("Shuffling data set!");

                    }

                    
                    /* Переформирование набора данных: оставляем неудачно решенные задачки (около 10% или около 5.5 тысяч) */
                    /* и дополняем случайными остальными, которых пусть для начала будет в два раза больше - 11 тыс к примеру */
                    if (keyCode == VK_F12) {

                        System.out.println("Hardcore part!");
                        int positives = validator.validate(0, numberOfTrainingItems - 1);
                        int negatives = numberOfTrainingItems - positives;
                        int numberOfHardcoreTrainingItems = negatives * 3;
                        System.out.println("Number of negatives " + negatives);
                        double[][] hardcoreDistances = new double[numberOfHardcoreTrainingItems][];
                        double[][] hardcoreDesiredOutputs = new double[numberOfHardcoreTrainingItems][];
                        int[] hardcoreBitNumber = new int[numberOfHardcoreTrainingItems];


                        
                        int j = 0;
                        for (int i=0; i < numberOfTrainingItems; i++) {
                            if (validator.validate(i) == false) { /* Добавляем образцы, с которыми сеть не справилась */
                                hardcoreDistances[j]      = dataSetReader.magicDistances[i];
                                hardcoreDesiredOutputs[j] = dataSetReader.desiredOutputs[i];
                                hardcoreBitNumber[j] = dataSetReader.bitNumber[i];
                                j++;
                            }
                        }


                        /* Разбавляем теми, с которыми сеть справилась  */
                        for (int i=0; i < numberOfTrainingItems; i++) {
                            if (validator.validate(i) == true) {
                                hardcoreDistances[j]      = dataSetReader.magicDistances[i];
                                hardcoreDesiredOutputs[j] = dataSetReader.desiredOutputs[i];
                                hardcoreBitNumber[j] = dataSetReader.bitNumber[i];
                                j++;
                                if (j == numberOfHardcoreTrainingItems) {
                                    break;
                                }
                            }
                        }


                        
                        /* Перемешиваем - нужно перемешать тройки (hardcoreDistances, hardcoreDesiredOutputs, hardcoreBitNumber) */
                        /* Черновой вариант */
                                                              Random random = new Random(10);
                                                              for (long i=0; i < j * 1000; i++) {
                                                                  int x = random.nextInt(j);
                                                                  int y = random.nextInt(j);
                                                                  if (x != y) {
                                                                      double[] tmpDistances = hardcoreDistances[x];
                                                                      double[] tmpLabel = hardcoreDesiredOutputs[x];
                                                                      
                                                                      hardcoreDistances[x] = hardcoreDistances[y];
                                                                      hardcoreDesiredOutputs[x] = hardcoreDesiredOutputs[y];
                                                                      
                                                                      
                                                                      hardcoreDistances[y] = tmpDistances;
                                                                      hardcoreDesiredOutputs[y] = tmpLabel;
                                                                      
                                                                      
                                                                      
                                                                      int tmp = hardcoreBitNumber[x];
                                                                      hardcoreBitNumber[x] = hardcoreBitNumber[y];
                                                                      hardcoreBitNumber[y] = tmp;
                                                                      
                                                                  }
                                                                  
                                                              }


                        
                        
                        /* Окончаение перемешивания */






                        
                        
                        
                        
                        if (j != numberOfHardcoreTrainingItems) {
                            System.out.println("Too few positives!");
                            throw new ArrayIndexOutOfBoundsException();

                        }
                        else /* Обучение на хардкорном наборе данных */
                        {


                            ArtConsole newConsole = new ArtConsole(25, 80);
                            /* создаём ЛОКАЛЬНЫЙ валидатор */
                            NeuralOneOrZeroValidator localValidator = new NeuralOneOrZeroValidator(network, hardcoreDistances, hardcoreBitNumber);
                            
                            console.println("j =" + j + "\n" + 
                                                "hardcoreDistances.length = " + hardcoreDistances.length + "\n" +
                                                "hardcoreDesiredOutputs.length = " + hardcoreDesiredOutputs.length + "\n" +
                                                "numberOfHardcoreTrainingItems = " + numberOfHardcoreTrainingItems);

                            while (true) {

                                network.setTrainingSet(hardcoreDistances, hardcoreDesiredOutputs);
                                for (int i=0; i + miniBatchSize - 1 <= numberOfHardcoreTrainingItems; i += miniBatchSize) {

                                    /* Устанавливаем границы мини-пакета в цикле по тренировочным данным */
                                    network.setCurrentMiniBatchRange(i, i + miniBatchSize - 1);

                                    /* Минимизируем функцию ошибки на данном мини-пакете */
                                    try {
                                        network.teachWithBackpropagation(); // пока закомментировано - удостоверюсь что валидация = 1/3
                                    }
                                    catch (Exception e) {
                                        console.println("i =" + i + "\n");
                                        throw e;
                                                         
                                    }

                                    if (i % 1000 == 0) {
                                        newConsole.print("▒");
                                    }

                                    if (newConsole.keyPressed) { /* Обработка нажатий клавиш */
                                        newConsole.clearKey();
                                        keyCode = newConsole.keyCode;
                                        if (keyCode == VK_ADD) {
                                            network.LEARNING_RATE *= 1.1;
                                            console.println("Increasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                                        }

                                        if (keyCode == VK_SUBTRACT) {
                                            network.LEARNING_RATE /= 1.1;
                                            console.println("Decreasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                                        }

                                        if (keyCode == VK_MULTIPLY) {
                                            network.LEARNING_RATE *= 10;
                                            console.println("Increasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                                        }

                                        if (keyCode == VK_DIVIDE) {
                                            network.LEARNING_RATE /= 10;
                                            console.println("Decreasing Learning rate! Now Learning rate is " + network.LEARNING_RATE);
                                        }

                                        if (keyCode == VK_R) {
                                            console.println("Reinitializing weights");
                                            network.initializeWeightsWithNormalRandomNumbers(1);
                                            programStartTime = System.currentTimeMillis();
                                        }


                                    }
                                } /* for */
                                    
                                                                
                                network.setTrainingSet(hardcoreDistances, hardcoreDesiredOutputs);                              
                                int localPositives = localValidator.validate(0, numberOfHardcoreTrainingItems - 1);
                                
                                

                                network.setCurrentMiniBatchRange(0 , numberOfHardcoreTrainingItems - 1);
                                double localError = network.getMeanSquareError();
                                
                                





                                /* Переключаемся на основной набор данных - чтобы проверять на нём валидацию */
                                network.setTrainingSet(dataSetReader.magicDistances, dataSetReader.desiredOutputs);

                                network.setCurrentMiniBatchRange(0 , numberOfTrainingItems - 1);
                                error = network.getMeanSquareError();

                                

                                System.err.println(error);
                                

                                positives = validator.validate(TOTAL_NUMBER_OF_FACES - numberOfTestingItems, TOTAL_NUMBER_OF_FACES - 1);

                                

                                newConsole.clearScreen();
                                newConsole.println("\nLOCAL Validation: " + localPositives + " / " + numberOfHardcoreTrainingItems + " (" + String.format("%.2f", 100.0 * localPositives / numberOfHardcoreTrainingItems) + "%)");
                                newConsole.println("Total validation: " + positives + " / " + numberOfTestingItems + " (" + String.format("%.2f", 100.0 * positives / numberOfTestingItems) + "%)");
                                newConsole.println("\nLOCAL error: " + localError);


                                newConsole.println("Total error: " + error);

                                if (error > lastError) {
                                    newConsole.println("Total Error getting bigger!");
                                }
                                else {
                                    newConsole.println("Total Error difference: " + String.format("%1.1e", lastError - error));
                                }


                                lastError = error;


                                System.err.println(localError);                             

                                newConsole.println("\nTraining with:\n    Learning rate: " + network.LEARNING_RATE);
                                

                                
                                


                            } /* While */
                        }

                        

                    } /* F12 */




                    



                } /* Обработка клавиш */



                epoch++;
                long cycleStartTime = System.currentTimeMillis();

                for (int i=0; i + miniBatchSize - 1 <= numberOfTrainingItems - 1; i += miniBatchSize) {

                    /* Устанавливаем границы мини-пакета в цикле по тренировочным данным */
                    network.setCurrentMiniBatchRange(i, i + miniBatchSize - 1);

                    /* Минимизируем функцию ошибки на данном мини-пакете */
                    if (backpropagationAllowed) network.teachWithBackpropagationL2();


                    if (randomWalkAllowed) { 
                        network.setCurrentMiniBatchRange(0, numberOfTrainingItems - 1);
                        network.teachByShakingWeights(1);
                    }

                    if (i % 1000 == 0) {
                        console.print("▒");
                    }



                  
                    
                }



                /* Вычисляем ошибку на всём тренировочном наборе (можно заменить на случайный промежуток впоследствии) */
                network.setCurrentMiniBatchRange(0 , numberOfTrainingItems - 1);

                error = network.getMeanSquareError();

                console.clearScreen();

                /* Печать ошибки, количество успешных распознаваний и модуль градиента */
                System.err.println(error);
                console.println("\nCost function: " + error);

                if (error > lastError) {
                    console.println("Error getting bigger!");
                }
                else {
                    console.println("Error difference: " + String.format("%1.1e", lastError - error));
                }


                lastError = error;

                int positives = validator.validate(TOTAL_NUMBER_OF_FACES - numberOfTestingItems, TOTAL_NUMBER_OF_FACES - 1);
                if (positives > maxValidation) 
                {
                    maxValidation = positives;
                    maxValidationCostFunction = error;

                    if (positives / numberOfTestingItems > 0.9)
                        ArtiomArrayUtils.serialize3DArray(network.weight, "max_validation_weights.serialized");
                }
                console.println("Validation: " + positives + " / " + numberOfTestingItems + " (" + String.format("%.2f", 100.0 * positives / numberOfTestingItems) + "%)");
                console.println("\n_______________________\nMax validation: " + maxValidation + "\nMax validation error: " + maxValidationCostFunction + "\n");
                console.println("Test accuracy: " + validator.validate(0, numberOfTrainingItems - 1) + " / " + numberOfTrainingItems + "\n");                
                console.println("Cost gradient absolute value: " + ArtiomArrayUtils.abs(network.costGradient));

                console.println("Backpropagation: " + (backpropagationAllowed? "ON " : "OFF") + "\nRandom walk: " + (randomWalkAllowed? "ON" : "OFF"));
                console.println("Training with:\n    Learning rate: " + network.LEARNING_RATE + 
                                "\n    Random max step: " + network.randomWeightAdditionRange +
                                "\nMini-batch size: " + miniBatchSize + "\n");
            
                
                console.println("Time spent on the last training cycle: " + (System.currentTimeMillis() - cycleStartTime) + " ms");
                console.println("Time elapsed: " + (System.currentTimeMillis() - programStartTime) / 1000 + " seconds");
                console.println("Epoch number: " + epoch);

                console.println("Number of neurons in the 1st hidden layer: " + network.weight[0].length);
                
                
            }

            /*
            System.out.println("Bye! See you soon!");
            System.exit(0);
            */



       } /* Выше - если не было ошибки с открытием базы данных с цифрами MNIST */

       catch (IOException e) {
           System.out.println("Error opening data base! Exiting.");
       }
    }

}


