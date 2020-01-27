package labs;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

import static labs.WekaTools.*;

public class Lab2 {

    public static void main(String args[]) throws Exception {
        Instances arsenalTrain = Lab1.loadData("data/labsdata/Arsenal_TRAIN.arff");
        Instances arsenalTest = Lab1.loadData("data/labsdata/Arsenal_TEST.arff");

        Instances[] data2 = WekaTools.splitData(WekaTools.loadClassificationData("data/labsdata/Aedes_Female_VS_House_Fly_POWER.arff"),0.7);

        arsenalTrain.setClassIndex(arsenalTrain.numAttributes() -1);
        arsenalTest.setClassIndex(arsenalTest.numAttributes() -1);

        System.out.println("Arsenal");
        OneNN oneNN = new OneNN();
        oneNN.buildClassifier(arsenalTrain);
        for (Instance i : arsenalTest) {
            System.out.println(Lab1.printDoubleArray(oneNN.distributionForInstance(i)));
        }

        System.out.println("OneNN Accuracy:" + WekaTools.accuracy(oneNN, arsenalTest));
        System.out.println(Lab1.printDoubleArray(WekaTools.classDistribution(arsenalTrain)));
        System.out.println();

        System.out.println("Aedes Fly");
        oneNN.buildClassifier(data2[0]);
        for (Instance i : data2[1]) {
            System.out.println(Lab1.printDoubleArray(oneNN.distributionForInstance(i)));
        }

        System.out.println("OneNN Accuracy:" + WekaTools.accuracy(oneNN, data2[1]));
        System.out.println(Lab1.printDoubleArray(WekaTools.classDistribution(data2[0])));
        System.out.println();

        System.out.println("Football Player Problem");
        Instances football = WekaTools.loadClassificationData("data/labsdata/FootballPlayers.arff");
        System.out.println("Num attributes:    " + football.numAttributes());
        System.out.println("Num instances:     " + football.numInstances());
        System.out.println("Num classes:       " + football.numClasses());
        System.out.println("Distribution:      " + WekaTools.printDoubleArray(WekaTools.classDistribution(football)));

        Instances[] data3 = WekaTools.splitData(football, 0.7);
        Instances footballTrain = data3[0];
        Instances footballTest = data3[1];
        oneNN.buildClassifier(footballTrain);
        System.out.println("OneNN Accuracy:    " + WekaTools.accuracy(oneNN, footballTest));

        IBk ibk = new IBk();
        IBk ib1 = new IBk();
        ib1.setKNN(1);
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        SMO smo = new SMO();

//        ibk.buildClassifier(footballTrain);
//        ib1.buildClassifier(footballTrain);
//        mlp.buildClassifier(footballTrain);
//        smo.buildClassifier(footballTrain);
//        System.out.println("IBk Accuracy:      " + WekaTools.accuracy(ibk, footballTest));
//        System.out.println("IB1 Accuracy:      " + WekaTools.accuracy(ib1, footballTest));
//        System.out.println("Mul L P Accuracy:  " + WekaTools.accuracy(mlp, footballTest));
//        System.out.println("Sup Vec Accuracy:  " + WekaTools.accuracy(smo, footballTest));

        KNN knn5 = new KNN(5);
        knn5.buildClassifier(footballTrain);
        System.out.println("KNN(5) Accuracy:   " + WekaTools.accuracy(knn5, footballTest));


        System.out.println("\nNumeric Only");

        Instances footballNumeric = WekaTools.loadClassificationData("data/labsdata/FootballPlayers.arff");
        footballNumeric.deleteAttributeAt(3);
        footballNumeric.deleteAttributeAt(2);
        footballNumeric.deleteAttributeAt(1);
        footballNumeric.deleteAttributeAt(0);

        Instances[] data3Numeric = WekaTools.splitData(footballNumeric, 0.7);
        ibk.buildClassifier(data3Numeric[0]);
        System.out.println("IBk Accuracy:      " + WekaTools.accuracy(ibk, data3Numeric[1]));

        System.out.println("KNN Varying K");
        System.out.println("k,Accuracy");
        for (int k = 1; k < 100; k+=2) {
            KNN iKNN = new KNN(k);
            iKNN.buildClassifier(footballTrain);
            System.out.println(k + "," + WekaTools.accuracy(iKNN, footballTest));
        }

        //lab 3


    }

    public static class OneNN extends AbstractClassifier {
        Instances train;

        @Override
        public void buildClassifier(Instances instances) throws Exception {
            train = instances;
        }

        public double classifyInstance(Instance instance) {
            int closestIndex = 0;
            double closestDistance = Integer.MAX_VALUE;

            for (int i = 0; i < train.numInstances(); i++) {
                double distance = distance(instance, train.get(i));
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestIndex = i;
                }
            }
            return train.instance(closestIndex).classValue();
        }

        public double distance(Instance a, Instance b) {
            double distance = 0;

            for (int i = 0; i < a.numAttributes() -1; i++) {
                distance = distance + Math.pow((a.value(i) - b.value(i)), 2);
            }
            return Math.pow(distance, 0.5);
        }

        public double[] distributionForInstance(Instance instance) {
            double[] distribution = new double[instance.numClasses()];
            distribution[(int)classifyInstance(instance)] = 1.0;
            return distribution;
        }
    }

    public static class KNN extends AbstractClassifier {

        Instances train;
        int k;

        public KNN(int k) {
            this.k = k;
        }

        @Override
        public void buildClassifier(Instances instances) throws Exception {
            train = instances;
        }

        public double classifyInstance(Instance instance) {
            //the closest at pos 0.
            int[] closestIndexes = new int[k];
            double[] closestDistance = new double[k];
            for (int d = 0; d < closestDistance.length; d++) {
                closestDistance[d] = Integer.MAX_VALUE;
                closestIndexes[d] = -1;
            }

            for (int i = 0; i < train.numInstances(); i++) {
                double distance = distance(instance, train.get(i));

                if (distance < closestDistance[closestDistance.length-1]) {
                    int[] newIndexArray = new int[k];
                    double[] newDistArray = new double[k];

                    int newDistPos = -1;

                    for (int j = k-1; j >= 0; j--) {
                        if (distance < closestDistance[j]) {
                            newDistPos = j;
                        }
                        else {
                            break;
                        }
                    }

                    for (int j = 0; j < newDistPos; j++) {
                        newDistArray[j] = closestDistance[j];
                        newIndexArray[j] = closestIndexes[j];
                    }
                    newDistArray[newDistPos] = distance;
                    newIndexArray[newDistPos] = i;
                    for (int j = newDistPos + 1; j < newDistArray.length; j++) {
                        newDistArray[j] = closestDistance[j - 1];
                        newIndexArray[j] = closestIndexes[j - 1];
                    }
                    closestIndexes = newIndexArray;
                    closestDistance = newDistArray;
                }
            }


            //Problem from here?
            int[] classCount = new int[instance.numClasses()];
            for (int n = 0; n < k; n++) {
                for (int c = 0; c < classCount.length; c++) {
                    classCount[c] += train.instance(n).value(c);
                }
            }


            int modeIndex = 0;
            for (int i = 0; i < classCount.length; i++) {
                if (classCount[i] > classCount[modeIndex]) {
                    modeIndex = i;
                }
            }
            return train.instance(modeIndex).classValue();
        }


        private double distance(Instance a, Instance b) {
            double distance = 0;

            for (int i = 0; i < a.numAttributes() -1; i++) {
                distance = distance + Math.pow((a.value(i) - b.value(i)), 2);
            }
            return Math.pow(distance, 0.5);
        }


    }
}
