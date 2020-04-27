//Run timing experiment to find best accuracy when varying the window size.
//Run timing experiment just changing R (window size), K=1 (default)
//Run timing experiemnt just changing K (NN), R=1 (default)
//Run timing experiment changing both.
//Already have data for both on default.


package timing;

import weka.core.Instances;

import java.io.FileWriter;
import java.util.Random;

public class DTWTrainingExperiments {

    private static Instances[] trainTrainData;
    private static Instances[] testTrainData;
    private static int numFolds;
    private static int threads;

    public static void main(String[] args) throws Exception {

        String dataset = "ArrowHead";
        String datapath = "data/Univariate_arff/" + dataset + "/"+ dataset;
        String resultsPath = "results/DTWTraining/" + dataset + ".csv";

        if (args.length > 0) {
            dataset = args[0];
            datapath = args[1];
            resultsPath = args[2];
        }

        Instances train = WekaTools.loadClassificationData(datapath + "_TRAIN.arff");
        Instances test =  WekaTools.loadClassificationData(datapath +"_TEST.arff");

        numFolds = 10;
        threads = 4;
        setupTrainData(train);

        FileWriter csv = new FileWriter(resultsPath);
        csv.append("Dataset,Description,R,K,Time,Accuracy\n");
        csv.flush();

        //For the findOptimal methods, time and accuracy are the train time and train accuracy.
        //-TEST accuracy is for the test data.
        System.out.println("Dataset,Description,R,K,BuildTime,Accuracy");

        long start = System.nanoTime();
        FastDTW_1NN settingR = findOptimalR(100, 1, "settingR", dataset, csv);
        long settingRTime = System.nanoTime() - start;

        start = System.nanoTime();
        FastDTW_1NN settingK = findOptimalK(1.0, 2, 100, "settingK", dataset, csv);
        long settingKTime = System.nanoTime() - start;

        //Setting K given optimal R already found.
        start = System.nanoTime();
        FastDTW_1NN settingRK = findOptimalK(settingR.getR(), 2, 100, "settingRK", dataset, csv);
        long settingRKTime = (System.nanoTime() - start) + settingRTime;

        settingR.buildClassifier(train);
        double acc = WekaTools.accuracy(settingR, test);
        System.out.print(dataset + ",settingR-TEST," + settingR.getR() +","+ settingR.getK() +","+ settingRTime +","+ acc + "\n");
        csv.append(dataset + ",settingR-TEST," + settingR.getR() +","+ settingR.getK() +","+ settingRTime +","+ acc + "\n");

        settingK.buildClassifier(train);
        acc = WekaTools.accuracy(settingK, test);

        System.out.print(dataset + ",settingK-TEST," + settingK.getR() +","+ settingK.getK() +","+ settingKTime +","+ acc + "\n");
        csv.append(dataset + ",settingK-TEST," + settingK.getR() +","+ settingK.getK() +","+ settingKTime +","+ acc + "\n");

        settingRK.buildClassifier(train);
        acc = WekaTools.accuracy(settingRK, test);
        System.out.print(dataset + ",settingRK-TEST," + settingRK.getR() +","+ settingRK.getK() +","+ settingRKTime +","+ acc + "\n");
        csv.append(dataset + ",settingRK-TEST," + settingRK.getR() +","+ settingRK.getK() +","+ settingRKTime +","+ acc + "\n");

        csv.close();
    }

    public static FastDTW_1NN findOptimalK(double r, int increment, int maxK, String desc, String dataset, FileWriter csv) throws Exception {
        int c = 0;
        double[] accuracies = new double[maxK/increment];
        FastDTW_1NN[] classifiers = new FastDTW_1NN[maxK/increment];


        for (int k = 1; k < maxK; k += increment) {
            classifiers[c] = new FastDTW_1NN();
            classifiers[c].setMaxNoThreads(4);
            classifiers[c].setK(k);
            classifiers[c].setR(r);

            long startTrain = System.nanoTime();
            for (int f = 0; f < numFolds; f++) {
                classifiers[c].buildClassifier(trainTrainData[f]);
                double foldAccuracy = WekaTools.accuracy(classifiers[c], testTrainData[f]);
                accuracies[c] = accuracies[c] + foldAccuracy;
            }
            long trainTime = System.nanoTime() - startTrain;
            accuracies[c] = accuracies[c] / numFolds;
            System.out.print(dataset + "," + desc + "," + r + "," + k + "," + trainTime + "," + accuracies[c] + "\n");
            csv.append(dataset + "," + desc + "," + r + "," + k + "," + trainTime + "," + accuracies[c] + "\n");
            csv.flush();
            c++;
        }
        int highestAccuracyIndex = 0;
        for (int i = 0; i < accuracies.length; i++) {
            if (accuracies[i] > accuracies[highestAccuracyIndex]) {
                highestAccuracyIndex = i;
            }
        }
        return classifiers[highestAccuracyIndex];
    }

    public static void setupTrainData(Instances data) {
        trainTrainData = new Instances[numFolds];
        testTrainData = new Instances[numFolds];

        shuffleData(data, data.numInstances()*4);
        int foldLength = data.numInstances() / numFolds;
        for (int i = 0; i < numFolds; i++) {
            trainTrainData[i] = new Instances(data);
            testTrainData[i] = new Instances(data,i*foldLength, foldLength);
            for (int j = (i+1 * foldLength); j > (i*foldLength); j--) {
                trainTrainData[i].remove(j);
            }
        }
    }

    public static FastDTW_1NN findOptimalR(int wDivider, int k, String desc, String dataset, FileWriter csv) throws Exception {
        double wIncrement = (double) 1 / wDivider;
        FastDTW_1NN[] classifiers = new FastDTW_1NN[wDivider];
        double[] accuracies = new double[wDivider];

        for (int i = 0; i < wDivider; i++) {
            classifiers[i] = new FastDTW_1NN();
            classifiers[i].setMaxNoThreads(4);
            classifiers[i].setK(k);
            classifiers[i].setR(i * wIncrement);

            accuracies[i] = 0;

            long startTrain = System.nanoTime();
            for (int f = 0; f < numFolds; f++) {
                classifiers[i].buildClassifier(trainTrainData[f]);
                double foldAccuracy = WekaTools.accuracy(classifiers[i], testTrainData[f]);
                accuracies[i] = accuracies[i] + foldAccuracy;
            }
            long trainTime = System.nanoTime() - startTrain;
            accuracies[i] = accuracies[i] / numFolds;

            System.out.print(dataset + "," + desc + "," + classifiers[i].getR() + "," + k + "," + trainTime + "," + accuracies[i] + "\n");
            csv.append(dataset + "," + desc + "," + classifiers[i].getR() + "," + k + "," + trainTime + "," + accuracies[i] + "\n");
            csv.flush();
        }

        //Try changing here.
        int highestAccuracyIndex = 0;
        for (int i = 0; i < accuracies.length; i++) {
            if (accuracies[i] > accuracies[highestAccuracyIndex]) {
                highestAccuracyIndex = i;
            }
        }
        return classifiers[highestAccuracyIndex];
    }

    private static void shuffleData(Instances data, int numSwaps) {
        Random rnd = new Random();
        for (int swap = 0; swap < numSwaps; swap++) {
            data.swap((rnd.nextInt(Integer.MAX_VALUE) % data.numInstances()), (rnd.nextInt(Integer.MAX_VALUE) % data.numInstances()));
        }
    }


}
