package timing;

import timeseriesweka.classifiers.distance_based.FastDTW_1NN;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.FileWriter;
import java.util.Random;

import static timing.NewRunner.createClassifierArray;
import static timing.NewRunner.loadData;

public class ReducedTrainDataTesting {

    private static String DataPath;
    private static double PercentageTrainToUse;

    public static void main(String[] args) throws Exception {
        DataPath = "data/Univariate_arff/ChlorineConcentration";


        Instances dataTrain = loadData(DataPath, NewRunner.DatasetType.TRAIN);
        Instances dataTest = loadData(DataPath, NewRunner.DatasetType.TEST);

        double[] trainProportions = {0.2, 0.4, 0.6, 0.8, 1};
        Instances[] trainSplits = new Instances[trainProportions.length];

        Random rnd = new Random();

        for (int i = 0; i < trainProportions.length; i++) {
            trainSplits[i] = new Instances(dataTrain);
            int numToDelete = (int) (dataTrain.numInstances() * (1-trainProportions[i]));
            for (int j = 0; j < numToDelete; j++) {
                trainSplits[i].delete(rnd.nextInt(trainSplits[i].numInstances()));
            }
        }

        Classifier[] classifiers = createClassifierArray();
        Evaluation eval = new Evaluation(dataTrain);

        System.out.println("ClassifierIndex,ProportionTrainData,Accuracy,AvgClassifyTime,TotalClassifyTime,TrainTime");
        FileWriter csv = new FileWriter("results/reducedData.csv");
        csv.append("ClassifierIndex,ProportionTrainData,Accuracy,AvgClassifyTime,TotalClassifyTime,TrainTime\n");
        csv.flush();

        //Parallel
        ProcessClassifer[] threads = new ProcessClassifer[classifiers.length];
        for (int c = 0; c < classifiers.length; c++) {
            threads[c] = new ProcessClassifer(csv, c, classifiers, trainSplits, trainProportions, dataTest);
            threads[c].start();
        }

    }

    public static class ProcessClassifer extends Thread {


        private final FileWriter csv;
        private final int classifierIndex;
        private final Classifier[] classifiers;
        private final Instances[] trainSplits;
        private final double[] splitProportions;
        private final Instances testData;

        public ProcessClassifer (FileWriter csv, int classifierIndex, Classifier[] classifiers, Instances[] trainSplits, double[] trainProportions, Instances testData) {
            this.csv = csv;
            this.classifierIndex = classifierIndex;
            this.classifiers = classifiers;
            this.trainSplits = trainSplits;
            this.splitProportions = trainProportions;
            this.testData = testData;
        }

        @Override
        public void run() {
            try {
                for (int i = 0; i < trainSplits.length; i++) {
                    TimingExperiment tm0 = new TimingExperiment(classifiers[classifierIndex], testData, trainSplits[i]);
                    ResultWrapper rw0 = tm0.runExperiment(0);
                    double accuracy = rw0.getClassifierResults().getAcc();
                    double avgClassify = rw0.getTimingResults().averageTime();
                    double totalClassify = rw0.getTimingResults().totalClassifyTime();
                    double trainTime = rw0.getTimingResults().trainTime();
                    String out = classifierIndex + "," + splitProportions[i] + "," + accuracy + "," + avgClassify + "," + totalClassify + "," + trainTime + "\n";
                    System.out.print(out);
                    csv.append(out);
                    csv.flush();
                }





            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

}
