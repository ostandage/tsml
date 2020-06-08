/**
 * Class to experiment with reduced training data. The dataset and output path is passed in via command line args.
 * Can optionally use threads to increase speed.
 */
package timing;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.io.FileWriter;
import java.util.Random;
import static timing.TimingRunner.createClassifierArray;
import static timing.WekaTools.loadData;

public class ReducedTrainDataTesting {

    private static String DataPath;
    private static boolean UseThreads = true;
    private static String DatasetName;
    private static String OutputPath;


    public static void main(String[] args) throws Exception {

        DataPath = "data/Univariate_arff/GunPoint";
        OutputPath = "results/";

        if (args.length > 0) {
            DataPath = args[0];
            OutputPath = args[1];
            UseThreads = false;
        }

        String[] pathSplit = DataPath.split("/");
        DatasetName = pathSplit[pathSplit.length-1];

        Instances dataTrain = loadData(DataPath, TimingRunner.DatasetType.TRAIN);
        Instances dataTest = loadData(DataPath, TimingRunner.DatasetType.TEST);

        //Tests 20%, 40%, 60%, and 100% of the training data.
        double[] trainProportions = {0.2, 0.4, 0.6, 0.8, 1};
        Instances[] trainSplits = new Instances[trainProportions.length];

        Random rnd = new Random();

        //randomly remove x percent of the train data.
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
        FileWriter csv = new FileWriter( OutputPath + "reducedData" + DatasetName + ".csv");
        csv.append("ClassifierIndex,ProportionTrainData,Accuracy,AvgClassifyTime,TotalClassifyTime,TrainTime\n");
        csv.flush();

        //Create a thread for each classifier and start running.
        ProcessClassifer[] threads = new ProcessClassifer[classifiers.length];
        for (int c = 0; c < classifiers.length; c++) {
            threads[c] = new ProcessClassifer(csv, c, classifiers, trainSplits, trainProportions, dataTest);
            if (UseThreads) {
                threads[c].start();
            }
            else {
                //runs sequentially.
                threads[c].run();
            }
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
