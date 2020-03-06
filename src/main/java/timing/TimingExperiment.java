/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timing;

import evaluation.storage.ClassifierResults;
import java.io.FileReader;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ostandage
 */
public class TimingExperiment {
    
    private Classifier classifier;
    private Instances test;
    private Instances train;
    private double timeForTrain;
    
    public static void main(String Args[]) throws Exception{
        IBk c = new IBk();
        Instances te = loadClassificationData("data/ECG5000_TEST.arff");
        Instances tr = loadClassificationData("data/ECG5000_TRAIN.arff");
        TimingExperiment t = new TimingExperiment(c, te, tr);


    }
    
    public TimingExperiment(Classifier classifier, Instances data, Instances train) throws Exception {
        this.classifier = classifier; 
        this.test = data;
        this.train = train;
    }


    public ClassifierResults runCrossValidation(int numFolds) throws Exception{
        ClassifierResults[] foldResults = new ClassifierResults[numFolds];
        int foldLength = train.numInstances() / numFolds;
        for (int fold = 0; fold < numFolds; fold++){

            foldResults[fold] = trainFold(fold, foldLength);
        }

        int bestIndex = 0;

        for (int fold = 0; fold < foldResults.length; fold++) {

            if (foldResults[fold].getAcc() > foldResults[bestIndex].getAcc()) {
                bestIndex = fold;
            }
        }

        return foldResults[bestIndex];
    }

    public ClassifierResults trainFold(int foldIndex, int foldLength) throws Exception{
        ClassifierResults foldResult = new ClassifierResults();
        foldResult.setTimeUnit(TimeUnit.NANOSECONDS);

        Instances trainFoldData = new Instances(train);
        Instances testFoldData = new Instances(train, foldIndex * foldLength, foldLength);

        for (int i = (foldIndex+1)*foldLength; i < foldIndex*foldLength; i--) {
            trainFoldData.delete(i);
        }

        classifier.buildClassifier(trainFoldData);

        for (int i = 0; i < testFoldData.numInstances(); i++) {
            Instance inst = testFoldData.get(i);
            long startTime = System.nanoTime();
            double[] dist = classifier.distributionForInstance(inst);
            long time = System.nanoTime() - startTime;

            int index = 0;
            for (int j = 1; j < dist.length; j++) {
                if (dist[j] > dist[index]) {
                    index = j;
                }
            }
            foldResult.addPrediction(inst.classValue(), dist, index, time, "");
        }
        return foldResult;
    }

    public ClassifierResults runOptimalTraining(int numCVFolds) throws Exception{

        //If classifier is tuneable run CV with varing parameters, else just train.
        if (false) {

        }
        return normalTrain();
    }

    public ClassifierResults normalTrain() throws Exception{
        return trainFold(0, train.numInstances());
    }

    public ResultWrapper runExperiment(int resample) throws Exception {
        
        if (resample > 0) {
            shuffleData((test.numInstances() + train.numInstances()) * 10, resample);
        }

        double startTrain = System.nanoTime();
        classifier.buildClassifier(train);
        double trainTime = System.nanoTime() - startTrain;
        timeForTrain = trainTime;

        ClassifierResults trainResults = runOptimalTraining(10);
        double[] times = new double[test.numInstances()];

        ClassifierResults cresults = new ClassifierResults();
        cresults.setTimeUnit(TimeUnit.NANOSECONDS);

        for (int i = 0; i < test.numInstances(); i++) {
            Instance inst = test.get(i);
            double startTime = System.nanoTime();
            double[] dist = classifier.distributionForInstance(inst);
            double time = System.nanoTime() - startTime;
            times[i] = time;

            int maxIndex = 0;
            for (int j = 1; j < dist.length; j++) {
                if (dist[j] > dist[maxIndex]) {
                    maxIndex = j;
                }
            }
            cresults.addPrediction(inst.classValue(), dist, maxIndex, (long)time, "");
        }

        TimingResults tresults = new TimingResults(times, trainTime, null);
        cresults.finaliseResults();
        System.out.println("Accuracy: " + cresults.getAcc());

        return new ResultWrapper(tresults, cresults, trainResults);
    }

    private void shuffleData(int numSwaps, long seed) {
        int numTrain = train.numInstances();
        Instances all = new Instances(train);
        all.addAll(test);
        Random rnd = new Random(seed);
        for (int swap = 0; swap < numSwaps; swap++) {
            all.swap((rnd.nextInt(Integer.MAX_VALUE) % all.numInstances()), (rnd.nextInt(Integer.MAX_VALUE) % all.numInstances()));
        }
        train = new Instances(all, 0, numTrain);
        test = new Instances(all, numTrain, all.numInstances()-numTrain);
    }
    
    private void train() throws Exception {
        double startTime = System.nanoTime();
        classifier.buildClassifier(train);
        timeForTrain = System.nanoTime() - startTime;
    }
    
    private static Instances loadClassificationData(String path) {
        Instances train;
       
        FileReader reader;
        try
        {
            reader = new FileReader(path);
            train = new Instances(reader);
            train.setClassIndex(train.numAttributes() - 1);
            return train;
            
        } catch (Exception e)
        {
            System.out.println("Exception: " + e);
        }
        return null;
    }
      
    
    public double getTrainTime() {
        return timeForTrain;
    }
    
    public static String timingResultArrayToString(TimingResults[] results) throws Exception{
        String output = "";
        int i = 0;
        for (TimingResults result : results) {
            if (i == 0) {
                output = output + result + "\n";
            }
            else {
                output = output + ",," + result + "\n";
            }

            i++;
        }
        return output;
    }

}
