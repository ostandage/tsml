/**
 * This class allows a single fold of a classifier-dataset combination to be benchmarked for accuracy and time.
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

public class TimingExperiment {
    
    private Classifier classifier;
    private Instances test;
    private Instances train;
    private double timeForTrain;
    
    public TimingExperiment(Classifier classifier, Instances data, Instances train) {
        this.classifier = classifier; 
        this.test = data;
        this.train = train;
    }

    //This method estimates the accuracy of a classifier based on just training data.
    public ClassifierResults getTrainAccuracy(int foldIndex, int foldLength) throws Exception{
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

    public ResultWrapper runExperiment(int resample) throws Exception {
        if (resample > 0) {
            shuffleData((test.numInstances() + train.numInstances()) * 10, resample);
        }

        int foldSize = (int) (train.numInstances() * 0.2);
        ClassifierResults trainResults = getTrainAccuracy(0, foldSize);

        double startTrain = System.nanoTime();
        classifier.buildClassifier(train);
        timeForTrain = System.nanoTime() - startTrain;

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

        TimingResults tresults = new TimingResults(times, timeForTrain);
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
