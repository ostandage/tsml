package Coursework;

import evaluation.storage.ClassifierResults;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import weka.classifiers.Classifier;
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

    public ResultWrapper runExperiment(int resample) throws Exception {
        
        if (resample > 0) {
            shuffleData((test.numInstances() + train.numInstances()) * 10, resample);
        }

        double startTrain = System.nanoTime();
        classifier.buildClassifier(train);
        double trainTime = System.nanoTime() - startTrain;
        timeForTrain = trainTime;

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

        TimingResults tresults = new TimingResults(times, trainTime);
        cresults.finaliseResults();
        return new ResultWrapper(tresults, cresults);
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
}
