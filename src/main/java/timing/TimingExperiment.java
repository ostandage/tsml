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

import timeseriesweka.classifiers.TrainAccuracyEstimator;
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
    private Instances data;
    private Instances train;
    private double timeForTrain;
    
    public static void main(String Args[]) throws Exception{
        IBk c = new IBk();
        Instances te = loadClassificationData("data/ECG5000_TEST.arff");
        Instances tr = loadClassificationData("data/ECG5000_TRAIN.arff");
        TimingExperiment t = new TimingExperiment(c, te, tr);
        ResultWrapper rw = t.runNormalExperiment(30, 123456789);
        
        ClassifierResults[] cres = rw.getClassifierResults();
        TimingResults[] eres = rw.getTimingResults();
        
        System.out.println("\n********** Timing **********");
        System.out.println("Average Classify Time,Total Classify Time,Train Time");
        System.out.println(timingResultArrayToString(eres));
        
        System.out.println("\n********** Timing Combined **********");
        TimingResults rcom = TimingResults.combineResults(eres);
        System.out.println("Average Classify Time,Total Classify Time,Train Time");
        System.out.println(rcom);
        
        System.out.println("\n********** Classifier Results **********");
        System.out.println(cres[0].writeSummaryResultsToString());

    }
    
    public TimingExperiment(Classifier classifier, Instances data, Instances train) throws Exception {
        this.classifier = classifier; 
        this.data = data;
        this.train = train;
    }
    
    public ResultWrapper runNormalExperiment(int numResamples, long resampleSeed) throws Exception {
        return runExperiment(numResamples, (data.numInstances() + train.numInstances()) * 5, resampleSeed);
    }
    
    public ResultWrapper runExperiment(int numResamples, int numSwaps, long resampleSeed) throws Exception {
        
        TimingResults[] tresults = new TimingResults[numResamples];
        ClassifierResults[] cresults = new ClassifierResults[numResamples];
        
        for (int run = 0; run < numResamples; run++) {
            shuffleData(numSwaps, resampleSeed);

            double startTrain = System.nanoTime();
            train();
            double trainTime = System.nanoTime() - startTrain;

            TrainAccuracyEstimator tae = null;
//            try {
//                tae = (TrainAccuracyEstimator) classifier;
//            }
//            catch (Exception e){
//                throw new Exception();
//            }

            double[] times = new double[data.numInstances()];

            cresults[run] = new ClassifierResults();
            cresults[run].setTimeUnit(TimeUnit.NANOSECONDS);

            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.get(i);
                double startTime = System.nanoTime();
                double[] dist = classifier.distributionForInstance(inst);
                double time = System.nanoTime() - startTime;
                times[i] = time;

                int index = 0;
                for (int j = 1; j < dist.length; j++) {
                    if (dist[j] > dist[index]) {
                        index = j;
                    }
                }

                cresults[run].addPrediction(inst.classValue(), dist, dist[index], (long)time, "");
            }

            tresults[run] = new TimingResults(times, trainTime, tae);
        }
        return new ResultWrapper(tresults, cresults);
    }

    //need to pass in the same seed for each classifier.
    private void shuffleData(int numSwaps, long seed) {
        int numTrain = train.numInstances();
        Instances all = new Instances(train);
        all.addAll(data);
        Random rnd = new Random(seed);
        for (int swap = 0; swap < numSwaps; swap++) {
            all.swap((rnd.nextInt(Integer.MAX_VALUE) % all.numInstances()), (rnd.nextInt(Integer.MAX_VALUE) % all.numInstances()));
        }
        train = new Instances(all, 0, numTrain);
        data = new Instances(all, numTrain, all.numInstances()-numTrain);
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
