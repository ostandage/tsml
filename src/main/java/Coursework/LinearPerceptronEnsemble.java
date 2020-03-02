//Want to have some sort of mechanism to configure the enhancedLinearPerceptron.
//Seems very inefficient to be duplicating so much data. Maybe a quicker way with references / on the fly ignore.


//Have an array of attributes for each ensemble to use, then copy the build / classify methods from the elp
//class, and instead of looping through all attributes, only loop through those that are in the array.
//More efficient and less messing around with copying / references.


//Disable attributes after build. Clone the base classifier. Means we only have to build one classifier for the ensemble.
//Could just enable and disable on the fly from an array. Store class value in classifier and have a reset method.



package Coursework;

import labs.WekaTools;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class LinearPerceptronEnsemble extends EnhancedLinearPerceptron {

    private int EnsembleSize;
    private double AttributeSubsetProportion;
    private EnhancedLinearPerceptron[] Ensemble;
    private boolean BuildThreaded;

    public static void main (String[] args) throws Exception {
        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TEST.arff");

        lpe.setMaxNoIterations(100000000);
        lpe.setEnsembleSize(1);
        lpe.setAttributeSubsetProportion(1);
        lpe.setStandardiseAttributes(false);
        lpe.setUseOnlineAlgorithm(false);
        lpe.setModelSelection(false);
        lpe.BuildThreaded = true;
        lpe.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(lpe, test);
        System.out.println(eval.toSummaryString());
        System.out.println("Error Rate: " + eval.errorRate());
    }

    public LinearPerceptronEnsemble() {
        super();
        EnsembleSize = 50;
        AttributeSubsetProportion = 0.5;

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Ensemble = new EnhancedLinearPerceptron[EnsembleSize];
        Random rnd = new Random();
        int attributesToDisable = data.numAttributes() - (int) (data.numAttributes() * AttributeSubsetProportion);

        BuildThread[] bts = new BuildThread[EnsembleSize];

        for (int i = 0; i < EnsembleSize; i++) {

            Ensemble[i] = new EnhancedLinearPerceptron(data.numAttributes());
            Ensemble[i].setMaxNoIterations(MaxNoIterations);
            Ensemble[i].setStandardiseAttributes(StandardiseAttributes);
            Ensemble[i].setModelSelection(ModelSelection);
            Ensemble[i].setUseOnlineAlgorithm(UseOnlineAlgorithm);
            Ensemble[i].setLearningRate(LearningRate);
//
//            //Work out which attributes this classifier will use.
//            int count = 0;
//            while (count < attributesToDisable) {
//                int indexToDisable = rnd.nextInt(data.numAttributes());
//                if (indexToDisable != data.classIndex()) {
//                    Ensemble[i].disableAttribute(indexToDisable);
//                    count++;
//                }
//            }
//
//            Ensemble[i].buildClassifier(data);
            bts[i] = new BuildThread(i, data, rnd, attributesToDisable);
            if (BuildThreaded) {
                bts[i].start();
            }
            else {
                bts[i].run();
            }

        }

        if (BuildThreaded) {
            for (int i = 0; i < EnsembleSize; i++) {
                bts[i].join();
            }
        }

    }

    private class BuildThread extends Thread {

        private int i;
        private Instances data;
        private Random rnd;
        private int attributesToDisable;

        public BuildThread(int i, Instances data, Random rnd, int attributesToDisable) {

            this.i = i;
            this.data = data;
            this.rnd = rnd;
            this.attributesToDisable = attributesToDisable;
        }

        @Override
        public void run() {
            Ensemble[i] = new EnhancedLinearPerceptron(data.numAttributes());
            Ensemble[i].setMaxNoIterations(MaxNoIterations);
            Ensemble[i].setStandardiseAttributes(StandardiseAttributes);
            Ensemble[i].setModelSelection(ModelSelection);
            Ensemble[i].setUseOnlineAlgorithm(UseOnlineAlgorithm);
            Ensemble[i].setLearningRate(LearningRate);

            //Work out which attributes this classifier will use.
            int count = 0;
            while (count < attributesToDisable) {
                int indexToDisable = rnd.nextInt(data.numAttributes());
                if (indexToDisable != data.classIndex()) {
                    Ensemble[i].disableAttribute(indexToDisable);
                    count++;
                }
            }
            try {
            Ensemble[i].buildClassifier(data);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public double classifyInstance(Instance data) throws Exception {
        double[] dist = distributionForInstance(data);
        int curMaxIndex = 0;
        for (int i = 1; i < dist.length; i++) {
            if (dist[i] > dist[curMaxIndex]) {
                curMaxIndex = i;
            }
        }
        return dist[curMaxIndex];
    }

    @Override
    public double[] distributionForInstance(Instance data) throws Exception {
        double[] distribution = new double[data.numClasses()];
        for (int i = 0; i < EnsembleSize; i++) {
            double[] singleDist = Ensemble[i].distributionForInstance(data);
            for (int j = 0; j < data.numClasses(); j++) {
                distribution[j] += singleDist[j];
            }
        }

        for (int i = 0; i < data.numClasses(); i++) {
            distribution[i] = distribution[i] / EnsembleSize;
        }
        return distribution;
    }

    public int getEnsembleSize() {
        return EnsembleSize;
    }

    public void setEnsembleSize(int size) {
        EnsembleSize = size;
    }

    public double getAttributeSubsetProportion() {
        return AttributeSubsetProportion;
    }

    public void setAttributeSubsetProportion(double proportion) {
        AttributeSubsetProportion = proportion;
    }
}
