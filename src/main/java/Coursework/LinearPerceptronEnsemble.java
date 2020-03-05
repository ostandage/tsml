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
    private boolean[][] AttributesToDisable;

    public static void main (String[] args) throws Exception {
        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TEST.arff");

        lpe.setMaxNoIterations(100000000);
        lpe.setAttributeSubsetProportion(0.5);
        lpe.setEnsembleSize(50);
        lpe.setStandardiseAttributes(true);
        lpe.setModelSelection(true);
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
        getCapabilities().testWithFail(data);

        if (AttributeDisabled == null) {
            AttributeDisabled = new boolean[data.numAttributes()];
        }
        else if (AttributeDisabled.length != data.numAttributes()) {
            //If we use the same classifier on a different dataset.
            AttributeDisabled = new boolean[data.numAttributes()];
        }

        Random rnd = new Random();
        int numAttributesToDisable = data.numAttributes() - (int) (data.numAttributes() * AttributeSubsetProportion);
        AttributesToDisable = new boolean[EnsembleSize][data.numAttributes()];


        for (int i = 0; i < EnsembleSize; i++) {

            //Work out which attributes this classifier will (won't) use.
            int count = 0;
            while (count < numAttributesToDisable) {
                int indexToDisable = rnd.nextInt(data.numAttributes());
                if (indexToDisable != data.classIndex()) {
                    AttributesToDisable[i][indexToDisable] = true;
                    count++;
                }
            }
            AttributesToDisable[i][data.classIndex()] = true;
        }

        super.buildClassifier(data);
    }

    @Override
    public double classifyInstance(Instance data) throws Exception {
        int[] classPredictions = new int[data.numClasses()];
        for (int e = 0; e < EnsembleSize; e++) {
            for(int attr = 0; attr < AttributesToDisable[e].length; attr++) {
                AttributeDisabled[attr] = AttributesToDisable[e][attr];
            }

            double prediction = super.classifyInstance(data);
            classPredictions[(int) prediction]++;
        }

        int curMax = classPredictions[0];
        for (int c = 1; c < classPredictions.length; c++) {
            if (classPredictions[c] > classPredictions[curMax]) {
                curMax = c;
            }
        }

        return curMax;
        //        double[] dist = distributionForInstance(data);
//        int curMaxIndex = 0;
//        for (int i = 1; i < dist.length; i++) {
//            if (dist[i] > dist[curMaxIndex]) {
//                curMaxIndex = i;
//            }
//        }
//        return dist[curMaxIndex];
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[instance.numClasses()];


        for (int e = 0; e < EnsembleSize; e++) {

            for(int attr = 0; attr < AttributesToDisable[e].length; attr++) {
                AttributeDisabled[attr] = AttributesToDisable[e][attr];
            }

            double prediction = super.classifyInstance(instance);
            distribution[(int) prediction]++;

        }

        for (int i = 0; i < instance.numClasses(); i++) {
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
