//Need to confirm:
//      - Bias term = y offset - allows for shifting. Could be useful for -1 and 1 class vals.


//Classify instance should return one of the class values. Change part 1 data to be 0 and 1 and check

package Coursework;

import labs.WekaTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class LinearPerceptron extends AbstractClassifier {

    protected int MaxNoIterations;
    protected double Bias;
    protected double LearningRate;
    protected boolean[] AttributeDisabled;
    protected int NumAttrDisabled;


    //change back to private.
    protected double[] w;

    public static void main (String[] args) throws Exception{
        LinearPerceptron lp = new LinearPerceptron();


//        Instances part1Data = WekaTools.loadClassificationData("data/labsdata/part1.arff");
//        lp.setMaxNoIterations(100000000);
//        lp.buildClassifier(part1Data);
//        System.out.println("W: " + lp.w[0] + ", " +  lp.w[1]);
//        System.out.println("Done");

        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/blood/blood_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/blood/blood_TEST.arff");

        //MaxValue =         2147483647
        lp.setMaxNoIterations(100000000);
        lp.buildClassifier(train);
        lp.setLearningRate(1);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(lp, test);
        System.out.println(eval.toSummaryString());
        System.out.println("Error Rate: " + eval.errorRate());
        //100000000 gives accuracy of 64.4385%

        //100000 gives accuracy of 77.0053%
//        MultilayerPerceptron mlp = new MultilayerPerceptron();
//        mlp.setLearningRate(1);
//        mlp.setHiddenLayers("0");
//        mlp.setGUI(false);
//        mlp.setTrainingTime(100000);
//        mlp.buildClassifier(train);
//        Evaluation eval2 = new Evaluation(train);
//        eval2.evaluateModel(mlp, test);
//        System.out.println(eval2.toSummaryString());
//        System.out.println("Error Rate: " + eval2.errorRate());

    }


    public LinearPerceptron () {
        MaxNoIterations = Integer.MAX_VALUE;
        Bias = 0;
        LearningRate = 1;
    }

    public LinearPerceptron (int numAttributes) {
        MaxNoIterations = Integer.MAX_VALUE;
        Bias = 0;
        LearningRate = 1;
        AttributeDisabled = new boolean[numAttributes];
    }


    public boolean disableAttribute(int attrIndex) {
        NumAttrDisabled++;
        return AttributeDisabled[attrIndex] = true;
    }

    public boolean enableAttributes(int attrIndex) {
        NumAttrDisabled--;
        return AttributeDisabled[attrIndex] = false;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.BINARY_CLASS);
        capabilities.enable(Capabilities.Capability.BINARY_ATTRIBUTES);
        capabilities.setMinimumNumberInstances(1);
        return capabilities;
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

        //Disable the class value as a predictor.
        disableAttribute(data.classIndex());


        //Doesn't seem to work well random initial vector.
        w = new double[data.numAttributes()];
        Random rnd = new Random();
        for (int i = 0; i < w.length; i++) {
            //Include -ve random?
            w[i] = rnd.nextInt();
            //w[i] = 1;
        }


        int iteration = 0;
        do {
            iteration++;
            for (int i = 0; i < data.numInstances(); i++) {
                double y = calculateYi(data.instance(i));
                double t = 1;
                if (data.instance(i).classValue() == 0) {
                    t = -1;
                }

                for (int j = 0; j < data.numAttributes(); j++) {
                    if (!AttributeDisabled[j]) {
                        //                                  cv - pred
                        double deltaW = (0.5 * LearningRate * (t - y) * data.instance(i).value(j));
                        w[j] = w[j] + deltaW;
                    }
                }

            }

            boolean madeFullPass = false;
            for (int i = 0; i < data.numInstances(); i++) {

                double y = calculateYi(data.instance(i));

                if (y != data.instance(i).classValue()) {
                    madeFullPass = false;
                    break;
                }
                madeFullPass = true;
            }
            if (madeFullPass) {
                break;
            }

        } while (iteration < MaxNoIterations);

        if (iteration == MaxNoIterations) {
            System.out.println("Maximum number of iterations for training reached.");
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double y =  calculateYi(instance);
        if (y == 1.0) {
            return 1;
        }
        else {
            return 0;
        }

    }

    protected double calculateYi(Instance data) {
        double yU = 0;
        for (int c = 0; c < data.numAttributes(); c++) {
            if (!AttributeDisabled[c]) {
                yU = yU + (data.value(c) * w[c]);
            }
        }
        double y = 1;
        if (yU < 0) {
            y = -1;
        }
        return y;
    }

    public void setMaxNoIterations (int maxNoIterations) {
        this.MaxNoIterations = maxNoIterations;
    }

    public void setConstantTerm(Double bias) {
        this.Bias = bias;
    }

    public void setLearningRate(double learningRate) {
        this.LearningRate = learningRate;
    }
}
