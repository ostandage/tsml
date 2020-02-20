//Need to confirm:
//      - Start with random vector.
//      - Bias term = learning rate.

package Coursework;

import labs.WekaTools;
import org.apache.commons.math3.analysis.function.Max;
import scala.tools.reflect.Eval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class LinearPerceptron extends AbstractClassifier {

    protected int MaxNoIterations;
    protected double Bias;
    protected double LearingRate;

    //change back to private.
    public double[] w;

    public static void main (String[] args) throws Exception{
        Instances part1Data = WekaTools.loadClassificationData("data/labsdata/part1.arff");
        part1Data.setClassIndex(part1Data.numAttributes() -1);
        LinearPerceptron lp = new LinearPerceptron();
        lp.setMaxNoIterations(100000000);
        lp.buildClassifier(part1Data);
        System.out.println("W: " + lp.w[0] + ", " +  lp.w[1]);
        System.out.println("Done");

        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TEST.arff");


        lp.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(lp, test);
        System.out.println("Error Rate: " + eval.errorRate());

    }


    public LinearPerceptron () {
        MaxNoIterations = Integer.MAX_VALUE;
        Bias = 0;
        LearingRate = 1;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();
        capabilities.enable(Capabilities.Capability.NUMERIC_CLASS);
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.BINARY_CLASS);
        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {

        getCapabilities().testWithFail(data);
        //Doesn't seem to work well random initial vector.
        w = new double[data.numAttributes()];
        Random rnd = new Random();
        for (int i = 0; i < w.length; i++) {
            //Include -ve random?
            w[i] = rnd.nextInt();
        }

//        w[0] = 1;
//        w[1] = 1;

        int iteration = 0;
        do {
            iteration++;
            for (int i = 0; i < data.numInstances(); i++) {
                double y = calculateYi(data.instance(i));

                for (int j = 0; j < data.numAttributes(); j++) {
                    w[j] = w[j] + (0.5 * LearingRate * (data.instance(i).classValue() - y) * data.instance(i).value(j));
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
        return calculateYi(instance);
    }

    protected double calculateYi(Instance data) {
        double yU = 0;
        for (int c = 0; c < data.numAttributes(); c++) {
            yU = yU + (data.value(c) * w[c]);
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

    public void setLearingRate(double learingRate) {
        this.LearingRate = learingRate;
    }
}
