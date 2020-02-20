package Coursework;

import labs.WekaTools;
import org.apache.commons.math3.analysis.function.Max;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class EnhancedLinearPerceptron extends LinearPerceptron {

    private boolean StandardiseAttributes;
    private boolean UseOnlineAlgorithm;
    private boolean ModelSelection;
    private double[] Mean;
    private double[] StdDev;

    public static void main (String[] args) throws Exception{
//        Instances part1Data = WekaTools.loadClassificationData("data/labsdata/part1.arff");
//        part1Data.setClassIndex(part1Data.numAttributes() -1);
//        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
//        elp.setMaxNoIterations(100000000);
//
//        elp.buildClassifier(part1Data);
//        System.out.println("W: " + elp.w[0] + ", " +  elp.w[1]);
//        System.out.println("Done");

        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/planning/planning_TEST.arff");

//        elp.setUseOnlineAlgorithm(false);
//        elp.buildClassifier(train);
//        Evaluation eval = new Evaluation(train);
//        eval.evaluateModel(elp, test);
//        System.out.println("Error Rate Online: " + eval.errorRate());
//
//        elp.setUseOnlineAlgorithm(false);
//        elp.buildClassifier(train);
//        eval.evaluateModel(elp, test);
//        System.out.println("Error Rate Offline: " + eval.errorRate());

        System.out.println("Model Selection");
        EnhancedLinearPerceptron ms = new EnhancedLinearPerceptron();
        ms.setModelSelection(true);
        ms.setStandardiseAttributes(true);
        ms.setMaxNoIterations(1000000);
        ms.buildClassifier(train);
        Evaluation mse = new Evaluation(train);
        mse.evaluateModel(ms, test);
        System.out.println("Error Rate: " + mse.errorRate());
    }

    public EnhancedLinearPerceptron() {
        super();
        StandardiseAttributes = true;
        UseOnlineAlgorithm = true;
        ModelSelection = false;
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {
        //Copy so as not to standardise original data via reference.
        Instances processedData = new Instances(data);


        if (StandardiseAttributes) {
            //calculate mean of each attribute.
            Mean = new double[data.numAttributes()];
            for (int a = 0; a < data.numAttributes(); a++) {
                for (int i = 0; i < data.numInstances(); i++) {
                    Mean[a] = Mean[a] + data.get(i).value(a);
                }
                Mean[a] = Mean[a] / data.numInstances();
            }

            //calculate std dev of each attribute.
            StdDev = new double[data.numAttributes()];
            for (int a = 0; a < data.numAttributes(); a++) {
                for (int i = 0; i < data.numInstances(); i++) {
                    StdDev[a] = StdDev[a] + Math.pow((data.get(i).value(a) - Mean[a]), 2);
                }
                StdDev[a] = StdDev[a] / data.numInstances();
            }

            for (int i = 0; i < data.numInstances(); i++) {
//                for (int a = 0; a < data.numAttributes()-1; a++) {
//                    double x = processedData.get(i).value(a);
//                    x = (x-Mean[a]) / StdDev[a];
//                    processedData.get(i).setValue(a, x);
//                }
                standardiseInstance(processedData.get(i));
            }
        }

        if (!ModelSelection) {
            if (UseOnlineAlgorithm) {
                super.buildClassifier(processedData);
            } else {
                buildOfflineClassifier(processedData);
            }
        }
        else {
            Evaluation evaluation = new Evaluation(processedData);
            EnhancedLinearPerceptron online = new EnhancedLinearPerceptron();
            EnhancedLinearPerceptron offline = new EnhancedLinearPerceptron();
            offline.setUseOnlineAlgorithm(false);
            online.setMaxNoIterations(MaxNoIterations);
            offline.setMaxNoIterations(MaxNoIterations);
            Random rnd = new Random();

            evaluation.crossValidateModel(online, processedData, 10, rnd);
            double onlineError = evaluation.errorRate();

            evaluation.crossValidateModel(offline, processedData, 10, rnd);
            double offlineError = evaluation.errorRate();

            if (onlineError < offlineError) {
                super.buildClassifier(processedData);
                System.out.println("Use online");
            }
            else {
                buildOfflineClassifier(processedData);
                System.out.println("Use offline");
            }
        }

    }

    private void buildOfflineClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        w = new double[data.numAttributes()];
        Random rnd = new Random();
        for (int i = 0; i < w.length; i++) {
            //Include -ve random?
            w[i] = rnd.nextInt();
        }

        int iteration = 0;
        do {
            iteration++;

            double[] deltaW = new double[data.numAttributes()];

            for (int i = 0; i < data.numInstances(); i++) {
                double y = calculateYi(data.instance(i)) ;
                for (int a = 0; a < data.numAttributes(); a++) {
                    deltaW[a] = deltaW[a] + (0.5 * LearingRate * (data.instance(i).classValue() - y) * data.instance(i).value(a));
                }
            }

            for (int a = 0; a < data.numAttributes(); a++) {
                w[a] = w[a] + deltaW[a];
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



    private void standardiseInstance(Instance instance) {
        for (int a = 0; a < instance.numAttributes()-1; a++) {
            double x = instance.value(a);
            x = (x-Mean[a]) / StdDev[a];
            instance.setValue(a, x);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (StandardiseAttributes) {
            //Copy instance first??
//            for (int a = 0; a < instance.numAttributes() -1; a++) {
//                double x = instance.value(a);
//                x = (x-Mean[a]) / StdDev[a];
//                instance.setValue(a, x);
//            }
            standardiseInstance(instance);
        }

        return super.classifyInstance(instance);
    }

    public void setStandardiseAttributes(boolean standardiseAttributes) {
        StandardiseAttributes = standardiseAttributes;
    }

    public void setUseOnlineAlgorithm(boolean useOnlineAlgorithm) {
        UseOnlineAlgorithm = useOnlineAlgorithm;
    }

    public void setModelSelection(boolean modelSelection) {
        ModelSelection = modelSelection;
    }
}
