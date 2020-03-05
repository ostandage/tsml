package Coursework;

import labs.WekaTools;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class EnhancedLinearPerceptron extends LinearPerceptron {

    protected boolean StandardiseAttributes;
    protected boolean UseOnlineAlgorithm;
    protected boolean ModelSelection;
    private double[] Mean;
    private double[] StdDev;
    private int NumCVFoldsForModelSelection;
    private boolean DataAlreadyStandardised;

    public static void main (String[] args) throws Exception{
//        Instances part1Data = WekaTools.loadClassificationData("data/labsdata/part1.arff");
//        part1Data.setClassIndex(part1Data.numAttributes() -1);
//        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
//        elp.setMaxNoIterations(100000000);
//
//        elp.buildClassifier(part1Data);
//        System.out.println("W: " + elp.w[0] + ", " +  elp.w[1]);
//        System.out.println("Done");

        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/blood/blood_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/blood/blood_TEST.arff");

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
        ms.setModelSelection(false);
        ms.setStandardiseAttributes(false);
        ms.setUseOnlineAlgorithm(true);
        ms.setMaxNoIterations(100000000);
        ms.setNumCVFoldsForModelSelection(5);
        ms.buildClassifier(train);
        Evaluation mse = new Evaluation(train);
        mse.evaluateModel(ms, test);
        System.out.println(mse.toSummaryString());
        System.out.println("Error Rate: " + mse.errorRate());
    }

    public EnhancedLinearPerceptron() {
        super();
        StandardiseAttributes = true;
        UseOnlineAlgorithm = true;
        ModelSelection = false;
        NumCVFoldsForModelSelection = 10;
    }

    public EnhancedLinearPerceptron(int numAttributes) {
        super(numAttributes);
        StandardiseAttributes = true;
        UseOnlineAlgorithm = true;
        ModelSelection = false;
        NumCVFoldsForModelSelection = 10;
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        //Copy so as not to standardise original data via reference.
        if (AttributeDisabled == null) {
            AttributeDisabled = new boolean[data.numAttributes()];
        }
        else if (AttributeDisabled.length != data.numAttributes()) {
            //If we use the same classifier on a different dataset.
            AttributeDisabled = new boolean[data.numAttributes()];
        }

        disableAttribute(data.classIndex());


        Instances processedData = new Instances(data);


        if (StandardiseAttributes && !DataAlreadyStandardised) {
            //calculate mean of each attribute.
            Mean = new double[data.numAttributes()];
            for (int a = 0; a < data.numAttributes(); a++) {
                if (!AttributeDisabled[a]) {
                    for (int i = 0; i < data.numInstances(); i++) {
                        Mean[a] = Mean[a] + data.get(i).value(a);
                    }
                    Mean[a] = Mean[a] / (data.numInstances() - NumAttrDisabled);
                }
                else {
                    Mean[a] = 0;
                }
            }

            //calculate std dev of each attribute.
            StdDev = new double[data.numAttributes()];
            for (int a = 0; a < data.numAttributes(); a++) {
                if (!AttributeDisabled[a]) {
                    for (int i = 0; i < data.numInstances(); i++) {
                        StdDev[a] = StdDev[a] + Math.pow((data.get(i).value(a) - Mean[a]), 2);
                    }
                    StdDev[a] = StdDev[a] / (data.numInstances() - NumAttrDisabled);
                }
                else {
                    StdDev[a] = 0;
                }
            }

            for (int i = 0; i < data.numInstances(); i++) {
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
            Evaluation evaluationOnline = new Evaluation(processedData);
            Evaluation evaluationOffline = new Evaluation(processedData);
            EnhancedLinearPerceptron online = new EnhancedLinearPerceptron();
            EnhancedLinearPerceptron offline = new EnhancedLinearPerceptron();
            offline.setUseOnlineAlgorithm(false);
            online.setMaxNoIterations(MaxNoIterations);
            offline.setMaxNoIterations(MaxNoIterations);
            online.NumAttrDisabled = NumAttrDisabled;
            offline.NumAttrDisabled = NumAttrDisabled;
            online.AttributeDisabled = this.AttributeDisabled;
            offline.AttributeDisabled = this.AttributeDisabled;
            if (StandardiseAttributes) {
                online.setStandardiseAttributes(true);
                online.DataAlreadyStandardised = true;
                online.Mean = this.Mean;
                online.StdDev = this.StdDev;

                offline.setStandardiseAttributes(true);
                offline.DataAlreadyStandardised = true;
                offline.Mean = this.Mean;
                offline.StdDev = this.StdDev;
            }
            Random rnd = new Random();

            evaluationOnline.crossValidateModel(online, processedData, NumCVFoldsForModelSelection, rnd);
            double onlineError = evaluationOnline.errorRate();

            evaluationOffline.crossValidateModel(offline, processedData, NumCVFoldsForModelSelection, rnd);
            double offlineError = evaluationOffline.errorRate();

            if (onlineError == offlineError) {
                System.out.println("Same error rate");
            }

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
        if (AttributeDisabled == null) {
            AttributeDisabled = new boolean[data.numAttributes()];
        }
        else if (AttributeDisabled.length != data.numAttributes()) {
            //If we use the same classifier on a different dataset.
            AttributeDisabled = new boolean[data.numAttributes()];
        }

        //Disable the class value as a predictor.
        disableAttribute(data.classIndex());

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
                double t = 1;
                if (data.instance(i).classValue() == 0) {
                    t = -1;
                }

                for (int a = 0; a < data.numAttributes(); a++) {
                    if (!AttributeDisabled[a]) {
                        deltaW[a] = deltaW[a] + (0.5 * LearningRate * (t - y) * data.instance(i).value(a));
                    }
                }
            }

            for (int a = 0; a < data.numAttributes(); a++) {
                if (!AttributeDisabled[a]) {
                    w[a] = w[a] + deltaW[a];
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

    private void standardiseInstance(Instance instance) {
        for (int a = 0; a < instance.numAttributes(); a++) {
            if (!AttributeDisabled[a]) {
                double x = instance.value(a);
                x = (x - Mean[a]) / StdDev[a];
                instance.setValue(a, x);
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (StandardiseAttributes) {
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

    public int getNumCVFoldsForModelSelection() {
        return NumCVFoldsForModelSelection;
    }

    public void setNumCVFoldsForModelSelection(int numCVFoldsForModelSelection) {
        NumCVFoldsForModelSelection = numCVFoldsForModelSelection;
    }
}
