package Coursework;

import labs.WekaTools;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Experiment {

    public static void main(String[] args) throws Exception {
        Experiment e = new Experiment();
        e.runExp();

    }

    public void runExp() {
        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/blood/blood_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/blood/blood_TEST.arff");

        ExperimentThread[] experiments = new ExperimentThread[4];

        EnhancedLinearPerceptron c0 = new EnhancedLinearPerceptron();
        c0.setModelSelection(false);
        c0.setStandardiseAttributes(false);
        c0.setUseOnlineAlgorithm(true);
        c0.setMaxNoIterations(100000000);
        experiments[1] = new ExperimentThread(c0, train, test, "LP");

        EnhancedLinearPerceptron c1 = new EnhancedLinearPerceptron();
        c1.setModelSelection(false);
        c1.setStandardiseAttributes(true);
        c1.setUseOnlineAlgorithm(true);
        c1.setMaxNoIterations(100000000);
        experiments[1] = new ExperimentThread(c1, train, test, "LP Standardised");

        EnhancedLinearPerceptron c2 = new EnhancedLinearPerceptron();
        c2.setModelSelection(false);
        c2.setStandardiseAttributes(false);
        c2.setUseOnlineAlgorithm(false);
        c2.setMaxNoIterations(100000000);
        experiments[2] = new ExperimentThread(c1, train, test, "LP Offline");


        EnhancedLinearPerceptron c3 = new EnhancedLinearPerceptron();
        c3.setModelSelection(false);
        c3.setStandardiseAttributes(true);
        c3.setUseOnlineAlgorithm(true);
        c3.setMaxNoIterations(100000000);
        experiments[3] = new ExperimentThread(c1, train, test, "LP Offline Standardised");

        for (int i = 1; i < experiments.length; i++) {
            experiments[i].start();
        }
    }

    public class ExperimentThread extends Thread {

        LinearPerceptron Classifier;
        Instances Train;
        Instances Test;
        String Info;

        public ExperimentThread(LinearPerceptron classifier, Instances train, Instances test, String info) {
            this.Classifier = classifier;
            this.Train = train;
            this.Test = test;
            this.Info = info;
        }

        @Override
        public void run() {
            try {
                Classifier.buildClassifier(Train);
                Evaluation eval = new Evaluation(Train);
                eval.evaluateModel(Classifier, Test);
                System.out.println(Info);
                System.out.println(eval.toSummaryString());
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }

    }
}
