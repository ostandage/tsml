package labs;

import utilities.generic_storage.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.mi.MISVM;
import weka.core.Instances;

public class Lab4 {

    public static void main(String[] args) throws Exception {
        Instances aedes = WekaTools.loadClassificationData("data/labsdata/Aedes_Female_VS_House_Fly_POWER.arff");
        Instances[] aedesSplit = WekaTools.splitData(aedes, 0.7);
        Instances aedesTrain = aedesSplit[0];
        Instances aedesTest = aedesSplit[1];

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.buildClassifier(aedesTrain);
        Evaluation eval = new Evaluation(aedesTrain);
        eval.evaluateModel(mlp, aedesTest);
        System.out.println(eval.errorRate());
        System.out.println(eval.toSummaryString());


    }

    public Pair<Double, Double> findOptimalParameters(SMO classifier, Instances train, Instances test, double[] Gamma, double[] C) throws Exception{
        Evaluation eval = new Evaluation(train);

        for (int i = 0; i < Gamma.length; i++) {

        }
    }

}
