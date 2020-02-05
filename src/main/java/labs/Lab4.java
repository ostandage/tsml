package labs;

import utilities.generic_storage.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.mi.MISVM;
import weka.core.Instances;

import java.util.Random;

public class Lab4 {

    public static void main(String[] args) throws Exception {
        Instances aedes = WekaTools.loadClassificationData("data/labsdata/Aedes_Female_VS_House_Fly_POWER.arff");
        Instances[] aedesSplit = WekaTools.splitData(aedes, 0.7);
        Instances aedesTrain = aedesSplit[0];
        Instances aedesTest = aedesSplit[1];

//        MultilayerPerceptron mlp = new MultilayerPerceptron();
//        mlp.buildClassifier(aedesTrain);
//        Evaluation eval = new Evaluation(aedesTrain);
//        eval.evaluateModel(mlp, aedesTest);
//        System.out.println(eval.errorRate());
//        System.out.println(eval.toSummaryString());

//        double[] gamma = {0.001,0.01,0.1,1,10};
        double [] gamma = new double[30];
        int count = 0;
        for (int exp = -15; exp < 15; exp++) {
            gamma[count++] = Math.pow(2, exp);
        }

        double[] C = {0.001,0.01,0.1,1,10};

        double[] lr = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999};


        findOptimalParameters(aedesTrain, aedesTest, lr, lr);
    }

    public static void findOptimalParameters(Instances train, Instances test, double[] Gamma, double[] C) throws Exception{
        Evaluation eval = new Evaluation(train);
//        PolyKernel kernel = new PolyKernel();
//        RBFKernel kernel = new RBFKernel();
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        Random r = new Random();
        mlp.setSeed(r.nextInt());

        for (int i = 0; i < Gamma.length; i++) {

//            kernel.setGamma(Gamma[i]);
//            kernel.setExponent(Gamma[i]);

            mlp.setLearningRate(Gamma[i]);
            //SMO svm = new SMO();
            //svm.setKernel(kernel);
            for (int j = 0; j < C.length; j++) {
                //svm.setC(C[j]);
                //svm.buildClassifier(train);
                mlp.setMomentum(C[j]);
                mlp.buildClassifier(train);
                eval.evaluateModel(mlp, test);
                System.out.println("G: " + Gamma[i] + "    C: " + C[j] + "    Error Rate: " + eval.errorRate());
            }
        }
    }

}
