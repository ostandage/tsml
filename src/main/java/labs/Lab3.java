package labs;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

import static labs.WekaTools.*;
import static labs.WekaTools.getClassValues;

public class Lab3 {

    public static void main(String[] args) throws Exception{

        //Load football data
        Instances football = WekaTools.loadClassificationData("data/labsdata/FootballPlayers.arff");
        Instances[] data3 = WekaTools.splitData(football, 0.7);
        Instances footballTrain = data3[0];
        Instances footballTest = data3[1];

        System.out.println("Part 1");
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(footballTrain);
        printConfusionMatrix(confusionMatrix(classifyInstances(nb, footballTest), getClassValues(footballTest)));

        System.out.println("\n\nPart 2");
        MajorityClassClassifier mc = new MajorityClassClassifier();
        mc.buildClassifier(footballTrain);
        printConfusionMatrix(confusionMatrix(classifyInstances(mc, footballTest), getClassValues(footballTest)));

        System.out.println();
        ZeroR zr = new ZeroR();
        zr.buildClassifier(footballTrain);
        printConfusionMatrix(confusionMatrix(classifyInstances(zr, footballTest), getClassValues(footballTest)));

        System.out.println("\n\nPart 5");
        System.out.println("5-A\n");

        Instances aedes = WekaTools.loadClassificationData("data/labsdata/Aedes_Female_VS_House_Fly_POWER.arff");
        Instances[] aedesSplit = WekaTools.splitData(aedes, 0.7);
        Instances aedesTrain = aedesSplit[0];
        Instances aedesTest = aedesSplit[1];

        IB1 ib1 = new IB1();
        ib1.buildClassifier(aedesTrain);
        System.out.println("IB1:    " + accuracy(ib1, aedesTest));
        printConfusionMatrix(confusionMatrix(classifyInstances(ib1, aedesTest), getClassValues(aedesTest)));

        IBk ibk = new IBk();
        ibk.buildClassifier(aedesTrain);
        System.out.println("\nIBK:    " + accuracy(ibk, aedesTest));
        printConfusionMatrix(confusionMatrix(classifyInstances(ibk, aedesTest), getClassValues(aedesTest)));

        Logistic lgc = new Logistic();
        lgc.buildClassifier(aedesTrain);
        System.out.println("\nLGC:    " + accuracy(lgc, aedesTest));
        printConfusionMatrix(confusionMatrix(classifyInstances(lgc, aedesTest), getClassValues(aedesTest)));

        Lab2.OneNN oneNN = new Lab2.OneNN();
        oneNN.buildClassifier(aedesTrain);
        System.out.println("\n1NN:    " + accuracy(oneNN, aedesTest));
        printConfusionMatrix(confusionMatrix(classifyInstances(oneNN, aedesTest), getClassValues(aedesTest)));


        System.out.println("Accuracy Run");
        System.out.println("run,IB1,IBK,Logistic,1NN");
        for (int run = 0; run < 30; run++) {
            aedesSplit = WekaTools.splitData(aedes, 0.7);
            aedesTrain = aedesSplit[0];
            aedesTest = aedesSplit[1];

            ib1.buildClassifier(aedesTrain);
            ibk.buildClassifier(aedesTrain);
            lgc.buildClassifier(aedesTrain);
            oneNN.buildClassifier(aedesTrain);

            System.out.println(run + "," + accuracy(ib1, aedesTest)
                                   + "," + accuracy(ibk, aedesTest)
                                   + "," + accuracy(lgc, aedesTest)
                                   + "," + accuracy(oneNN, aedesTest));
        }
    }

    public static class MajorityClassClassifier extends AbstractClassifier {

        int cvToPredict = -1;

        @Override
        public void buildClassifier(Instances data) throws Exception {
            int[] cvs = WekaTools.getClassValues(data);
            int[] cvCount = new int[data.numClasses()];
            for (int i = 0; i < cvs.length; i++) {
                cvCount[cvs[i]]++;
            }

            int maxIndex = 0;
            for (int i = 0; i < cvCount.length; i++) {
                if (cvCount[i] > cvCount[maxIndex]) {
                    maxIndex = i;
                }
            }
            cvToPredict = cvs[maxIndex];
        }

        public double classifyInstance(Instance data) {
            return cvToPredict;
        }
    }
}
