package labs;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
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
