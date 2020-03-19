package Coursework;

import experiments.data.DatasetLists;
import labs.WekaTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;

public class ExperimentRunner {

    static Classifier[] Classifiers;
    static Instances[] Datasets;

    public static void main(String[] args) throws Exception {
        loadDatasets();

        Classifiers = new Classifier[3];
        Classifiers[0] = new LinearPerceptron();

        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
        elp.setModelSelection(true);
        Classifiers[1] = elp;

        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        lpe.setModelSelection(true);
        Classifiers[2] = lpe;

        String[] ClassifierNames = {"Linear Perceptron", "Enhanced Linear Perceptron", "Linear Perceptron Ensemble"};

        Instances[] trains = new Instances[Datasets.length];
        Instances[] tests = new Instances[Datasets.length];

        for (int d = 0; d < Datasets.length; d++) {
            Instances[] split = WekaTools.splitData(Datasets[d], 0.6);
            trains[d] = split[0];
            tests[d] = split[1];
        }


        for (int d = 0; d < Datasets.length; d++) {
            for (int c = 0; c < Classifiers.length; c++) {
                try {
                    Classifiers[c].buildClassifier(trains[d]);
                    String info = DatasetLists.ReducedUCI[d] + " - " + ClassifierNames[c];
                    Evaluation eval = new Evaluation(trains[d]);
                    eval.evaluateModel(Classifiers[c], tests[d]);
                    System.out.println(eval.toSummaryString(info, false));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }





    }

    public static void loadDatasets() {
        File dataDir = new File("data/UCIContinuous");
        String[] datasets = DatasetLists.ReducedUCI;
        Datasets = new Instances[datasets.length];

        for (int d = 0; d < datasets.length; d++) {
            Datasets[d] = WekaTools.loadClassificationData(dataDir + "/" + datasets[d] + "/" + datasets[d] + ".arff");
        }
    }

}
