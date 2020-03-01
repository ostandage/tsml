package Coursework;

import experiments.data.DatasetLists;
import labs.WekaTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;

public class ExperimentRunner {

    Classifier[] Classifiers;
    Instances[] Datasets;

    public static void main(String[] args) {

    }

    public void setupClassifiers() {


        //Linear Perceptrons


    }

    public void loadDatasets() {
        File dataDir = new File("data/UCIContinuous");
        String[] datasets = DatasetLists.ReducedUCI;
        Datasets = new Instances[datasets.length];

        for (int d = 0; d < datasets.length; d++) {
            Datasets[d] = WekaTools.loadClassificationData(dataDir + "/" + datasets[d] + "/" + datasets[d] + ".arff");
        }
    }

}
