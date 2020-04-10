package Coursework;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLists;
import labs.WekaTools;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.DD_DTW;
import timeseriesweka.classifiers.interval_based.TSF;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

public class ExperimentRunner {

    static Classifier[] Classifiers;
    static Instances[] Datasets;

    public static void main(String[] args) {
        loadDatasets();

        //Create classifiers
        Classifiers = new Classifier[11];
        Classifiers[0] = new LinearPerceptron();

        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
        elp.setModelSelection(true);
        Classifiers[1] = elp;

        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        lpe.setModelSelection(true);
        Classifiers[2] = lpe;

        RotationForest rotf = new RotationForest();
        RandomForest randf = new RandomForest();
        IBk ibk = new IBk();
        NaiveBayes nb = new NaiveBayes();
        DD_DTW dd_dtw = new DD_DTW();
        BOSS boss = new BOSS();
        TSF tsf = new TSF();
        J48 j48 = new J48();
        Classifiers[3] = rotf;
        Classifiers[4] = randf;
        Classifiers[5] = ibk;
        Classifiers[6] = nb;
        Classifiers[7] = dd_dtw;
        Classifiers[8] = boss;
        Classifiers[9] = tsf;
        Classifiers[10] = j48;
        String[] ClassifierNames = {"LinearPerceptron", "EnhancedLinearPerceptron", "LinearPerceptronEnsemble",
                                    "RotF", "RandF", "IBk", "DTW_1NN", "NB", "DD_DTW", "BOSS", "TSF", "HiveCote", "ElasticEnsemble", "C4.5"};

        //Create train/test data, convert non-binary class problems.
        Instances[] trains = new Instances[Datasets.length];
        Instances[] tests = new Instances[Datasets.length];
        for (int d = 0; d < Datasets.length; d++) {
            if (Datasets[d].numClasses() > 2) {
                Datasets[d] = convertToBinaryClassProblem(Datasets[d]);
            }
            Instances[] split = WekaTools.splitData(Datasets[d], 0.6);
            trains[d] = split[0];
            tests[d] = split[1];
        }

        for (int d = 0; d < Datasets.length; d++) {
            for (int c = 0; c < Classifiers.length; c++) {
                for (int r = 0; r < 10; r++) {
                    try {
                        //Run experiment
                        TimingExperiment t = new TimingExperiment(Classifiers[c], tests[d], trains[d]);
                        ResultWrapper rw = t.runExperiment(r);
                        TimingResults tresults = rw.getTimingResults();
                        ClassifierResults cresults = rw.getClassifierResults();

                        //Create files
                        File dir = new File("results/perceptrons/" + ClassifierNames[c] + "/Predictions/" + DatasetLists.ReducedUCI[d] + "/" );
                        dir.mkdirs();
                        FileWriter timingCSV = new FileWriter("results/perceptrons/" + ClassifierNames[c] + "/Predictions/" + DatasetLists.ReducedUCI[d] + "/timing" + r + ".csv");
                        timingCSV.append("Classifier,Dataset,Resample,Average Classify Time,Total Classify Time,Train Time" + "\n");

                        //Save timing data
                        String output = ClassifierNames[c] + "," + DatasetLists.ReducedUCI[d] + "," + r + "," + tresults + "\n";
                        System.out.println(output);
                        timingCSV.append(output);
                        timingCSV.flush();
                        timingCSV.close();

                        //Save evaluation data
                        cresults.setClassifierName(ClassifierNames[c]);
                        cresults.setDatasetName(DatasetLists.ReducedUCI[d]);
                        cresults.setSplit("test");
                        cresults.writeFullResultsToFile("results/perceptrons/" + ClassifierNames[c] + "/Predictions/" + DatasetLists.ReducedUCI[d] + "/testFold" + r + ".csv");

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
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

    //Only for numeric problems.
    public static Instances convertToBinaryClassProblem(Instances problem) {
        //Find majority class
        int[] classDistribution = new int[problem.numInstances()];
        int majorityClass = 0;

        for (Instance instance : problem) {
            classDistribution[(int) instance.classValue()]++;
        }

        for (int c = 1; c < classDistribution.length; c++) {
            if (classDistribution[c] > classDistribution[majorityClass]) {
                majorityClass = c;
            }
        }

        //Convert actual class values.
        Instances binaryProblem = new Instances(problem);
        for (int i = 0; i < binaryProblem.numInstances(); i++) {
            if (binaryProblem.get(i).classValue() == majorityClass) {
                binaryProblem.get(i).setValue(binaryProblem.classIndex(), 0);
            }
            else {
                binaryProblem.get(i).setValue(binaryProblem.classIndex(), 1);
            }
        }

        //Convert class values object to binary - required as only generated when we create a new Instances object.
        ArrayList<Object> val2 = new ArrayList<>();
        val2.add("0");
        val2.add("1");
        binaryProblem.classAttribute().forceUpdateValues(val2);

        return binaryProblem;
    }
}
