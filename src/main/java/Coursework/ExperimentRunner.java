package Coursework;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLists;
import jdk.nashorn.internal.runtime.Timing;
import labs.WekaTools;
import multivariate_timeseriesweka.measures.EuclideanDistance_D;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.DD_DTW;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import timeseriesweka.classifiers.distance_based.FastDTW_1NN;
import timeseriesweka.classifiers.hybrids.HiveCote;
import timeseriesweka.classifiers.interval_based.TSF;
import timing.ResultWrapper;
import timing.TimingExperiment;
import timing.TimingResults;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

public class ExperimentRunner {

    static Classifier[] Classifiers;
    static Instances[] Datasets;

    public static void main(String[] args) throws Exception {
        loadDatasets();


        Classifiers = new Classifier[13];
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
        FastDTW_1NN dtw_1NN = new FastDTW_1NN();
        NaiveBayes nb = new NaiveBayes();
        DD_DTW dd_dtw = new DD_DTW();
        BOSS boss = new BOSS();
        TSF tsf = new TSF();
        HiveCote hc = new HiveCote();
        ElasticEnsemble ee = new ElasticEnsemble();

        Classifiers[3] = rotf;
        Classifiers[4] = randf;
        Classifiers[5] = ibk;
        Classifiers[6] = dtw_1NN;
        Classifiers[7] = nb;
        Classifiers[8] = dd_dtw;
        Classifiers[9] = boss;
        Classifiers[10] = tsf;
        Classifiers[11] = hc;
        Classifiers[12] = ee;
        String[] ClassifierNames = {"LinearPerceptron", "EnhancedLinearPerceptron", "LinearPerceptronEnsemble",
                                    "RotF", "RandF", "IBk", "DTW_1NN", "NB", "DD_DTW", "BOSS", "TSF", "HiveCote", "ElasticEnsemble"};


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


        for (int d = 4; d < Datasets.length; d++) {
            //for (int c = 0; c < Classifiers.length; c++) {
            for (int c = 0; c < 3; c++) {
                for (int r = 0; r < 1; r++) {
                    try {
//                    Classifiers[c].buildClassifier(trains[d]);
//                    String info = DatasetLists.ReducedUCI[d] + " - " + ClassifierNames[c];
//                    Evaluation eval = new Evaluation(trains[d]);
//                    eval.evaluateModel(Classifiers[c], tests[d]);
//                    System.out.println(eval.toSummaryString(info, false));

                        TimingExperiment t = new TimingExperiment(Classifiers[c], tests[d], trains[d]);
                        ResultWrapper rw = t.runExperiment(r);
                        TimingResults tresults = rw.getTimingResults();
                        ClassifierResults cresults = rw.getClassifierResults();

                        File dir = new File("results/perceptrons/" + ClassifierNames[c] + "/Predictions/" + DatasetLists.ReducedUCI[d] + "/" );
                        dir.mkdirs();

                        FileWriter timingCSV = new FileWriter("results/perceptrons/" + ClassifierNames[c] + "/Predictions/" + DatasetLists.ReducedUCI[d] + "/timing" + r + ".csv");
                        timingCSV.append("Classifier,Dataset,Resample,Average Classify Time,Total Classify Time,Train Time" + "\n");

                        String output = ClassifierNames[c] + "," + DatasetLists.ReducedUCI[d] + "," + r + "," +
                                tresults + "\n";
                        System.out.println(output);
                        timingCSV.append(output);
                        timingCSV.flush();
                        timingCSV.close();

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

        Instances binaryProblem = new Instances(problem);
        for (int i = 0; i < binaryProblem.numInstances(); i++) {
            if (binaryProblem.get(i).classValue() == majorityClass) {
                binaryProblem.get(i).setValue(binaryProblem.classIndex(), 0);
            }
            else {
                binaryProblem.get(i).setValue(binaryProblem.classIndex(), 1);
            }
        }

        ArrayList<Object> val2 = new ArrayList<>();
        val2.add("0");
        val2.add("1");

        binaryProblem.classAttribute().forceUpdateValues(val2);

        return binaryProblem;
    }

}
