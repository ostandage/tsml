package timing;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.dictionary_based.*;
import timeseriesweka.classifiers.distance_based.*;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.frequency_based.cRISE;
import timeseriesweka.classifiers.hybrids.FlatCote;
import timeseriesweka.classifiers.hybrids.cote.HiveCotePostProcessed;
import timeseriesweka.classifiers.interval_based.LPS;
import timeseriesweka.classifiers.interval_based.TSBF;
import timeseriesweka.classifiers.interval_based.TSF;
import timeseriesweka.classifiers.shapelet_based.FastShapelets;
import timeseriesweka.classifiers.shapelet_based.LearnShapelets;
import timeseriesweka.classifiers.shapelet_based.ShapeletTransformClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

public class NewRunner {

    private static final int NumClassifiers = 25;
    private static String Identifier;
    private static int Resample;
    private static int ClassifierIndex;
    private static String DataPath;
    private static String ResultsPath;

    private static enum DatasetType {
        TRAIN,
        TEST;
    }

    //args
    // 0    Identifier
    // 1    Resample
    // 2    ClassifierIndex
    // 3    DataPath
    // 4    ResultsPath
    public static void main(String[] args) {
        /*Identifier = "id4";
        Resample = 0;
        ClassifierIndex = 11;
        DataPath = "data/Univariate_arff/ItalyPowerDemand";
        ResultsPath = "results";*/

        Identifier = args[0];
        Resample = Integer.parseInt(args[1]) -1;
        ClassifierIndex = Integer.parseInt(args[2]);
        DataPath = args[3];
        ResultsPath = args[4];

        Classifier[] classifiers = createClassifierArray();
        Instances dataTrain = loadData(DataPath, DatasetType.TRAIN);
        Instances dataTest = loadData(DataPath, DatasetType.TEST);


        File outputPath = new File(ResultsPath + "/" + Identifier);
        outputPath.mkdir();

        System.out.println(dataTrain.relationName());
        System.out.println(classifiers[ClassifierIndex].getClass().getSimpleName());

        if (ClassifierIndex == 25) {
            //Hive-Cote
            ArrayList<String> hiveCoteClassifierNames = new ArrayList<>();
            hiveCoteClassifierNames.add(classifiers[7].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[8].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[9].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[10].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[11].getClass().getSimpleName());

            try {
                System.out.println("HIVE-COTE");
                HiveCotePostProcessed hivecote = new HiveCotePostProcessed(ResultsPath + "/" + Identifier + "/", dataTrain.relationName(), Resample, hiveCoteClassifierNames);
                hivecote.loadResults();
                hivecote.distributionForInstance(0);
                hivecote.writeTestSheet(ResultsPath + "/" + Identifier + "/");
            } catch (Exception e) {
                System.out.println("ERROR: Something went wrong in Hivecote.");
                e.printStackTrace();
            }
        }
        else {
            try {
                TimingExperiment t = new TimingExperiment(classifiers[ClassifierIndex], dataTest, dataTrain);


                ResultWrapper rw = t.runExperiment(Resample);
                ClassifierResults cresults = rw.getClassifierResults();
                TimingResults tresults = rw.getTimingResults();
                ClassifierResults trainResults = rw.getTrainResults();


                File dir = new File(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/" );
                dir.mkdirs();

                FileWriter timingCSV = new FileWriter(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/timing" + Resample + ".csv");
                timingCSV.append("Classifier,Dataset,Resample,Average Classify Time,Total Classify Time,Train Time" + "\n");

                String output = classifiers[ClassifierIndex].getClass().getSimpleName() + "," + dataTrain.relationName() + "," + Resample + "," +
                        tresults + "\n";
                System.out.println(output);
                timingCSV.append(output);
                timingCSV.flush();

                cresults.setClassifierName(classifiers[ClassifierIndex].getClass().getSimpleName());
                cresults.setDatasetName(dataTrain.relationName());
                cresults.setSplit("test");
                cresults.writeFullResultsToFile(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/testFold" + Resample + ".csv");

                trainResults.setClassifierName(classifiers[ClassifierIndex].getClass().getSimpleName());
                trainResults.setDatasetName(dataTrain.relationName());
                trainResults.setSplit("train");
                trainResults.writeFullResultsToFile(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/trainFold" + Resample + ".csv");

            } catch (Exception e) {
                System.out.println("Something went wrong :( " + dataTrain.relationName() + " - " + classifiers[ClassifierIndex].getClass().getSimpleName());
                e.printStackTrace();
            }
        }
    }

    public static Instances loadData(String directoryPath, DatasetType type) {
        Instances dataset = null;
        FileReader reader;
        String filePath = null;

        File dir = new File(directoryPath);
        File[] files = dir.listFiles();

        for (File file : files) {
            if (file.getName().endsWith("_" + type.name() + ".arff")) {
                filePath = file.getPath();
            }
        }

        try
        {
            reader = new FileReader(filePath);
            dataset = new Instances(reader);
            dataset.setClassIndex(dataset.numAttributes() - 1);
        } catch (Exception e)
        {
            System.out.println("Exception: " + e);
        }
        return dataset;
    }


    private static Classifier[] createClassifierArray() {
        Classifier[] classifiers = new Classifier[NumClassifiers];
        classifiers[0] = new IBk();
        classifiers[1] = new NaiveBayes();
        classifiers[2] = new DTW1NN();
        classifiers[3] = new FastDTW();
        classifiers[4] = new FastDTW_1NN();
        classifiers[5] = new ProximityForestWrapper();
        classifiers[6] = new DD_DTW();
        //Hive-Cote Classifiers
        classifiers[7] = new ElasticEnsemble();
        classifiers[8] = new ShapeletTransformClassifier();
        classifiers[9] = new RISE();
        classifiers[10] = new BOSS();
        classifiers[11] = new TSF();
        //--
        classifiers[12] = new DTD_C();
        classifiers[13] = new NN_CID();
        classifiers[14] = new SAX_1NN(10, 10);
        classifiers[15] = new cBOSS();
        classifiers[16] = new BagOfPatterns();
        classifiers[17] = new WEASEL();
        classifiers[18] = new SAXVSM();
        classifiers[19] = new cRISE();
        classifiers[20] = new LearnShapelets();
        classifiers[21] = new FastShapelets();
        classifiers[22] = new TSBF();
        classifiers[23] = new LPS();
        classifiers[24] = new FlatCote();

        //Replace this with a list of strings, and then use setclassifier method to get the classifier,
        //within the timing experiment class. ClassifierLists.

        return classifiers;
    }


}