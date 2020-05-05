/**
 * This class runs all of the timing experiments, performing the basis for benchmarking the classifiers. This class is
 * designed to be run on the HPC, hence it uses command line arguments. This class represents a single classifier
 * dataset fold.
 */
package timing;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.dictionary_based.*;
import timeseriesweka.classifiers.distance_based.*;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.frequency_based.cRISE;
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
import weka.classifiers.meta.RotationForest;
import weka.core.Instances;
import java.io.*;
import java.util.ArrayList;

public class TimingRunner {

    private static final int NumClassifiers = 25;
    private static String Identifier;
    private static int Resample;
    private static int ClassifierIndex;
    private static String DataPath;
    private static String ResultsPath;
    private static int NumThreads;

    public static enum DatasetType {
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
        //Debug
        if (args.length == 0) {
            Identifier = "dtwKNN";
            Resample = 0;
            ClassifierIndex = 1;
            DataPath = "data/Univariate_arff/Beef";
            ResultsPath = "results";
            NumThreads = 2;
        }
        else {
            Identifier = args[0];
            Resample = Integer.parseInt(args[1]) - 1;
            ClassifierIndex = Integer.parseInt(args[2]);
            DataPath = args[3];
            ResultsPath = args[4];

            //Optional parameter.
            if (args.length == 6) {
                NumThreads = Integer.parseInt(args[5]);
            }
        }

        Classifier[] classifiers = createClassifierArray();
        Instances dataTrain = WekaTools.loadData(DataPath, DatasetType.TRAIN);
        Instances dataTest = WekaTools.loadData(DataPath, DatasetType.TEST);

        File outputPath = new File(ResultsPath + "/" + Identifier);
        outputPath.mkdir();

        if (ClassifierIndex == 25) {
            //Hive-Cote is post processed as all of its constituents are already computed.
            ArrayList<String> hiveCoteClassifierNames = new ArrayList<>();
            hiveCoteClassifierNames.add(classifiers[7].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[8].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[9].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[10].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[11].getClass().getSimpleName());

            //Postprocess the classifier results files.
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

            String[] dataPathSplit = DataPath.split("/");
            String dataset = dataPathSplit[dataPathSplit.length-1];
            String[] hcClassifierNames = {"ElasticEnsemble", "ShapeletTransformClassifier", "RISE", "BOSS", "TSF"};

            //Combine timing data.
            double avgClassifyTime = 0;
            double totalClassifyTime = 0;
            double trainTime = 0;
            try {
                for (String classifier : hcClassifierNames) {
                    File timingFile = new File(ResultsPath + "/" + Identifier + "/" + classifier + "/Predictions/" + dataset + "/timing" + Resample + ".csv");
                    BufferedReader csvRead = new BufferedReader(new FileReader(timingFile));
                    csvRead.readLine();
                    String[] dataLine = csvRead.readLine().split(",");
                    avgClassifyTime = avgClassifyTime + Double.parseDouble(dataLine[3]);
                    totalClassifyTime = totalClassifyTime + Double.parseDouble(dataLine[4]);
                    trainTime = trainTime + Double.parseDouble(dataLine[5]);

                }

                FileWriter timingCSV = new FileWriter(ResultsPath + "/" + Identifier + "/HIVE-COTE/Predictions/" + dataset + "/timing" + Resample + ".csv");
                timingCSV.append("Classifier,Dataset,Resample,Average Classify Time,Total Classify Time,Train Time" + "\n");
                timingCSV.append("HIVE-COTE," + dataset + "," + Resample + "," + avgClassifyTime + "," + totalClassifyTime + "," + trainTime + "\n");
                timingCSV.flush();
                timingCSV.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        else {
            //All other classifiers.
            System.out.println(dataTrain.relationName());
            System.out.println(classifiers[ClassifierIndex].getClass().getSimpleName());
            try {
                //Benchmark classifier
                TimingExperiment t = new TimingExperiment(classifiers[ClassifierIndex], dataTest, dataTrain);
                ResultWrapper rw = t.runExperiment(Resample);
                ClassifierResults cresults = rw.getClassifierResults();
                TimingResults tresults = rw.getTimingResults();
                ClassifierResults trainResults = rw.getTrainResults();

                //Write results
                File dir = new File(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/" );
                dir.mkdirs();

                //Timing data
                FileWriter timingCSV = new FileWriter(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/timing" + Resample + ".csv");
                timingCSV.append("Classifier,Dataset,Resample,Average Classify Time,Total Classify Time,Train Time" + "\n");
                String output = classifiers[ClassifierIndex].getClass().getSimpleName() + "," + dataTrain.relationName() + "," + Resample + "," +
                        tresults + "\n";
                System.out.println(output);
                timingCSV.append(output);
                timingCSV.flush();
                timingCSV.close();

                //Test fold data
                cresults.setClassifierName(classifiers[ClassifierIndex].getClass().getSimpleName());
                cresults.setDatasetName(dataTrain.relationName());
                cresults.setSplit("test");
                cresults.writeFullResultsToFile(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/testFold" + Resample + ".csv");

                //Train fold data
                trainResults.setClassifierName(classifiers[ClassifierIndex].getClass().getSimpleName());
                trainResults.setDatasetName(dataTrain.relationName());
                trainResults.setSplit("train");
                trainResults.writeFullResultsToFile(ResultsPath + "/" + Identifier + "/" + classifiers[ClassifierIndex].getClass().getSimpleName() + "/Predictions/" + dataTrain.relationName() + "/trainFold" + Resample + ".csv");

            } catch (Exception e) {
                System.out.println("Something went wrong " + dataTrain.relationName() + " - " + classifiers[ClassifierIndex].getClass().getSimpleName());
                e.printStackTrace();
            }
        }
    }


    public static Classifier[] createClassifierArray() {
        Classifier[] classifiers = new Classifier[NumClassifiers];
        classifiers[0] = new IBk();
        classifiers[1] = new NaiveBayes();
        classifiers[2] = new DTW1NN();
        classifiers[3] = new FastDTW();

        FastDTW_1NN dtw_1NN = new FastDTW_1NN();
        dtw_1NN.setMaxNoThreads(NumThreads);
        classifiers[4] = dtw_1NN;

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
        classifiers[24] = new RotationForest();
        return classifiers;
    }
}
