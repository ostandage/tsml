//Critical Difference diagrams

package timing;

import evaluation.storage.ClassifierResults;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import timeseriesweka.classifiers.dictionary_based.*;
import timeseriesweka.classifiers.distance_based.*;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.frequency_based.cRISE;
import timeseriesweka.classifiers.hybrids.FlatCote;
import timeseriesweka.classifiers.hybrids.HiveCote;
import timeseriesweka.classifiers.hybrids.cote.HiveCotePostProcessed;
import timeseriesweka.classifiers.interval_based.LPS;
import timeseriesweka.classifiers.interval_based.TSBF;
import timeseriesweka.classifiers.interval_based.TSF;
import timeseriesweka.classifiers.shapelet_based.FastShapelets;
import timeseriesweka.classifiers.shapelet_based.LearnShapelets;
import timeseriesweka.classifiers.shapelet_based.ShapeletTransformClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka_extras.classifiers.ensembles.CAWPE;
import weka_extras.classifiers.ensembles.HIVE_COTE;

/**
 *
 * @author ostandage
 */
public class ExperimentRunner {

    private static final int NumClassifiers = 25;
    private static int NumResamples = 5;
    private static int NumDatasets;
    private static final int NumSwaps = 10000;
    private static String DataPath = "data/Univariate_arff";
    private static String ResultsPath = "results";



    // args:
    // 0    numResamples
    // 1    data
    // 2    results
    public static void main(String[] args) {

            if (args.length > 2) {
                ResultsPath = args[2];
            }
            if (args.length > 1) {
                DataPath = args[1];
            }
            if (args.length > 0) {
                NumResamples = Integer.parseInt(args[0]);
            }

            Instances[] dataTrain = loadData("arff", "TRAIN", DataPath);
            Instances[] dataTest = loadData("arff", "TEST", DataPath);
            Classifier[] classifiers = new Classifier[NumClassifiers];
            ClassifierResults[][][] cresults = new ClassifierResults[dataTest.length][NumClassifiers][];
            TimingResults[][][] tresults = new TimingResults[dataTest.length][NumClassifiers][];
            NumDatasets = dataTest.length;

            classifiers[0] = new IBk();
            classifiers[1] = new DTW_kNN(1);
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

            ArrayList<String> hiveCoteClassifierNames = new ArrayList<>();
            hiveCoteClassifierNames.add(classifiers[7].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[8].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[9].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[10].getClass().getSimpleName());
            hiveCoteClassifierNames.add(classifiers[11].getClass().getSimpleName());


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

            Random resampleSeedGenerator = new Random();


            String timeStamp = java.time.ZonedDateTime.now().toLocalDateTime().toString();
            timeStamp = timeStamp.replace(':', '-');
            System.out.println(timeStamp + "\n");
            File path = new File(ResultsPath +"/"+ timeStamp);
            path.mkdirs();

            FileWriter csv = null;
            try {
                csv = new FileWriter(ResultsPath + "/" + timeStamp + "/" + "timing.csv");
                csv.append("Classifier,Dataset,Average Classify Time,Total Classify Time,Train Time" + "\n");
                System.out.println("Classifier,Dataset,Average Classify Time,Total Classify Time,Train Time" + "\n");
            } catch (Exception e) {
                System.out.println("ERROR: Something went wrong in csv setup.");
                e.printStackTrace();
                System.exit(1);
            }

            //Change this back to 0.
            //12 Chinatown
            //6 Beef.
            for (int dataset = 0; dataset < dataTest.length; dataset++) {
                long resampleSeed = resampleSeedGenerator.nextLong();
                System.out.println(dataTrain[dataset].relationName());

                //Change this back to 0. 7 for HC.
                for (int classifier = 0; classifier < NumClassifiers; classifier++) {
                    System.out.println(classifiers[classifier].getClass().getSimpleName());
                    try {
                        TimingExperiment t = new TimingExperiment(classifiers[classifier], dataTest[dataset], dataTrain[dataset]);

                        //CrossValidation
                        ClassifierResults[] cvResults = t.runCrossValidation(NumResamples);

                        //Test
                        ResultWrapper rw = t.runNormalExperiment(NumResamples, resampleSeed);
                        cresults[dataset][classifier] = rw.getClassifierResults();
                        tresults[dataset][classifier] = rw.getTimingResults();

                        String output = classifiers[classifier].getClass().getSimpleName() + "," + dataTrain[dataset].relationName() + "," +
                                TimingExperiment.timingResultArrayToString(tresults[dataset][classifier]) + "\n";
                        System.out.println(output);
                        csv.append(output);
                        csv.flush();
                        File dir = new File(ResultsPath +"/"+ timeStamp + "/" + classifiers[classifier].getClass().getSimpleName() + "/Predictions/" + dataTrain[dataset].relationName() + "/" );
                        dir.mkdirs();

                        for (int resample = 0; resample < NumResamples; resample++) {
                            cresults[dataset][classifier][resample].setClassifierName(classifiers[classifier].getClass().getSimpleName());
                            cresults[dataset][classifier][resample].setDatasetName(dataTrain[dataset].relationName());
                            cresults[dataset][classifier][resample].setSplit("test");
                            cresults[dataset][classifier][resample].writeFullResultsToFile(ResultsPath +"/"+ timeStamp + "/" + classifiers[classifier].getClass().getSimpleName() + "/Predictions/" + dataTrain[dataset].relationName() + "/testFold" + resample + ".csv");

                            cvResults[resample].setClassifierName(classifiers[classifier].getClass().getSimpleName());
                            cvResults[resample].setDatasetName(dataTrain[dataset].relationName());
                            cvResults[resample].setSplit("train");
                            cvResults[resample].writeFullResultsToFile(ResultsPath +"/"+ timeStamp + "/" + classifiers[classifier].getClass().getSimpleName() + "/Predictions/" + dataTrain[dataset].relationName() + "/trainFold" + resample + ".csv");
                        }


                    } catch (Exception e) {
                        System.out.println("Something went wrong :( " + dataTrain[dataset].relationName() + " - " + classifiers[classifier].getClass().getSimpleName());
                        e.printStackTrace();
                    }
                }

                try {
                    System.out.println("HIVE-COTE");
                    for (int resample = 0; resample < NumResamples; resample++) {
                        HiveCotePostProcessed hivecote = new HiveCotePostProcessed(ResultsPath + "/" + timeStamp + "/", dataTrain[dataset].relationName(), resample, hiveCoteClassifierNames);
                        hivecote.loadResults();
                        hivecote.distributionForInstance(0);
                        hivecote.writeTestSheet(ResultsPath + "/" + timeStamp + "/");
                    }
                } catch (Exception e) {
                    System.out.println("ERROR: Something went wrong in Hivecote.");
                    e.printStackTrace();
                }


            }
            try {
                csv.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
    }


    public static Instances[] loadData(String extension, String type, String folderPath) {

        ArrayList<File> allFiles = new ArrayList<>();
        findAllFilesIncSubDirectories(allFiles, folderPath);
        ArrayList<File> validFiles = new ArrayList<>();
        for (File file : allFiles) {
            if (file.getName().endsWith("_" + type + "." + extension)) {
                validFiles.add(file);
            }
        }


        Instances[] datasets = new Instances[validFiles.size()];
        FileReader reader;
        for (int i = 0; i < datasets.length; i++)
        {
            try
            {
                reader = new FileReader(validFiles.get(i).getPath());
                datasets[i] = new Instances(reader);
                datasets[i].setClassIndex(datasets[i].numAttributes() - 1);
            } catch (Exception e)
            {
                System.out.println("Exception: " + e);
            }
        }
        return datasets;

    }

    private static void findAllFilesIncSubDirectories(ArrayList<File> allFiles, String path) {
        File curDir = new File(path);
        if (curDir.isDirectory()) {
            File[] files = curDir.listFiles();
            for (File file : files) {
                if (file.isDirectory()) {
                    findAllFilesIncSubDirectories(allFiles, file.getPath());
                }
                else {
                    allFiles.add(file);
                }
            }
        }
    }
}
