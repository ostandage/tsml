//Critical Difference diagrams

package timing;

import evaluation.storage.ClassifierResults;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;

import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.*;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.hybrids.HiveCote;
import timeseriesweka.classifiers.hybrids.cote.HiveCotePostProcessed;
import timeseriesweka.classifiers.interval_based.TSF;
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
    
    private static final int NumClassifiers = 13;
    private static final int NumResamples = 5;
    
    
    public static void main(String[] args) throws Exception {
            Instances[] dataTrain = loadData("arff", "TRAIN", "data/Univariate_arff");
            Instances[] dataTest = loadData("arff", "TEST", "data/Univariate_arff");
            Classifier[] classifiers = new Classifier[NumClassifiers];
            ClassifierResults[][][] cresults = new ClassifierResults[dataTest.length][NumClassifiers][];
            TimingResults[][][] tresults = new TimingResults[dataTest.length][NumClassifiers][];
            
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


            String timeStamp = java.time.ZonedDateTime.now().toLocalDateTime().toString();
            timeStamp = timeStamp.replace(':', '-');
            System.out.println(timeStamp + "\n");
            File path = new File("results/" + timeStamp);
            path.mkdirs();

            FileWriter csv = new FileWriter("results/" + timeStamp + "/" + "timing.csv");
            csv.append("Classifier,Dataset,Average Classify Time,Total Classify Time,Train Time" + "\n");
            System.out.println("Classifier,Dataset,Average Classify Time,Total Classify Time,Train Time" + "\n");

            //Change this back to 0. 6 is beef.
            for (int dataset = 6; dataset < dataTest.length; dataset++) {
                System.out.println(dataTrain[dataset].relationName());
                //Change this back to 0. 7 for HC.
                for (int classifier = 7; classifier < NumClassifiers-1; classifier++) {
                    System.out.println(classifiers[classifier].getClass().getSimpleName());
                    try {
                        TimingExperiment t = new TimingExperiment(classifiers[classifier], dataTest[dataset], dataTrain[dataset]);
                        ResultWrapper rw = t.runNormalExperiment(NumResamples);
                        cresults[dataset][classifier] = rw.getClassifierResults();
                        tresults[dataset][classifier] = rw.getTimingResults();


                        String output = classifiers[classifier].getClass().getSimpleName() + "," + dataTrain[dataset].relationName() + "," +
                                TimingExperiment.timingResultArrayToString(tresults[dataset][classifier]) + "\n";
                        System.out.println(output);
                        csv.append(output);
                        csv.flush();
//                        File dir = new File("results/" + timeStamp + "/" + dataTrain[dataset].relationName() + "/" );
                        File dir = new File("results/" + timeStamp + "/" + classifiers[classifier].getClass().getSimpleName() + "/Predictions/" + dataTrain[dataset].relationName() + "/" );
                        dir.mkdirs();

                        //How to handle resamples0?
                        for (int resample = 0; resample < NumResamples; resample++) {
                            cresults[dataset][classifier][resample].writeFullResultsToFile("results/" + timeStamp + "/" + classifiers[classifier].getClass().getSimpleName() + "/Predictions/" + dataTrain[dataset].relationName() + "/testFold" + resample + ".csv");
                            //No - need to make this the same as above, but need to do the crossvalidation first to get the trainFold data. 
                            tresults[dataset][classifier][resample].getTrainAccuracyEstimator().writeTrainEstimatesToFile("results/" + timeStamp + "/" + classifiers[classifier].getClass().getSimpleName() + "/Predictions/" + dataTrain[dataset].relationName() + "/trainFold" + resample + ".csv");
                        }


                    } catch (Exception e) {
                        System.out.println("Something went wrong :( " + dataTrain[dataset].relationName() + " - " + classifiers[classifier].getClass().getSimpleName());
                        e.printStackTrace();
                    }
                }

                //Hive-Cote Processing
                //Need to setup for resamples.
                //HiveCotePostProcessed hivecote = new HiveCotePostProcessed("results/" + timeStamp + "/" , dataTrain[dataset].relationName(), 0, hiveCoteClassifierNames);
                for (int resample = 0; resample < NumResamples; resample++) {
                    HiveCotePostProcessed hivecote = new HiveCotePostProcessed("results/" + timeStamp + "/", dataTrain[dataset].relationName(), resample, hiveCoteClassifierNames);
                    hivecote.setAlpha(1);
                    hivecote.writeTestSheet(timeStamp);
                }


//                TimingExperiment hivecoteT = new TimingExperiment(hivecote, dataTest[dataset], dataTrain[dataset]);
//                ResultWrapper hivecoteRW = hivecoteT.runNormalExperiment(NumResamples);
//                cresults[dataset][NumClassifiers-1] = hivecoteRW.getClassifierResults();
//                tresults[dataset][NumClassifiers-1] = hivecoteRW.getTimingResults();



            }
            csv.close();
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
