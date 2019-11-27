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

            //Hive-Cote
            classifiers[7] = new ElasticEnsemble();
            classifiers[8] = new ShapeletTransformClassifier();
            classifiers[9] = new RISE();
            classifiers[10] = new BOSS();
            classifiers[11] = new TSF();

            Classifier[] hcClassifiers = Arrays.copyOfRange(classifiers, 7,12);

            HIVE_COTE hc = new HIVE_COTE();
            hc.setupDefaultEnsembleSettings();
            hc.setClassifiers(hcClassifiers, hc.getClassifierNames(), null);
            classifiers[12] = hc;


            String timeStamp = java.time.ZonedDateTime.now().toLocalDateTime().toString();
            timeStamp = timeStamp.replace(':', '-');
            System.out.println(timeStamp + "\n");
            File path = new File("results/" + timeStamp);
            path.mkdirs();

            FileWriter csv = new FileWriter("results/" + timeStamp + "/" + "timing.csv");
            csv.append("Dataset,Classifier,Average Classify Time,Total Classify Time,Train Time" + "\n");
            System.out.println("Dataset,Classifier,Average Classify Time,Total Classify Time,Train Time" + "\n");
            
            //Change this back to 0
            for (int dataset = 6; dataset < dataTest.length; dataset++) {
                //Change this back to 0
                for (int classifier = 8; classifier < NumClassifiers-1; classifier++) {
                    try {
                        TimingExperiment t = new TimingExperiment(classifiers[classifier], dataTest[dataset], dataTrain[dataset]);
                        ResultWrapper rw = t.runNormalExperiment(NumResamples);
                        cresults[dataset][classifier] = rw.getClassifierResults();
                        tresults[dataset][classifier] = rw.getTimingResults();


                        String output = dataTrain[dataset].relationName() + "," + classifiers[classifier].getClass().getSimpleName() + "," +
                                        TimingExperiment.timingResultArrayToString(tresults[dataset][classifier]) + "\n";
                        System.out.println(output);
                        csv.append(output);
                        csv.flush();
                        File dir = new File("results/" + timeStamp + "/" + dataTrain[dataset].relationName() + "/" );
                        dir.mkdirs();

                        //How to handle resamples?
                        for (int resample = 0; resample < NumResamples; resample++) {
                            cresults[dataset][classifier][resample].writeFullResultsToFile("results/" + timeStamp + "/" + dataTrain[dataset].relationName() + "/" + classifiers[classifier].getClass().getSimpleName() + "-" + resample + ".csv");
                        }


                    } catch (Exception e) {
                        System.out.println("Something went wrong :( " + dataTrain[dataset].relationName() + " - " + classifiers[classifier].getClass().getSimpleName());
                        e.printStackTrace();
                    }
                }
            }
            csv.close();
    }
    
    
    private static Instances[] loadData(String extension, String type, String folderPath) {
        
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

    private static void HiveCotePostProcessed() {
        
    }
}
