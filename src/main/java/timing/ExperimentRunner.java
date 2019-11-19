//Critical Difference diagrams
//File writer

package timing;

import evaluation.storage.ClassifierResults;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.DD_DTW;
import timeseriesweka.classifiers.distance_based.DTW_kNN;
import timeseriesweka.classifiers.distance_based.FastDTW;
import timeseriesweka.classifiers.distance_based.FastDTW_1NN;
import timeseriesweka.classifiers.distance_based.ProximityForestWrapper;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.hybrids.HiveCote;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

/**
 *
 * @author ostandage
 */
public class ExperimentRunner {
    
    private static final int NumClassifiers = 10;
    private static final int NumResamples = 1;
    
    
    public static void main(String[] args) throws Exception {
            Instances[] dataTrain = loadData("arff", "TRAIN", "data/Univariate_arff");
            Instances[] dataTest = loadData("arff", "TEST", "data/Univariate_arff");
            Classifier[] classifiers = new Classifier[NumClassifiers];
            ClassifierResults[][][] cresults = new ClassifierResults[dataTest.length][NumClassifiers][];
            TimingResults[][][] tresults = new TimingResults[dataTest.length][NumClassifiers][];
            
            classifiers[0] = new IBk();
            classifiers[1] = new DTW_kNN(1);
            classifiers[2] = new DTW_kNN(2);
            classifiers[3] = new DTW1NN();
            classifiers[4] = new BOSS();
            classifiers[5] = new DD_DTW();
            classifiers[6] = new FastDTW();
            classifiers[7] = new FastDTW_1NN();
            classifiers[8] = new ProximityForestWrapper();
            classifiers[9] = new HiveCote(); //HIVE_COTE -- post process previous results.


            String timeStamp = java.time.ZonedDateTime.now().toLocalDateTime().toString();
            timeStamp = timeStamp.replace(':', '-');
            System.out.println(timeStamp + "\n");

            FileWriter csv = new FileWriter("results/timingExperiment-" + timeStamp + ".csv");
            
            
            //Change this back :)
            for (int dataset = 6; dataset < 7; dataset++) {
                for (int classifier = 0; classifier < NumClassifiers; classifier++) {
                    try {
                        TimingExperiment t = new TimingExperiment(classifiers[classifier], dataTest[dataset], dataTrain[dataset]);
                        ResultWrapper rw = t.runNormalExperiment(NumResamples);
                        cresults[dataset][classifier] = rw.getClassifierResults();
                        tresults[dataset][classifier] = rw.getTimingResults();

                        String output = dataTrain[dataset].relationName() + " - " + classifiers[classifier].getClass().getSimpleName() + "\n" + 
                                        "Average Classify Time,Total Classify Time,Train Time" + "\n" + 
                                        TimingExperiment.timingResultArrayToString(tresults[dataset][classifier]) + "\n" + 
                                        cresults[dataset][classifier][0].writeFullResultsToString();
                        System.out.println(output);
                        csv.append(output);
                        csv.flush();
                    } catch (Exception e) {
                        
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
}
