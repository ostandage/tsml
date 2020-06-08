/**
 * This class is primarily for testing other classes as command line arguments are passed in as on the HPC.
 */

package timing;

import experiments.data.DatasetLists;

public class QuickRunner {
    public static void main(String[] args) throws Exception {
        //postProcessHiveCote();
        dtwKnnTesting();
        //DTWTrainingTesting();
    }

    private static void DTWTrainingTesting() throws Exception {
        String[] datasets = DatasetLists.tscProblems85;
        boolean skip = true;
        for (int i = 0; i < datasets.length; i++) {

            //Allows us to resume at a given dataset.
            if (datasets[i] == "Haptics") {
                skip = false;
            }

            if (!skip)
            {
                String datapath = "data/Univariate_arff/" + datasets[i] + "/"+ datasets[i];
                String resultsPath = "results/DTWTraining/" + datasets[i] + ".csv";

                String[] args = {datasets[i], datapath, resultsPath};
                DTWTrainingExperiments.main(args);
            }
        }
    }

    private static void dtwKnnTesting() {
        String Identifier = "dtwKnn";
        String Resample = "0";
        String ClassifierIndex = "4";
        String DataPath = "data/Univariate_arff/";
        String ResultsPath = "results";
        String NumThreads = "4";

        String[] runnerArgs = new String[6];
        runnerArgs[0] = Identifier;
        runnerArgs[1] = Resample;
        runnerArgs[2] = ClassifierIndex;
        runnerArgs[4] = ResultsPath;
        runnerArgs[5] = NumThreads;

//        String[] datasets = DatasetLists.tscProblems85;
//
//        for (String dataset : datasets) {
//            //runnerArgs[3] = DataPath + dataset;
//            try {
//                TimingRunner.main(runnerArgs);
//            } catch (Exception e) {
//                System.out.println("Error on dataset: " + dataset);
//            }
//        }
        runnerArgs[3] = DataPath + "GunPoint";
        try {
            TimingRunner.main(runnerArgs);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void postProcessHiveCote() {
        String Identifier = "20200305";
        String Resample = "10";
        String ClassifierIndex = "25";
        String DataPath = "data/Univariate_arff/";
        String ResultsPath = "results";
        String NumThreads = "1";

        String[] runnerArgs = new String[6];
        runnerArgs[0] = Identifier;
        runnerArgs[1] = Resample;
        runnerArgs[2] = ClassifierIndex;
        runnerArgs[4] = ResultsPath;
        runnerArgs[5] = NumThreads;

        String[] datasets = DatasetLists.tscProblems85;

        for (String dataset : datasets) {
            runnerArgs[3] = DataPath + dataset;
            try {
                TimingRunner.main(runnerArgs);
            } catch (Exception e) {
                System.out.println("Error on dataset: " + dataset);
            }
        }
    }
}
