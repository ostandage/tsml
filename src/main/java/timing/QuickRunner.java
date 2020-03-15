package timing;

import experiments.data.DatasetLists;

public class QuickRunner {


    public static void main(String[] args) throws Exception {

        //postProcessHiveCote();
        //dtwKnnTesting();
        DTWTrainingTesting();

    }

    private static void DTWTrainingTesting() throws Exception {
        String[] datasets = DatasetLists.tscProblems85;
        boolean skip = false;
        for (int i = 0; i < datasets.length; i++) {
            //Allows us to resume at a given dataset.
            if (datasets[i] == "ElectricDevices") {
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

        String[] datasets = DatasetLists.tscProblems85;

        for (String dataset : datasets) {
            runnerArgs[3] = DataPath + dataset;
            try {
                NewRunner.main(runnerArgs);
            } catch (Exception e) {
                System.out.println("Error on dataset: " + dataset);
            }
        }
    }

    //KNN better (K=23) for:            1       23
    //DistalPhalanxOutlineCorrect       0.717   0.746
    //DistalPhalanxTW                   0.589   0.626
    //Earthquakes                       0.719   0.726
    //FordA                             0.554   0.576
    //FordB                             0.619   0.622

    //Tested alphabetically up to and including Ham.


    private static void postProcessHiveCote() {
        String Identifier = "20200116";
        String Resample = "1";
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
                NewRunner.main(runnerArgs);
            } catch (Exception e) {
                System.out.println("Error on dataset: " + dataset);
            }
        }
    }

}
