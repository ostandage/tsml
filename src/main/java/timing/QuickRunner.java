package timing;

import experiments.data.DatasetLists;

public class QuickRunner {


    public static void main(String[] args) {

        //postProcessHiveCote();
        dtwKnnTesting();

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
