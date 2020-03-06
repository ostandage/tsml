package timing;

import experiments.data.DatasetLists;

public class HiveCotePPRunner {

    private static String Identifier = "20200116";
    private static String Resample = "1";
    private static String ClassifierIndex = "25";
    private static String DataPath = "data/Univariate_arff/";
    private static String ResultsPath = "results";
    private static String NumThreads = "1";

    public static void main(String[] args) {

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
