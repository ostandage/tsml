package Coursework;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLists;

public class EvaluationRunner {

    public static void main(String[] args) throws Exception {
        String[] ClassifierNames = {"LinearPerceptron", "EnhancedLinearPerceptron", "LinearPerceptronEnsemble",
                "RotF", "RandF", "IBk", "NB", "DD_DTW", "BOSS", "TSF", "C4.5"};

        double[][] balAccs = new double[DatasetLists.ReducedUCI.length][ClassifierNames.length];
        double[][] nlls = new double[DatasetLists.ReducedUCI.length][ClassifierNames.length];
        double[][] meanAUROCs = new double[DatasetLists.ReducedUCI.length][ClassifierNames.length];

        for (int d = 0; d < DatasetLists.ReducedUCI.length; d++) {
            for (int c = 0; c < ClassifierNames.length; c++) {
                int numResamples = 10;
                ClassifierResults[] crs = new ClassifierResults[numResamples];

                double balAcc = 0;
                double nll = 0;
                double meanAUROC = 0;

                for (int r = 0; r < numResamples; r++) {
                    crs[r] = new ClassifierResults();
                    crs[r].loadResultsFromFile("results/perceptrons/" + ClassifierNames[c] + "/Predictions/" + DatasetLists.ReducedUCI[d] + "/testFold" + r + ".csv");
                    crs[r].findAllStats();

                    balAcc += crs[r].balancedAcc;
                    nll += crs[r].nll;
                    meanAUROC += crs[r].meanAUROC;
                }

                balAccs[d][c] = balAcc / numResamples;
                nlls[d][c] = nll / numResamples;
                meanAUROCs[d][c] = meanAUROC / numResamples;
            }
        }

        System.out.println(",LinearPerceptron,EnhancedLinearPerceptron,LinearPerceptronEnsemble,RotF,RandF,IBk,NB,DD_DTW,BOSS,TSF,C4.5");

        //Print matrix
        for (int d = 0; d < DatasetLists.ReducedUCI.length; d++) {
            System.out.print(DatasetLists.ReducedUCI[d]);
            for (int c = 0; c < ClassifierNames.length; c++) {
                //change array
                System.out.print("," + meanAUROCs[d][c]);
            }
            System.out.println();
        }
    }
}
