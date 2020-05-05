/**
 * This class is used for generating statistics about the datasets for evaluation.
 */
package timing;

import experiments.data.DatasetLists;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;

public class DataDescription {

    public static void main(String[] args) {
        File dataDir = new File("data/Univariate_arff");
        String[] datasets = DatasetLists.tscProblems85;
        Instances[] trains = new Instances[datasets.length];
        Instances[] tests  = new Instances[datasets.length];

        for (int d = 0; d < datasets.length; d++) {
            trains[d] = WekaTools.loadData(dataDir + "/" + datasets[d], TimingRunner.DatasetType.TRAIN);
            tests[d] = WekaTools.loadData(dataDir + "/" + datasets[d], TimingRunner.DatasetType.TEST);
        }

        System.out.println("Name,Num Att,Num Train,Num Test,Num Classes,Class Distribution");
        for (int d = 0; d < datasets.length; d++) {
            String distribution = classDistribution(trains[d], tests[d]);
            System.out.println(trains[d].relationName() + "," +
                            trains[d].numAttributes() + "," +
                            trains[d].numInstances()  + "," +
                            tests[d].numInstances()  + "," +
                            trains[d].numClasses()  + "," +
                            distribution);
        }
    }

    public static String classDistribution(Instances train, Instances test) {
        Instances data = new Instances(train);
        data.addAll(test);
        int[] classDistribution = new int[data.numInstances()];

        for (Instance instance : data) {
            classDistribution[(int) instance.classValue()]++;
        }

        String distribution = "";
        boolean tailFound = false;
        for (int i = classDistribution.length-1; i >= 0; i--) {
            if (tailFound) {
                distribution = i + ":" + classDistribution[i] + " " + distribution;
            }
            else if (classDistribution[i] > 0) {
                tailFound = true;
                distribution = i + ":" + classDistribution[i] + " " + distribution;
            }
        }
        return distribution;
    }
}
