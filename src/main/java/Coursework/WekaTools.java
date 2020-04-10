package Coursework;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Iterator;
import java.util.Random;


public class WekaTools {

    public static void main(String args[]) {
        int[] actual    = {0,0,1,1,1,0,0,1,1,1};
        int[] predicted = {0,1,1,1,1,1,1,1,1,1};

        int[][] conf = confusionMatrix(predicted, actual);

        printConfusionMatrix(conf);
    }

    public static int[] getClassValues(Instances data) {
        int[] res = new int[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            res[i] = (int) data.get(i).classValue();
        }
        return res;
    }

    public static int[] classifyInstances(Classifier c, Instances test) throws Exception {
        int[] res = new int[test.numInstances()];

        for (int i = 0; i < test.numInstances(); i++) {
            res[i] = (int) c.classifyInstance(test.get(i));
        }
        return res;
    }

    public static void printConfusionMatrix(int[][] matrix) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                System.out.print(matrix[x][y] + "    ");
            }
            System.out.println();
        }
    }

    public static int[][] confusionMatrix(int[] predicted, int[] actual) {
        int[][] matrix = new int[2][2];

        for (int i = 0; i < actual.length; i++) {
            if ((actual[i] == 0) && (predicted[i]) == 0) {
                matrix[0][0]++;
            }
            else if ((actual[i] == 1) && (predicted[i]) == 0) {
                matrix[1][0]++;
            }
            else if ((actual[i] == 0) && (predicted[i]) == 1) {
                matrix[0][1]++;
            }
            else {
                matrix[1][1]++;
            }
        }
        return matrix;
    }

    public static double accuracy(Classifier c, Instances test) throws Exception {
        double numCorrect = 0;
        Iterator<Instance> insts = test.iterator();
        while (insts.hasNext()) {
            Instance i = insts.next();
            double prediction = c.classifyInstance(i);
            if (i.classValue() == prediction) {
                numCorrect++;
            }
        }
        double accuracy = numCorrect / test.numInstances();
        return accuracy;     
    }
    
    public static Instances loadClassificationData(String path) {
        Instances train;
       
        FileReader reader;
        try
        {
            reader = new FileReader(path);
            train = new Instances(reader);
            train.setClassIndex(train.numAttributes() - 1);
            return train;
            
        } catch (Exception e)
        {
            System.out.println("Exception: " + e);
        }
        return null;
    }

    public static Instances[] splitData(Instances all, double proportion) {
        Random r = new Random();
        for (int i = 0; i < 100; i++) {
            all.swap((r.nextInt(1000) % (all.numInstances()-1) ) , (r.nextInt(1000) % (all.numInstances() -1)));
        }
        
        int endOf0 = (int) (proportion * all.numInstances());
        Instances[] split = new Instances[2];
        split[0] = new Instances(all, 0, endOf0);
        split[1] = new Instances(all, endOf0, all.numInstances()-endOf0);
        return split;
    }
    
    public static double[] classDistribution(Instances data) {
        double[] distribution = new double[data.numClasses()];

        for (int inst = 0; inst < data.numInstances(); inst++) {
            for (int c = 0; c < data.numClasses(); c++) {
                distribution[c] = distribution[c] + data.get(inst).value(c);
            }
        }

        double total = 0;
        for (int c = 0; c < data.numClasses(); c++) {
            total = total + distribution[c];
        }

        for (int c = 0; c < data.numClasses(); c++) {
            distribution[c] = distribution[c] / total;
        }
        return distribution;
    }

    public static String printDoubleArray(double[] data) {
        String output = "[";
        for (Double d: data) {
            output += d + ", ";
        }
        return output + "]";
    }
}
