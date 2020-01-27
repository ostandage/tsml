package labs;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Enumeration;

public class Lab1 {

    public static void main (String[] args) throws Exception {
        Instances train = loadData("data/Arsenal_TRAIN.arff");
        Instances test = loadData("data/Arsenal_TEST.arff");
        System.out.println(train.numInstances());
        System.out.println(test.numAttributes());

        Enumeration<Instance> instances = train.enumerateInstances();
        int numWins = 0;
        while (instances.hasMoreElements()) {
            Instance i = instances.nextElement();
            if (i.stringValue(3).equals("Win")) {
                numWins++;
            }
        }
        System.out.println("Num wins:       " + numWins);

        System.out.println(printDoubleArray(test.get(4).toDoubleArray()));
        System.out.println();
        for (int i = 0; i < train.numInstances(); i++) {
            System.out.println(train.get(i));
        }

        test.deleteAttributeAt(2);
        train.deleteAttributeAt(2);
        System.out.println();
        for (int i = 0; i < train.numInstances(); i++) {
            System.out.println(train.get(i));
        }

        train = loadData("data/Arsenal_TRAIN.arff");
        test = loadData("data/Arsenal_TEST.arff");
        train.setClassIndex(train.numAttributes() -1);
        test.setClassIndex(test.numAttributes() -1);

        NaiveBayes nb = new NaiveBayes();
        IBk ib = new IBk();

        nb.buildClassifier(train);
        ib.buildClassifier(train);

        int nbCorrect = 0;
        int ibCorrect = 0;

        for (Instance i : test) {
            if (nb.classifyInstance(i) == i.classValue()) {
                nbCorrect++;
            }
            if (ib.classifyInstance(i) == i.classValue()) {
                ibCorrect++;
            }
        }
        double nbAccuracy = (double) nbCorrect / test.numInstances();
        double ibAccuracy = (double) ibCorrect / test.numInstances();
        System.out.println("NB Accuracy: " + nbAccuracy);
        System.out.println("IB Accuracy: " + ibAccuracy);
        System.out.println();

        for (Instance i : test) {
            System.out.println(printDoubleArray(nb.distributionForInstance(i)));
            System.out.println(printDoubleArray(ib.distributionForInstance(i)));
            System.out.println();
        }

        System.out.println("Histogram Classifier");
        HistogramClassifier hc = new HistogramClassifier();
        hc.setNumBins(10);
        hc.setClassValue(0);
        Instances hInsts = loadData("data/histogram.arff");
        hc.buildClassifier(hInsts);

        for (int i = 0; i < hInsts.numInstances(); i++) {
            System.out.println(printDoubleArray(hc.distributionForInstance(hInsts.instance(i))));
        }
    }

    public static class HistogramClassifier implements Classifier {

        private int numBins = 10;
        private double interval;
        private int classValue = 0;
        private double[][] histograms;

        public void setNumBins (int numBins) {
            this.numBins = numBins;
        }

        public void setClassValue (int classValue) {
            this.classValue = classValue;
        }

        @Override
        public void buildClassifier(Instances instances) throws Exception {
            histograms = new double[numBins][instances.numClasses()];

            double min = Integer.MAX_VALUE;
            double max = Integer.MIN_VALUE;
            for (Instance i  : instances) {
                if (i.attribute(classValue).getUpperNumericBound() > max) {
                    max = i.attribute(classValue).getUpperNumericBound();
                }
                if (i.attribute(classValue).getLowerNumericBound() < min) {
                    min = i.attribute(classValue).getLowerNumericBound();
                }
            }

            interval = (max - min) / numBins;

            instances.setClassIndex(classValue);

            for (int bin = 0; bin < numBins; bin++) {
                for (Instance i : instances) {
                    double[] data = i.toDoubleArray();

                    for (int j = 0; j < data.length; j++) {
                        if ((data[j] >= interval*bin) && (data[j] < (interval+1)*bin)) {
                            histograms[bin][i.classIndex()]++;
                        }
                    }
                }
            }
        }

        @Override
        public double classifyInstance(Instance instance) throws Exception {
            double[] dist = distributionForInstance(instance);
            int maxIndex = 0;
            for (int i = 0; i < dist.length; i++) {
                if (dist[i] > dist[maxIndex]) {
                    maxIndex = i;
                }
            }
            return dist[maxIndex];
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            double[] data = instance.toDoubleArray();
            double[] dist = new double[instance.numClasses()];

            for (int bin = 0; bin < numBins; bin++) {
                for (int j = 0; j < data.length; j++) {
                    if ((data[j] >= interval*bin) && (data[j] < (interval+1)*bin)) {

                    }
                }
            }
            return dist;
        }

        @Override
        public Capabilities getCapabilities() {
            return null;
        }
    }

    public static Instances loadData (String path) {
        Instances data = null;
        try{
            FileReader reader = new FileReader(path);
            data = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return data;
    }

    public static String printDoubleArray(double[] data) {
        String output = "[";
        for (Double d: data) {
            output += d + ", ";
        }
        return output + "]";
    }
}
