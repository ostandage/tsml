//Want to have some sort of mechanism to configure the enhancedLinearPerceptron.
//Seems very inefficient to be duplicating so much data. Maybe a quicker way with references / on the fly ignore.
package Coursework;

import labs.WekaTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class LinearPerceptronEnsemble extends AbstractClassifier {

    private int EnsembleSize;
    private double AttributeSubsetProportion;
    private EnhancedLinearPerceptron[] Ensemble;
    private int[][] RemovedEnsembleAttributes;
    private EnhancedLinearPerceptron BaseClassifier;

    public static void main (String[] args) throws Exception {
        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        Instances train = WekaTools.loadClassificationData("data/UCIContinuous/conn-bench-vowel-deterding/conn-bench-vowel-deterding_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("data/UCIContinuous/conn-bench-vowel-deterding/conn-bench-vowel-deterding_TEST.arff");

        lpe.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(lpe, test);
        System.out.println(eval.toSummaryString());
        System.out.println("Error Rate: " + eval.errorRate());
    }

    public LinearPerceptronEnsemble() {
        EnsembleSize = 50;
        AttributeSubsetProportion = 0.5;

        //Default to Enhanced LP with default parameters.
        BaseClassifier = new EnhancedLinearPerceptron();
    }

    public LinearPerceptronEnsemble (EnhancedLinearPerceptron baseClassifier) {
        EnsembleSize = 50;
        AttributeSubsetProportion = 0.5;
        BaseClassifier = baseClassifier;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Random rnd = new Random();
        int attrToRemove = data.numAttributes() - (int)(data.numAttributes() * AttributeSubsetProportion);
        RemovedEnsembleAttributes = new int[EnsembleSize][attrToRemove];

        Instances[] datas = new Instances[EnsembleSize];
        for (int i = 0; i < EnsembleSize; i++) {
            //Duplicate the data for each classifier.
            datas[i] = new Instances(data);

            //Find out what attributes each classifier will use.
            //Priority queue?

            //Don't think we actually need to know which we use, just which we don't so can remove before classify / dist.
            TreeSet<Integer> attribs = new TreeSet<>();
            while (attribs.size() < attrToRemove) {
                attribs.add(rnd.nextInt(data.numAttributes()));
            }

            //This should leave us an int array ordered largest to smallest so can remove from the data.
            for (int j = 0; j < attrToRemove; j++) {
                RemovedEnsembleAttributes[i][j] = attribs.last();
                datas[i].deleteAttributeAt(RemovedEnsembleAttributes[i][j]);
            }

            //Would be nice to remove this cast somehow...
            //This cast will break the overrides on build classifier etc :(
            Ensemble[i] = (EnhancedLinearPerceptron) AbstractClassifier.makeCopy(BaseClassifier);

            Ensemble[i].buildClassifier(datas[i]);
        }
    }

    @Override
    public double classifyInstance(Instance data) throws Exception {
        double[] dist = distributionForInstance(data);
        int curMaxIndex = 0;
        for (int i = 1; i < dist.length; i++) {
            if (dist[i] > dist[curMaxIndex]) {
                curMaxIndex = i;
            }
        }
        return dist[curMaxIndex];
    }

    @Override
    public double[] distributionForInstance(Instance data) throws Exception {

        Instance[] datas = new Instance[EnsembleSize];
        double[] distribution = new double[data.numClasses()];

        for (int i = 0; i < EnsembleSize; i++) {
            datas[i] = (Instance) data.copy();

            for (int j = 0; j < RemovedEnsembleAttributes[i].length; j++) {
                datas[i].deleteAttributeAt(RemovedEnsembleAttributes[i][j]);
            }

            double[] singleDist = Ensemble[i].distributionForInstance(datas[i]);
            for (int j = 0; j < data.numClasses(); j++) {
                distribution[j] += singleDist[j];
            }
        }

        for (int i = 0; i < data.numClasses(); i++) {
            distribution[i] = distribution[i] / EnsembleSize;
        }
        return distribution;
    }

    public int getEnsembleSize() {
        return EnsembleSize;
    }

    public void setEnsembleSize(int size) {
        EnsembleSize = size;
    }

    public double getAttributeSubsetProportion() {
        return AttributeSubsetProportion;
    }

    public void setAttributeSubsetProportion(double proportion) {
        AttributeSubsetProportion = proportion;
    }
}
