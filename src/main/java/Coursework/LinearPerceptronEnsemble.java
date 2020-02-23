package Coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Random;

public class LinearPerceptronEnsemble extends AbstractClassifier {

    private int EnsembleSize;
    private double AttributeSubsetProportion;
    private LinearPerceptron[] Ensemble;
    private int[][] EnsembleAttributes;


    public LinearPerceptronEnsemble() {
        EnsembleSize = 50;
        AttributeSubsetProportion = 0.5;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Random rnd = new Random();
        int attrPerClassifier = (int) (data.numAttributes() * AttributeSubsetProportion);
        EnsembleAttributes = new int[EnsembleSize][attrPerClassifier];

        Instances[] datas = new Instances[EnsembleSize];
        for (int i = 0; i < EnsembleSize; i++) {
            //Duplicate the data for each classifier.
            datas[i] = new Instances(data);

            //Find out what attributes each classifier will use.
            for (int j = 0; j < attrPerClassifier; j++) {
                //need to remove x random attributes from classifiers data, but as remove i it will offset the values.
                //sort attributes array largest to smallest so can remove without messing up indexes.
                
            }
        }



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
