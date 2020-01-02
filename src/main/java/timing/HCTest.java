package timing;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.*;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.hybrids.cote.HiveCotePostProcessed;
import timeseriesweka.classifiers.interval_based.TSF;
import timeseriesweka.classifiers.shapelet_based.ShapeletTransformClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.util.ArrayList;

public class HCTest {
        private static final int NumClassifiers = 13;
        private static final int NumResamples = 1;

        public static void main(String args[]) throws Exception{
                Classifier[] classifiers = new Classifier[NumClassifiers];

                //Hive-Cote Classifiers
                classifiers[7] = new ElasticEnsemble();
                classifiers[8] = new ShapeletTransformClassifier();
                classifiers[9] = new RISE();
                classifiers[10] = new BOSS();
                classifiers[11] = new TSF();

                ArrayList<String> hiveCoteClassifierNames = new ArrayList<>();
                hiveCoteClassifierNames.add(classifiers[7].getClass().getSimpleName());
                hiveCoteClassifierNames.add(classifiers[8].getClass().getSimpleName());
                hiveCoteClassifierNames.add(classifiers[9].getClass().getSimpleName());
                hiveCoteClassifierNames.add(classifiers[10].getClass().getSimpleName());
                hiveCoteClassifierNames.add(classifiers[11].getClass().getSimpleName());


                HiveCotePostProcessed hivecote = new HiveCotePostProcessed("results/testHC/", "Chinatown", 1, hiveCoteClassifierNames);
                hivecote.setAlpha(1);
                hivecote.writeTestSheet("hiveCoteProcessed");
        }

}
