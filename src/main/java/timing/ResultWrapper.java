/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timing;

import evaluation.storage.ClassifierResults;
import weka.classifiers.Classifier;

/**
 *
 * @author ostandage
 */
public class ResultWrapper {
    private TimingResults timingResults;
    private ClassifierResults classifierResults;
    private ClassifierResults trainResults;
    
    public ResultWrapper(TimingResults timingResults, ClassifierResults classifierResults, ClassifierResults trainResults) {
        this.timingResults = timingResults;
        this.classifierResults = classifierResults;
        this.trainResults = trainResults;
    }
    
    public TimingResults getTimingResults() {
        return timingResults;
    }
    
    public ClassifierResults getClassifierResults() {
        return classifierResults;
    }
    public ClassifierResults getTrainResults() {
        return trainResults;
    }
}
