/**
 * Wrapper class to hold both ClassifierResults and TimingResults files for a single experiment.
 */
package timing;

import evaluation.storage.ClassifierResults;

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
