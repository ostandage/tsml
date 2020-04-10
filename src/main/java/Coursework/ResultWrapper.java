package Coursework;

import evaluation.storage.ClassifierResults;

public class ResultWrapper {
    private TimingResults timingResults;
    private ClassifierResults classifierResults;

    public ResultWrapper(TimingResults timingResults, ClassifierResults classifierResults) {
        this.timingResults = timingResults;
        this.classifierResults = classifierResults;
    }
    
    public TimingResults getTimingResults() {
        return timingResults;
    }
    public ClassifierResults getClassifierResults() {
        return classifierResults;
    }
}
