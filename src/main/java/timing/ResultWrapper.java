/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timing;

import evaluation.storage.ClassifierResults;

/**
 *
 * @author ostandage
 */
public class ResultWrapper {
    private TimingResults[] timingResults;
    private ClassifierResults[] classifierResults;
    
    public ResultWrapper(TimingResults[] timingResults, ClassifierResults[] classifierResults) {
        this.timingResults = timingResults;
        this.classifierResults = classifierResults;
    }
    
    public TimingResults[] getTimingResults() {
        return timingResults;
    }
    
    public ClassifierResults[] getClassifierResults() {
        return classifierResults;
    }
}
