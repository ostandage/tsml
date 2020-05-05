/**
 * Holds the timing data for a single classifier-dataset fold.
 */
package timing;

import java.text.DecimalFormat;

public class TimingResults {
    //Stores the time to classify each test instance.
    private double[] classifyTimes;

    private double trainTime;

    public TimingResults (double[] classifyTimes, double trainTime) {
        this.classifyTimes = classifyTimes;
        this.trainTime = trainTime;
    }
    
    public double averageTime() {
        return calcMean(classifyTimes);
    }
    
    public double totalClassifyTime() {
        return sumArray(classifyTimes);
    }
    
    public double trainTime() {
        return trainTime;
    }
    
    public double[] getTimes() {
        return classifyTimes;
    }

    private double sumArray(double[] data) {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum = sum + data[i];
        }
        return sum;
    }
    
    private double calcMean(double[] data) {
        return sumArray(data) / data.length;
    }
    
    @Override
    public String toString() {
        DecimalFormat accf = new DecimalFormat("00.00");
        DecimalFormat timef = new DecimalFormat("0.000");
        return (timef.format(averageTime()) + "," + 
                timef.format(totalClassifyTime()) + "," + timef.format(trainTime()));
        
    }
    
    public static TimingResults combineResults(TimingResults[] results) {
        int numSetsPerResult = results[0].getTimes().length;
        double[] times = new double[numSetsPerResult * results.length];
        int arrayPos = 0;
        double averageTrainTime = 0;
        for (TimingResults result : results) {
            System.arraycopy(result.getTimes(), 0, times, arrayPos, numSetsPerResult);
            averageTrainTime = averageTrainTime + result.trainTime();
            arrayPos = arrayPos + numSetsPerResult;
        }
        averageTrainTime = averageTrainTime / results.length;
        
        return new TimingResults(times, averageTrainTime);
    }
}