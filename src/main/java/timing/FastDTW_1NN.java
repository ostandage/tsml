/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * Elements of this class have been adapted for the project, specifically making classifyInstance threaded and capable
 * of calculating k nearest neighbours. Comments have been added to show changes.
 */

package timing;
import fileIO.OutFile;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import timeseriesweka.elastic_distance_measures.DTW;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import weka_extras.classifiers.SaveEachParameter;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.SaveParameterInfo;
import timeseriesweka.classifiers.TrainAccuracyEstimator;

public class FastDTW_1NN extends AbstractClassifier  implements SaveParameterInfo, TrainAccuracyEstimator,SaveEachParameter,ParameterSplittable{
    private boolean optimiseWindow=false;
    private double windowSize=1;
    private int maxPercentageWarp=100;
    private Instances train;
    private int trainSize;
    private int bestWarp;
    private int maxWindowSize;

    /** Project Added Code **/
    private int maxNoThreads = 1;
    private int k = 1;
    /** END **/

    DTW_DistanceBasic dtw;
    HashMap<Integer,Double> distances;
    double maxR=1;
    ArrayList<Double> accuracy=new ArrayList<>();
    String trainPath;
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
    private ClassifierResults res =new ClassifierResults();
    
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }
 @Override
    public void writeTrainEstimatesToFile(String train) {
        trainPath=train;
    } 
    public void setFindTrainAccuracyEstimate(boolean setCV){
        if(setCV==true)
            throw new UnsupportedOperationException("Doing a top level CV is not yet possible for FastDTW_1NN. It cross validates to optimize, so could store those, but will be biased");
    }
    public int getK() {return k;}

    @Override
    public ClassifierResults getTrainResults(){
        return res;
    }      
    @Override
    public String getParas() {
        return getParameters();
    }

    @Override
    public double getAcc() {
        return res.getAcc();
    }

    public void setK(int k) {this.k = k;}

    public FastDTW_1NN(){
        dtw=new DTW();
        accuracy=new ArrayList<>();
    }
    public FastDTW_1NN(DTW_DistanceBasic d){
        dtw=d;
        accuracy=new ArrayList<>();
    }
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.getBuildTime()+",CVAcc,"+res.getAcc()+",Memory,"+res.getMemory();
        result+=",BestWarpPercent,"+bestWarp+"AllAccs,";
       for(double d:accuracy)
            result+=","+d;
        return result;
    }
     
   
    
    public double getMaxR(){ return maxR;}
    public void setMaxPercentageWarp(int a){maxPercentageWarp=a;}
    public void optimiseWindow(boolean b){ optimiseWindow=b;}
    public void setR(double r){dtw.setR(r);}
    public double getR(){ return dtw.getR();}
    public int getBestWarp(){ return bestWarp;}
    public int getWindowSize(){ return dtw.getWindowSize(train.numAttributes()-1);}
    public void setMaxNoThreads(int maxNoThreads){this.maxNoThreads = maxNoThreads;}

    @Override
    public void buildClassifier(Instances d){
        res =new ClassifierResults();
        long t=System.currentTimeMillis();
        
        train=d;
        trainSize=d.numInstances();
        if(optimiseWindow){
            maxR=0;
            double maxAcc=0;
            int dataLength=train.numAttributes()-1;
/*  If the data length < 100 then there will be some repetition
            should skip some values I reckon
            if(dataLength<maxNosWindows)
                maxPercentageWarp=dataLength;
        */
            double previousPercentage=0;
            for(int i=maxPercentageWarp;i>=0;i-=1){
        //Set r for current value as the precentage of series length.
//                dtw=new DTW();
               
                int previousWindowSize=dtw.findWindowSize(previousPercentage,d.numAttributes()-1);
                int newWindowSize=dtw.findWindowSize(i/100.0,d.numAttributes()-1);
                if(previousWindowSize==newWindowSize)// no point doing this one
                    continue;
                previousWindowSize=newWindowSize;
                dtw.setR(i/100.0);
                        
                        
/*Can do an early abandon inside cross validate. If it cannot be more accurate 
 than maxR even with some left to evaluate then stop evaluation
*/                
                double acc=crossValidateAccuracy(maxAcc);
                accuracy.add(acc);
                if(acc>maxAcc){
                    maxR=i;
                    maxAcc=acc;
               }
//                System.out.println(" r="+i+" warpsize ="+x+" train acc= "+acc+" best acc ="+maxR);
/* Can ignore all window sizes bigger than the max used on the previous iteration
*/                
                
               if(maxWindowSize<(i-1)*dataLength/100){
                   System.out.println("WINDOW SIZE ="+dtw.getWindowSize()+" Can reset downwards at "+i+"% to ="+((int)(100*(maxWindowSize/(double)dataLength))));
                   i=(int)(100*(maxWindowSize/(double)dataLength));
                   i++;
//                   i=Math.round(100*(maxWindowSize/(double)dataLength))/100;
               } 

            }
            bestWarp=(int)(maxR*dataLength/100);
            System.out.println("OPTIMAL WINDOW ="+maxR+" % which gives a warp of"+bestWarp+" data");
  //          dtw=new DTW();
            dtw.setR(maxR/100.0);
            res.setAcc(maxAcc);
        }
        try {
            res.setBuildTime(System.currentTimeMillis()-t);
        } catch (Exception e) {
            System.err.println("Inheritance preventing me from throwing this error...");
            System.err.println(e);
        }
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        res.setMemory(usedBytes);
        
        if(trainPath!=null && trainPath!=""){  //Save basic train results
//            
//NEED TO FIND THE TRAIN ESTIMATES FOR EACH TEST HERE            
            OutFile f= new OutFile(trainPath);
            f.writeLine(train.relationName()+",FastDTW_1NN,Train");
            f.writeLine(getParameters());
            f.writeLine(res.getAcc()+"");
            for(int i=0;i<train.numInstances();i++){
                Instance test=train.remove(i);
                int pred=(int)classifyInstance(test);
                f.writeString((int)test.classValue()+","+pred+",");
                for(int j=0;j<train.numClasses();j++){
                    if(j==pred)
                        f.writeString(",1");
                    else
                        f.writeString(",0");
                }
                f.writeString("\n");
                train.add(i,test);
            }
        }        
        
    }
    @Override
    public double classifyInstance(Instance d){
        /** A large amount of this method has been changed for the project. The original logic is only used when both
         *  k and maxNoThreads are 1.
         */

        int index = 0;

        if (k == 1) {
            //if default params, run original method.
            if (maxNoThreads == 1) {
                /** Original Code **/
                double minSoFar = Double.MAX_VALUE;
                double dist;
                for (int i = 0; i < train.numInstances(); i++) {
                    dist = dtw.distance(train.instance(i), d, minSoFar);
                    if (dist < minSoFar) {
                        minSoFar = dist;
                        index = i;
                    }
                }
                /** End **/
            } else {
                ClassifyThread[] classifyThreads = new ClassifyThread[maxNoThreads];
                int intervalSize = train.numInstances() / maxNoThreads;
                InstanceDistance closestInstance = new InstanceDistance(Double.MAX_VALUE, -1);

                //Split up the data and start the threads.
                for (int t = 0; t < maxNoThreads; t++) {
                    if (t == maxNoThreads - 1) {
                        //Overspill at end due to integer division.
                        classifyThreads[t] = new ClassifyThread(d, t * intervalSize, train.numInstances(), closestInstance);
                    } else {
                        classifyThreads[t] = new ClassifyThread(d, t * intervalSize, (t + 1) * intervalSize, closestInstance);
                    }
                    classifyThreads[t].start();
                }

                //Wait for all threads to finish before getting the closest index.
                for (int t = 0; t < maxNoThreads; t++) {
                    try {
                        classifyThreads[t].join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                index = closestInstance.index;
            }
        }
        else {
            if (k < train.numInstances()) {
                k = train.numInstances();
            }

            //Predictions are stored in a sorted map, which sorts by distance.
            SortedMap<Double, Integer> predictions = Collections.synchronizedSortedMap(new TreeMap<Double, Integer>());
            predictions.put(Double.MAX_VALUE, -1);

            ClassifyThread[] classifyThreads = new ClassifyThread[maxNoThreads];
            int intervalSize = train.numInstances() / maxNoThreads;

            //Create threads as above and wait for them to finish.
            for (int t = 0; t < maxNoThreads; t++) {
                if (t == maxNoThreads - 1) {
                    classifyThreads[t] = new ClassifyThread(d, t * intervalSize, train.numInstances(), predictions);
                } else {
                    classifyThreads[t] = new ClassifyThread(d, t * intervalSize, (t + 1) * intervalSize, predictions);
                }
                classifyThreads[t].start();
            }

            for (int t = 0; t < maxNoThreads; t++) {
                try {
                    classifyThreads[t].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            //Calculate the frequency of each class value in the nearest neighbours.
            //<CV,count>
            HashMap<Double, Integer> counts = new HashMap<>();
            Collection<Integer> closestIndexes = predictions.values();
            Integer[] closestIndexesArr = new Integer[closestIndexes.size()];
            closestIndexesArr = closestIndexes.toArray(closestIndexesArr);
            for (int nk = 0; nk < k; nk++) {
                //Don't care about the last neighbour as we added max value element.
                if (nk == closestIndexesArr.length-1) {
                    break;
                }
                double cv = train.instance(closestIndexesArr[nk]).classValue();
                if (counts.containsKey(cv)) {
                    int count = counts.get(cv) + 1;
                    counts.replace(cv, count);
                }
                else {
                    counts.put(cv, 1);
                }
            }
            AtomicInteger maxCount = new AtomicInteger();
            AtomicReference<Double> maxCountCV = new AtomicReference<>();
            counts.forEach((k,v) -> {
                if (v > maxCount.get()) {
                    maxCount.set(v);
                    maxCountCV.set(k);
                }
            });
            //Returns the most voted class value.
            return maxCountCV.get();
        }

        return train.instance(index).classValue();
    }

    /**
     * Small class added to hold the distance of a neighbour for the threads.
     */
    private class InstanceDistance {
        private double distance;
        private int index;

        public InstanceDistance(double distance, int index) {
            this.distance = distance;
            this.index = index;
        }
    }

    /**
     * Thread class to perform the classification.
     */
    private class ClassifyThread extends Thread {
        private Instance d;
        private int start;
        private int end;
        //KNN
        private SortedMap<Double, Integer> predictions;
        //1NN
        private InstanceDistance closestInstance;

        public ClassifyThread(Instance d, int start, int end, InstanceDistance closestInstance) {
            this.d = d;
            this.start = start;
            this.end = end;
            this.closestInstance = closestInstance;
        }

        public ClassifyThread(Instance d, int start, int end, SortedMap<Double, Integer> predictions) {
            this.d = d;
            this.start = start;
            this.end = end;
            this.predictions = predictions;
        }

        public void run() {
            DTW_DistanceBasic temp = new DTW();
            try {
                temp.setOptions(dtw.getOptions());
            } catch (Exception e) {
                e.printStackTrace();
            }
            temp.setR(dtw.getR());

            for (int i = start; i < end; i++) {
                if (k == 1) {
                    //Calculate distance to neighbour and then update the nearest instance object if its closer.
                    double dist = temp.distance(train.instance(i), d, closestInstance.distance);
                    if (dist <= closestInstance.distance) {
                        synchronized (closestInstance) {
                            closestInstance.distance = dist;
                            closestInstance.index = i;
                        }
                    }
                }
                else {
                    //Calculate distance to neighbour and then add to map if its closer.
                    double dist = temp.distance(train.instance(i), d, predictions.firstKey());
                    if (dist <= predictions.firstKey()) {
                        synchronized (predictions) {
                            predictions.put(dist, i);
                        }
                    }
                }
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance){
        double[] dist=new double[instance.numClasses()];
        dist[(int)classifyInstance(instance)]=1;
        return dist;
    }


    /**Could do this by calculating the distance matrix, but then 	
 * you cannot use the early abandon. Early abandon about doubles the speed,
 * as will storing the distances. Given the extra n^2 memory, probably better
 * to just use the early abandon. We could store those that were not abandoned?
answer is to store those without the abandon in a hash table indexed by i and j,
*where index i,j == j,i

* @return 
 */
    private  double crossValidateAccuracy(double maxAcc){
        double a=0,d, minDist;
        int nearest;
        Instance inst;
        int bestNosCorrect=(int)(maxAcc*trainSize);
        maxWindowSize=0;
        int w;
        distances=new HashMap<>(trainSize);
        
        //Similar idea to above here.
        for(int i=0;i<trainSize;i++){
//Find nearest to element i
            nearest=0;
            minDist=Double.MAX_VALUE;
            inst=train.instance(i);
            for(int j=0;j<trainSize;j++){
                if(i!=j){
//  d=dtw.distance(inst,train.instance(j),minDist);
//Store past distances if not early abandoned 
//Not seen i,j before                    
                  if(j>i){
                        d=dtw.distance(inst,train.instance(j),minDist);
                        //Store if not early abandon
                        if(d!=Double.MAX_VALUE){
//                            System.out.println(" Storing distance "+i+" "+j+" d="+d+" with key "+(i*trainSize+j));
                            distances.put(i*trainSize+j,d);
//                            storeCount++;
                        }
//Else if stored recover                        
                    }else if(distances.containsKey(j*trainSize+i)){
                        d=distances.get(j*trainSize+i);
//                       System.out.println(" Recovering distance "+i+" "+j+" d="+d);
//                        recoverCount++;
                    }
//Else recalculate with new early abandon                    
                    else{
                        d=dtw.distance(inst,train.instance(j),minDist);
                    }        
                    if(d<minDist){
                        nearest=j;
                        minDist=d;
                        w=dtw.findMaxWindow();
                        if(w>maxWindowSize)
                            maxWindowSize=w;
                    }
                }
            }
                //Measure accuracy for nearest to element i			
            if(inst.classValue()==train.instance(nearest).classValue())
                a++;
           //Early abandon if it cannot be better than the best so far. 
            if(a+trainSize-i<bestNosCorrect){
//                    System.out.println(" Early abandon on CV when a="+a+" and i ="+i+" best nos correct = "+bestNosCorrect+" maxAcc ="+maxAcc+" train set size ="+trainSize);
                return 0.0;
            }
        }
//        System.out.println("trainSize ="+trainSize+" stored ="+storeCount+" recovered "+recoverCount);
        return a/(double)trainSize;
    }
    public static void main(String[] args){
            FastDTW_1NN c = new FastDTW_1NN();
            String path="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

            Instances test=DatasetLoading.loadDataNullable(path+"Coffee\\Coffee_TEST.arff");
            Instances train=DatasetLoading.loadDataNullable(path+"Coffee\\Coffee_TRAIN.arff");
            train.setClassIndex(train.numAttributes()-1);
            c.buildClassifier(train);

    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setParamSearch(boolean b) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setParametersFromIndex(int x) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }


}
