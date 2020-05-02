package timing;

import java.io.*;
import java.util.ArrayList;
import java.util.SortedSet;
import java.util.TreeSet;

public class DataReader {

    private ArrayList<DataFile> dataFiles;
    private SortedSet<String> datasets;
    private SortedSet<String> classifiers;
    private int numFolds;

    public static void main(String args[]) throws Exception {

        DataReader d = new DataReader();
//        d.loadSingleFold("results/20200305", NewRunner.DatasetType.TEST, 0);
//        d.writeFoldCSV(Stat.ACCURACY, "results/20200305", 0);
//        d.writeFoldCSV(Stat.AVG_CLASSIFY_TIME, "results/20200305", 0);
//        d.writeFoldCSV(Stat.TOTAL_CLASSIFY_TIME, "results/20200305", 0);
//        d.writeFoldCSV(Stat.TRAIN_TIME, "results/20200305", 0);

        d.loadMultiFolds("results/20200305", NewRunner.DatasetType.TEST, 5);
        d.writeFoldsAverageCSV(Stat.ACCURACY, "results/20200305", 5);
        d.writeFoldsAverageCSV(Stat.AVG_CLASSIFY_TIME, "results/20200305", 5);
        d.writeFoldsAverageCSV(Stat.TOTAL_CLASSIFY_TIME, "results/20200305", 5);
        d.writeFoldsAverageCSV(Stat.TRAIN_TIME, "results/20200305", 5);



        System.out.println("Done");

    }

    public DataReader() {
        dataFiles = new ArrayList<>();
        classifiers = new TreeSet<>();
        datasets = new TreeSet<>();
    }

    public enum Stat {
        ACCURACY,
        AVG_CLASSIFY_TIME,
        TOTAL_CLASSIFY_TIME,
        TRAIN_TIME
    }

    public void loadSingleFold(String resultsPath, NewRunner.DatasetType type, int foldNo) {
        File topDir = new File(resultsPath);
        String[] classifiers = topDir.list();

        for (int c = 0; c < classifiers.length; c++) {
            if (classifiers[c].startsWith(".")) {
                continue;
            }
            File classifierDir = new File(topDir.getPath() + "/" + classifiers[c] + "/Predictions");
            String[] datasets = classifierDir.list();

            for (int d = 0; d < datasets.length; d++) {
                if (datasets[d].startsWith(".")) {
                    continue;
                }

                try {
                    String typeStr = type.toString().toLowerCase();
                    File classifierResultsFile = new File(classifierDir.getPath() + "/" + datasets[d] + "/" + typeStr + "Fold" + foldNo + ".csv");
                    BufferedReader csvCR = new BufferedReader(new FileReader(classifierResultsFile));

                    String[] topLineCR = csvCR.readLine().split(",");
                    csvCR.readLine();
                    String[] accLineCR = csvCR.readLine().split(",");
                    double acc = Double.parseDouble(accLineCR[0]);

                    File timingResultsFile = new File(classifierDir.getPath() + "/" + datasets[d] + "/timing" + foldNo + ".csv");
                    BufferedReader csvTiming = new BufferedReader(new FileReader(timingResultsFile));
                    csvTiming.readLine();
                    String[] timingLine = csvTiming.readLine().split(",");
                    double avgClassifyTime = Double.parseDouble(timingLine[3]);
                    double totalClassifyTime = Double.parseDouble(timingLine[4]);
                    double trainTime = Double.parseDouble(timingLine[5]);

                    this.datasets.add(topLineCR[0]);
                    this.classifiers.add(topLineCR[1]);

                    dataFiles.add(new DataFile(topLineCR[1], topLineCR[0], acc, foldNo, avgClassifyTime, totalClassifyTime, trainTime));
                } catch (Exception e) {
                    System.out.println("****** RESULT MISSING ******");
                    e.printStackTrace();
                }
            }
        }

    }

    public void writeFoldCSV(Stat stat, String outputPath, int foldNo) {
        try {
            FileWriter matrix = new FileWriter(outputPath + "/fold" + foldNo + stat+ ".csv");

            //write classifiers
            for (String c : this.classifiers) {
                matrix.append("," + c);
            }
            matrix.append("\n");
            matrix.flush();

            //write datasets
            for (String dataset : this.datasets) {

                boolean found = false;

                matrix.append(dataset);

                SortedSet<DataFile> res = new TreeSet<>();
                for (DataFile d : this.dataFiles) {
                    if ((d.fold == foldNo) && (d.dataset.compareTo(dataset) == 0)) {
                        res.add(d);
                    }
                }


                int count = 0;
                Object[] classif = classifiers.toArray();
                for (DataFile d : res) {
                    Double value = -1.0;

                    while (d.classifier.compareTo((String) classif[count]) != 0) {
                        matrix.append(",-1");
                        count++;
                    }
                    if (d.classifier.compareTo((String) classif[count]) == 0) {
                        switch (stat) {
                            case ACCURACY:
                                value = d.accuracy;
                                break;
                            case AVG_CLASSIFY_TIME:
                                value = d.avgClassifyTime;
                                break;
                            case TOTAL_CLASSIFY_TIME:
                                value = d.totalClassifyTime;
                                break;
                            case TRAIN_TIME:
                                value = d.trainTime;
                                break;
                        }
                        count++;
                    }

                    matrix.append("," + Double.toString(value));

                }
                matrix.append("\n");
                matrix.flush();
            }

            matrix.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadMultiFolds(String resultsPath, NewRunner.DatasetType type, int numFolds) {
        for (int fold = 0; fold < numFolds; fold++) {
            loadSingleFold(resultsPath, type, fold);
        }
    }

    public void writeFoldsAverageCSV(Stat stat, String outputPath, int numFolds) throws Exception {
        ArrayList<DataPoint> points = new ArrayList<>();

        for (int df = 0; df < dataFiles.size(); df++) {
            DataFile toAdd = dataFiles.get(df);

            if (points.size() == 0) {
                DataPoint newDP = new DataPoint(toAdd.classifier, toAdd.dataset);
                newDP.dataFiles.add(toAdd);
                points.add(newDP);
            }
            else {
                boolean added = false;
                for (int dp = 0; dp < points.size(); dp++) {
                    if ((dataFiles.get(dp).dataset.compareTo(toAdd.dataset) == 0) && (dataFiles.get(dp).classifier.compareTo(toAdd.classifier) == 0)) {
                        points.get(dp).dataFiles.add(toAdd);
                        added = true;
                        break;
                    }
                }

                if (!added) {
                    DataPoint newDP = new DataPoint(toAdd.classifier, toAdd.dataset);
                    newDP.dataFiles.add(toAdd);
                    points.add(newDP);
                }
            }
        }

        try {
            FileWriter matrix = new FileWriter(outputPath + "/combined" + stat + ".csv");

            //write classifiers
            for (String c : this.classifiers) {
                matrix.append("," + c);
            }
            matrix.append("\n");
            matrix.flush();

            //write datasets
            for (String dataset : this.datasets) {

                boolean found = false;

                matrix.append(dataset);

                SortedSet<DataPoint> res = new TreeSet<>();
                for (DataPoint d : points) {
                    if (d.dataset.compareTo(dataset) == 0) {
                        res.add(d);
                    }
                }


                int count = 0;
                Object[] classif = classifiers.toArray();
                for (DataPoint d : res) {
                    Double value = -1.0;

                    while (d.classifier.compareTo((String) classif[count]) != 0) {
                        matrix.append(",-1");
                        count++;
                    }
                    if (d.classifier.compareTo((String) classif[count]) == 0) {
                        switch (stat) {
                            case ACCURACY:
                                value = d.averageAccuracy();
                                break;
                            case AVG_CLASSIFY_TIME:
                                value = d.averageAvgClassifyTime();
                                break;
                            case TOTAL_CLASSIFY_TIME:
                                value = d.averageTotalClassifyTime();
                                break;
                            case TRAIN_TIME:
                                value = d.averageTrainTime();
                                break;
                        }
                        count++;
                    }

                    matrix.append("," + Double.toString(value));

                }
                matrix.append("\n");
                matrix.flush();
            }

            matrix.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public class DataFile implements Comparable {
        String classifier;
        String dataset;
        double accuracy;
        int fold;
        double avgClassifyTime;
        double totalClassifyTime;
        double trainTime;

        public DataFile(String classifier, String dataset, double accuracy, int fold, double avgClassifyTime, double totalClassifyTime, double trainTime) {
            this.classifier = classifier;
            this.dataset = dataset;
            this.accuracy = accuracy;
            this.fold = fold;
            this.avgClassifyTime = avgClassifyTime;
            this.totalClassifyTime = totalClassifyTime;
            this.trainTime = trainTime;
        }

        @Override
        public int compareTo(Object o) {
            return this.classifier.compareTo(((DataFile) o).classifier);
        }
    }

    public class DataPoint implements Comparable{
        String classifier;
        String dataset;
        ArrayList<DataFile> dataFiles;

        public DataPoint(String classifier, String dataset) {
            this.classifier = classifier;
            this.dataset = dataset;
            this.dataFiles = new ArrayList<>();
        }

        public int numFolds() {
            return dataFiles.size();
        }

        public double averageAccuracy() throws Exception {
            double total = 0;
            for (DataFile d : dataFiles) {
                if (d.accuracy == 0.0) {
                    //throw new Exception("Accuracy is 0.");
                } else {
                    total += d.accuracy;
                }
            }
            return total / dataFiles.size();
        }

        public double averageAvgClassifyTime() throws Exception {
            double total = 0;
            for (DataFile d : dataFiles) {
                if (d.avgClassifyTime == 0.0) {
                    throw new Exception("Time is 0.");
                } else {
                    total += d.avgClassifyTime;
                }
            }
            return total / dataFiles.size();
        }

        public double averageTotalClassifyTime() throws Exception {
            double total = 0;
            for (DataFile d : dataFiles) {
                if (d.totalClassifyTime == 0.0) {
                    throw new Exception("Time is 0.");
                } else {
                    total += d.totalClassifyTime;
                }
            }
            return total / dataFiles.size();
        }

        public double averageTrainTime() throws Exception {
            double total = 0;
            for (DataFile d : dataFiles) {
                if (d.trainTime == 0.0) {
                    throw new Exception("Time is 0.");
                } else {
                    total += d.trainTime;
                }
            }
            return total / dataFiles.size();
        }

        @Override
        public int compareTo(Object o) {
            if (classifier.compareTo(((DataPoint) o).classifier) == 0) {
                return dataset.compareTo(((DataPoint) o).dataset);
            }
            return classifier.compareTo(((DataPoint) o).classifier);
        }
    }
}

//*****************************************************************
//
//      Data output format (CSV)
//
//                  ClassiferA      ClassiferB      ClassifierC
//      DatasetA    0.78            0.57            0.25
//      DatasetB    0.45            0.72            0.65
//      DatasetC    0.69            0.95            0.58
//
//*****************************************************************

