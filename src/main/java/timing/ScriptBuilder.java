/**
 * Class to create scripts for HPC job submission and various bash scripts.
 */
package timing;

import experiments.data.DatasetLists;

import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

public class ScriptBuilder {

    public static void main(String args[]) throws Exception{
        Date d = new Date();
        SimpleDateFormat df = new SimpleDateFormat("yyyyMMdd");
        String identifier = df.format(d);

        File dataDir = new File("data/Univariate_arff");
        String[] datasets = DatasetLists.tscProblems85;

//        for (String dataset : datasets) {
//            for (int classifier = 0; classifier < 25; classifier++) {
//                writeBsubTimingScript(5, dataset, classifier, identifier);
//            }
//        }

//        for (String dataset : datasets) {
//            writeBsubReducedDataScript("/gpfs/home/sjk17evu/Univariate_arff/" + dataset);
//        }

        writePiReducedDataScript(identifier);
//        writeBashScript();

    }

    public static void writeBsubReducedDataScript(String datasetPath) throws Exception{
        String[] pathSplit = datasetPath.split("/");

        FileWriter fw = new FileWriter("scripts/" + pathSplit[pathSplit.length-1] + "-Reduced.bsub");
        fw.append("#!/bin/csh\n");
        fw.append("#BSUB -q long-eth\n");
        fw.append("#BSUB -J " + pathSplit[pathSplit.length-1] + "-Reduced\n");
        fw.append("#BSUB -oo output/output_%I.txt\n");
        fw.append("#BSUB -eo error/error_%I.txt\n");
        fw.append("#BSUB -R \"rusage[mem=6000]\"\n");
        fw.append("#BSUB -M 10000\n");
        fw.append("module add java/jdk1.8.0_51\n");
        fw.append("java -jar -Xmx10000m ReducedExperiment.jar " + datasetPath + " /gpfs/scratch/sjk17evu/results/reducedData \n");
        fw.flush();
    }

    public static void writeBsubTimingScript(int numResamples, String dataset, int classifier, String id) throws Exception{
        FileWriter fw = new FileWriter("scripts/" + dataset + "_" + classifier + ".bsub");
        fw.append("#!/bin/csh\n");
        fw.append("#BSUB -q long-eth\n");
        fw.append("#BSUB -J " + dataset + "_" + classifier + "[1-" + (numResamples+1) + "]\n");
        fw.append("#BSUB -oo output/output_%I.txt\n");
        fw.append("#BSUB -eo error/error_%I.txt\n");
        fw.append("#BSUB -R \"rusage[mem=6000]\"\n");
        fw.append("#BSUB -M 10000\n");
        fw.append("module add java/jdk1.8.0_51\n");
        fw.append("java -jar -Xmx10000m TimingExperiment.jar " + id + " $LSB_JOBINDEX " + classifier + " /gpfs/home/sjk17evu/Univariate_arff/" + dataset + " /gpfs/scratch/sjk17evu/results $LSB_JOBINDEX true\n");
        fw.flush();
    }

    public static void writeBashScript() throws Exception{
        File dataDir = new File("scripts");
        String[] bsubs = dataDir.list();

        Arrays.sort(bsubs);
        FileWriter fw = new FileWriter("bash/script.sh");
        fw.append("#!/bin/bash\n");

        for (String sub : bsubs) {
            fw.append("bsub < scripts/" + sub + "\n");
        }

        fw.flush();
    }

    public static void writePiReducedDataScript(String id) throws Exception {
        FileWriter fw = new FileWriter("bash/piReduced.sh");
        fw.append("#!/bin/csh\n");

        String[] datasets = DatasetLists.tscProblems85;

        for (String dataset : datasets) {
            for (int cores = 1; cores <= 4; cores++) {
                fw.append("java -jar -Xmx1500m TimingExperiment.jar redData" + id + " 1 4 " + " data/Univariate_arff/" + dataset + " results/" + cores +"/ " + cores +"\n");
            }
        }
        fw.flush();
    }
}
