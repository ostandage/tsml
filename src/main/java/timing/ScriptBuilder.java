package timing;

import sun.java2d.pipe.SpanShapeRenderer;

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
        String[] datasets = dataDir.list();

        for (String dataset : datasets) {
            for (int classifier = 0; classifier < 24; classifier++) {
                writeBsubScript(5, dataset, classifier, identifier);
            }
        }

        writeBashScript();

    }

    public static void writeBsubScript(int numResamples, String dataset, int classifier, String id) throws Exception{
        FileWriter fw = new FileWriter("scripts/" + dataset + "_" + classifier + ".bsub");
        fw.append("#!/bin/csh\n");
        fw.append("#BSUB -q long-eth\n");
        fw.append("#BSUB -J " + dataset + "_" + classifier + "[1-" + (numResamples + 1) + "]\n");
        fw.append("#BSUB -oo output/output_%I.txt\n");
        fw.append("#BSUB -eo error/error_%I.txt\n");
        fw.append("#BSUB -R \"rusage[mem=6000]\"\n");
        fw.append("#BSUB -M 6000\n");
        fw.append("module add java/jdk1.8.0_51\n");
        fw.append("java -jar -Xmx6000m TimingExperiment.jar " + id + " $LSB_JOBINDEX " + classifier + " /gpfs/home/sjk17evu/Univariate_arff/" + dataset + " /gpfs/scratch/sjk17evu/results $LSB_JOBINDEX true\n");
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
}
