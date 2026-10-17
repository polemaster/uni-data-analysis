import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PageRankDriver {
    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: PageRankDriver <input> <output> <N> <K> <B>");
            System.exit(-1);
        }

        String inputPath = args[0];
        String outputBase = args[1];
        int numNodes = Integer.parseInt(args[2]);
        int numIters = Integer.parseInt(args[3]);
        double B = Double.parseDouble(args[4]);

        String lastOutput = inputPath;
        double danglingMass = 0.0;

        for (int i = 1; i <= numIters; i++) {
            Configuration conf = new Configuration();
            conf.setInt("numNodes", numNodes);
            conf.setDouble("taxRate", B);
            conf.setDouble("danglingMass", danglingMass);

            Job job = Job.getInstance(conf, "PageRank Iteration " + i);
            job.setJarByClass(PageRankDriver.class);

            job.setMapperClass(PageRankMapper.class);
            job.setReducerClass(PageRankReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(lastOutput));
            String thisOutput = outputBase + "/iter" + i;
            FileOutputFormat.setOutputPath(job, new Path(thisOutput));

            boolean success = job.waitForCompletion(true);
            if (!success) {
                System.err.println("Job failed");
                System.exit(1);
            }

            // Get dangling mass from counter
            Counter counter = job.getCounters().findCounter(PageRankMapper.PR_COUNTER.DANGLING_MASS);
            danglingMass = counter.getValue() / 1_000_000_000.0;
            lastOutput = thisOutput;
        }
    }
}

