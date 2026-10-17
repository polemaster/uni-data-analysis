import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.Random;

public class FilterSample extends Configured implements Tool {

    public static class FilterSampleMapper extends Mapper<Object, Text, NullWritable, Text> {
        private float percentage = 1.0f;
        private Random rand = new Random();

        @Override
        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            percentage = conf.getFloat("sample.percent", 1.0f);
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",", -1);
            if (fields.length > 11 && !fields[6].trim().isEmpty() && !fields[11].trim().isEmpty()) {
                if (rand.nextFloat() < percentage) {
                    context.write(NullWritable.get(), value);
                }
            }
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "WhiteHouse Filter and Sample");

        job.setJarByClass(FilterSample.class);
        job.setMapperClass(FilterSampleMapper.class);
        job.setNumReduceTasks(0);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new FilterSample(), args);
        System.exit(res);
    }
}
