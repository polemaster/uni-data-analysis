import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class Part2 extends Configured implements Tool {

    public static class ComboMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text combo = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",", -1);
            if (fields.length > 20) {
                String visitor = fields[0] + " " + fields[1] + " " + fields[2];
                String visitee = fields[19] + " " + fields[20];
                combo.set(visitor + "; " + visitee);
                context.write(combo, one);
            }
        }
    }

    public static class TopReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private TreeMap<Integer, List<Text>> topMap = new TreeMap<>(Collections.reverseOrder());
        private static final int TOP_N = 100;

        public void reduce(Text key, Iterable<IntWritable> values, Context context) {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            // Correctly accumulate the full total per key
            synchronized (topMap) {
                topMap.computeIfAbsent(sum, k -> new ArrayList<>()).add(new Text(key));

                int total = topMap.values().stream().mapToInt(List::size).sum();
                if (total > TOP_N) {
                    Integer lastKey = topMap.lastKey();
                    List<Text> list = topMap.get(lastKey);
                    list.remove(list.size() - 1);
                    if (list.isEmpty()) {
                        topMap.remove(lastKey);
                    }
                }
            }
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<Integer, List<Text>> entry : topMap.entrySet()) {
                for (Text t : entry.getValue()) {
                    context.write(t, new IntWritable(entry.getKey()));
                }
            }
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Most Frequent Visitor-Visitee Combos");
        job.setJarByClass(Part2.class);
        job.setMapperClass(ComboMapper.class);
        job.setReducerClass(TopReducer.class);
        job.setNumReduceTasks(1);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new Part2(), args);
        System.exit(res);
    }
}
