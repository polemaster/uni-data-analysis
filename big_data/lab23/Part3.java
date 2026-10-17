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
import java.util.Date;
import java.util.StringTokenizer;
import java.util.concurrent.TimeUnit;



public class Part3 extends Configured implements Tool {

    public static class VisitorMapper extends Mapper<Object, Text, Text, DateStatsWritable> {
        private SimpleDateFormat fmt = new SimpleDateFormat("MM/dd/yyyy HH:mm");

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",", -1);
            if (fields.length > 11 && !fields[11].trim().isEmpty()) {
                String name = fields[0] + " " + fields[1] + " " + fields[2];
                context.write(new Text(name), new DateStatsWritable(fields[11], fields[11], 1));
            }
        }
    }

    public static class VisitorReducer extends Reducer<Text, DateStatsWritable, Text, Text> {
        public void reduce(Text key, Iterable<DateStatsWritable> values, Context context) throws IOException, InterruptedException {
            DateStatsWritable combined = null;
            for (DateStatsWritable val : values) {
                if (combined == null) {
                    combined = new DateStatsWritable(val.getMinDate(), val.getMaxDate(), val.getCount());
                } else {
                    combined.merge(val);
                }
            }
            context.write(key, new Text(combined.toString()));
        }
    }


    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Visitor Appointment Stats");

        job.setJarByClass(Part3.class);
        job.setMapperClass(VisitorMapper.class);
        job.setReducerClass(VisitorReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DateStatsWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new Part3(), args);
        System.exit(res);
    }
}

