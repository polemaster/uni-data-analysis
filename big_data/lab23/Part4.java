
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
import java.util.concurrent.TimeUnit;



public class Part4 extends Configured implements Tool {

    public static class HourStatsWritable implements Writable {
        public int planned = 0;
        public int delayed = 0;
        public long totalDelay = 0;
        public int early = 0;
        public long totalEarly = 0;
        public long mae = 0;

        public void write(DataOutput out) throws IOException {
            out.writeInt(planned);
            out.writeInt(delayed);
            out.writeLong(totalDelay);
            out.writeInt(early);
            out.writeLong(totalEarly);
            out.writeLong(mae);
        }

        public void readFields(DataInput in) throws IOException {
            planned = in.readInt();
            delayed = in.readInt();
            totalDelay = in.readLong();
            early = in.readInt();
            totalEarly = in.readLong();
            mae = in.readLong();
        }

        public void combine(HourStatsWritable other) {
            this.planned += other.planned;
            this.delayed += other.delayed;
            this.totalDelay += other.totalDelay;
            this.early += other.early;
            this.totalEarly += other.totalEarly;
            this.mae += other.mae;
        }

        public String toString() {
            double avgDelay = (delayed == 0) ? 0 : (double) totalDelay / delayed;
            double avgEarly = (early == 0) ? 0 : (double) totalEarly / early;
            double avgMae = (planned == 0) ? 0 : (double) mae / planned;

            return planned + "\t" +
                   delayed + "\t" + String.format(Locale.US, "-%.1f", avgDelay).replace('.', ',') + "\t" +
                   early + "\t" + String.format(Locale.US, "%.1f", avgEarly).replace('.', ',') + "\t" +
                   String.format(Locale.US, "%.1f", avgMae).replace('.', ',');
        }
    }

    public static class HourMapper extends Mapper<Object, Text, IntWritable, HourStatsWritable> {
        private SimpleDateFormat fmt = new SimpleDateFormat("MM/dd/yyyy HH:mm");
        private IntWritable hourKey = new IntWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",", -1);
            if (fields.length > 11 && !fields[6].trim().isEmpty() && !fields[11].trim().isEmpty()) {
                try {
                    Date arrival = fmt.parse(fields[6]);
                    Date appt = fmt.parse(fields[11]);
                    long diff = Math.abs(arrival.getTime() - appt.getTime());
                    if (diff > TimeUnit.HOURS.toMillis(24)) return;

                    int hour = Integer.parseInt(new SimpleDateFormat("HH").format(appt));
                    hourKey.set(hour);

                    HourStatsWritable stat = new HourStatsWritable();
                    stat.planned = 1;
                    long minutesDiff = TimeUnit.MILLISECONDS.toMinutes(arrival.getTime() - appt.getTime());

                    stat.mae = Math.abs(minutesDiff);
                    if (minutesDiff > 0) {
                        stat.delayed = 1;
                        stat.totalDelay = minutesDiff;
                    } else {
                        stat.early = 1;
                        stat.totalEarly = -minutesDiff;
                    }
                    context.write(hourKey, stat);

                } catch (ParseException e) {
                    // ignore
                }
            }
        }
    }

    public static class HourReducer extends Reducer<IntWritable, HourStatsWritable, IntWritable, HourStatsWritable> {
        public void reduce(IntWritable key, Iterable<HourStatsWritable> values, Context context) throws IOException, InterruptedException {
            HourStatsWritable combined = new HourStatsWritable();
            for (HourStatsWritable val : values) {
                combined.combine(val);
            }
            context.write(key, combined);
        }
    }

    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(getConf(), "Hourly Appointment Analysis");
        job.setJarByClass(Part4.class);
        job.setMapperClass(HourMapper.class);
        job.setCombinerClass(HourReducer.class);
        job.setReducerClass(HourReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(HourStatsWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new Part4(), args);
        System.exit(res);
    }
}
