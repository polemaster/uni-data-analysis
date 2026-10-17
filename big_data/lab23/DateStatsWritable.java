
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



public class DateStatsWritable implements Writable {
    private String minDate = "";
    private String maxDate = "";
    private int count = 0;

    public DateStatsWritable() {}

    public DateStatsWritable(String min, String max, int count) {
        this.minDate = min;
        this.maxDate = max;
        this.count = count;
    }

    public void write(DataOutput out) throws IOException {
        out.writeUTF(minDate);
        out.writeUTF(maxDate);
        out.writeInt(count);
    }

    public void readFields(DataInput in) throws IOException {
        minDate = in.readUTF();
        maxDate = in.readUTF();
        count = in.readInt();
    }

    public String getMinDate() { return minDate; }
    public String getMaxDate() { return maxDate; }
    public int getCount() { return count; }

    public void merge(DateStatsWritable other) {
        try {
            SimpleDateFormat fmt = new SimpleDateFormat("MM/dd/yyyy HH:mm");
            Date minThis = fmt.parse(this.minDate);
            Date minOther = fmt.parse(other.minDate);
            Date maxThis = fmt.parse(this.maxDate);
            Date maxOther = fmt.parse(other.maxDate);

            if (minOther.before(minThis)) this.minDate = other.minDate;
            if (maxOther.after(maxThis)) this.maxDate = other.maxDate;
            this.count += other.count;
        } catch (ParseException e) {
            // Ignore
        }
    }

    public String toString() {
        return minDate + "\t" + maxDate + "\t" + count;
    }
}
