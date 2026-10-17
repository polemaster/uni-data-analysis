import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

public class StackOverflowAnalysis extends Configured implements Tool {

    public static class PostWritable implements Writable {
        public String type;
        public String id;
        public String parentId;
        public String acceptedAnswerId;
        public int score;
        public String tag;

        public PostWritable() {}

        public PostWritable(String type, String id, String parentId, String acceptedAnswerId, int score, String tag) {
            this.type = type;
            this.id = id;
            this.parentId = parentId;
            this.acceptedAnswerId = acceptedAnswerId;
            this.score = score;
            this.tag = tag;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeUTF(type);
            out.writeUTF(id);
            out.writeUTF(parentId);
            out.writeUTF(acceptedAnswerId);
            out.writeInt(score);
            out.writeUTF(tag);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            type = in.readUTF();
            id = in.readUTF();
            parentId = in.readUTF();
            acceptedAnswerId = in.readUTF();
            score = in.readInt();
            tag = in.readUTF();
        }
    }

    public static class StackOverflowMapper extends Mapper<Object, Text, Text, PostWritable> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",", -1);
            if (fields.length < 6) return;

            String type = fields[0];
            String id = fields[1];
            String parentId = fields[3];
            String acceptedAnswerId = fields[2];
            int score = 0;
            try {
                score = Integer.parseInt(fields[4]);
            } catch (NumberFormatException e) {
                // default to 0
            }
            String tag = fields[5];

            if (type.equals("1") && !tag.isEmpty()) {
                context.write(new Text(id), new PostWritable("question", id, "", acceptedAnswerId, 0, tag));
            } else if (type.equals("2")) {
                context.write(new Text(parentId), new PostWritable("answer", id, parentId, "", score, ""));
            }
        }
    }

    public static class StackOverflowReducer extends Reducer<Text, PostWritable, Text, Text> {
        private Map<String, TagStats> tagStatsMap = new HashMap<>();

        public static class TagStats {
            int solvedCount = 0;
            int unsolvedCount = 0;
            int solvedAnswerTotal = 0;
            int unsolvedAnswerTotal = 0;
        }

        public void reduce(Text key, Iterable<PostWritable> values, Context context) {
            String tag = null;
            String acceptedAnswerId = null;
            List<PostWritable> answers = new ArrayList<>();

            for (PostWritable pw : values) {
                if (pw.type.equals("question")) {
                    tag = pw.tag;
                    acceptedAnswerId = pw.acceptedAnswerId;
                } else if (pw.type.equals("answer")) {
                    answers.add(new PostWritable(pw.type, pw.id, pw.parentId, "", pw.score, ""));
                }
            }

            if (tag == null) return;

            boolean isSolved = acceptedAnswerId != null && !acceptedAnswerId.isEmpty();
            TagStats stats = tagStatsMap.getOrDefault(tag, new TagStats());

            if (isSolved) {
                stats.solvedCount++;
                stats.solvedAnswerTotal += answers.size();
            } else {
                stats.unsolvedCount++;
                stats.unsolvedAnswerTotal += answers.size();
            }

            tagStatsMap.put(tag, stats);
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            List<Map.Entry<String, TagStats>> entries = new ArrayList<>(tagStatsMap.entrySet());
            entries.sort((a, b) -> Integer.compare(b.getValue().solvedCount, a.getValue().solvedCount));

            for (Map.Entry<String, TagStats> entry : entries) {
                String tag = entry.getKey();
                TagStats s = entry.getValue();

                String avgSolved = s.solvedCount > 0
                        ? String.format("%.1f", (double) s.solvedAnswerTotal / s.solvedCount)
                        : "NaN";
                String avgUnsolved = s.unsolvedCount > 0
                        ? String.format("%.1f", (double) s.unsolvedAnswerTotal / s.unsolvedCount)
                        : "NaN";

                String result = String.format("%d\t%s\t|\t%d\t%s",
                        s.solvedCount, avgSolved, s.unsolvedCount, avgUnsolved);

                context.write(new Text(tag), new Text(result));
            }
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Stack Overflow Analysis");
        job.setJarByClass(StackOverflowAnalysis.class);
        job.setMapperClass(StackOverflowMapper.class);
        job.setReducerClass(StackOverflowReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(PostWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new StackOverflowAnalysis(), args);
        System.exit(res);
    }
}

