import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class PageRankMapper extends Mapper<Object, Text, Text, Text> {

    public static enum PR_COUNTER {
        DANGLING_MASS
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        if (value == null || value.toString().trim().isEmpty()) return;

        String[] parts = value.toString().trim().split("\\s+");
        if (parts.length < 1) return;

        String page = parts[0];
        double pr = 1.0;
        int linkStart = 1;

        // Try to parse PageRank value
        if (parts.length >= 2) {
            try {
                pr = Double.parseDouble(parts[1]);
                linkStart = 2;
            } catch (NumberFormatException e) {
                // It's first iteration, no PageRank field — links start at index 1
                linkStart = 1;
            }
        }

        // If no outlinks (dangling node)
        if (parts.length <= linkStart) {
            long scaledPr = (long)(pr * 1_000_000_000L);
            context.getCounter(PR_COUNTER.DANGLING_MASS).increment(scaledPr);
            context.write(new Text(page), new Text("LINKS"));
            return;
        }

        int outCount = parts.length - linkStart;

        for (int i = linkStart; i < parts.length; i++) {
            context.write(new Text(parts[i]), new Text("PR " + (pr / outCount)));
        }

        // Emit graph structure
        StringBuilder links = new StringBuilder();
        for (int i = linkStart; i < parts.length; i++) {
            links.append(parts[i]).append(" ");
        }
        context.write(new Text(page), new Text("LINKS " + links.toString().trim()));
    }
}

