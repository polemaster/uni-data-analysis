import java.io.IOException;
import java.util.ArrayList;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class PageRankReducer extends Reducer<Text, Text, Text, Text> {
    private int numNodes;
    private double taxRate;
    private double danglingMass = 0.0;

    @Override
    protected void setup(Context context) {
        numNodes = context.getConfiguration().getInt("numNodes", 1);
        taxRate = context.getConfiguration().getDouble("taxRate", 0.8);
        danglingMass = context.getConfiguration().getDouble("danglingMass", 0.0);
    }

    @Override
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        ArrayList<String> outLinks = new ArrayList<>();
        double rankSum = 0.0;

        for (Text val : values) {
            String[] tokens = val.toString().split(" ", 2);
            if (tokens[0].equals("PR")) {
                rankSum += Double.parseDouble(tokens[1]);
            } else if (tokens[0].equals("LINKS")) {
                if (tokens.length > 1) {
                    for (String out : tokens[1].split(" ")) {
                        outLinks.add(out);
                    }
                }
            }
        }

        double redistributed = danglingMass / numNodes;
        double newRank = (1 - taxRate) / numNodes + taxRate * (rankSum + redistributed);

        StringBuilder sb = new StringBuilder();
        sb.append(newRank);
        for (String out : outLinks) {
            sb.append(" ").append(out);
        }

        context.write(key, new Text(sb.toString()));
    }
}

