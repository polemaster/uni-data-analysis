HADOOP_CLASSPATH := ${JAVA_HOME}/lib/tools.jar
export HADOOP_CLASSPATH

INPUT=/user/adampap/WHVsmall
HDFS=user/bigdata06/whitehouse
OBJDIR=build
JAR=WhiteHouse.jar

TARGETS := run1 run2 run3 run4

.PHONY: final $(TARGETS) clean

final: $(TARGETS)

run1: $(OBJDIR)/FilterSample.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/1-output/
	hadoop jar $(JAR) FilterSample -D sample.percent=0.1 $(INPUT) /$(HDFS)/1-output
	@echo "hdfs dfs -cat /$(HDFS)/1-output/part*"

run2: $(OBJDIR)/Part2.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/2-output/
	hadoop jar $(JAR) Part2 $(INPUT) /$(HDFS)/2-output
	@echo "hdfs dfs -cat /$(HDFS)/2-output/part*"

run3: $(OBJDIR)/Part3.class $(OBJDIR)/DateStatsWritable.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/3-output/
	hadoop jar $(JAR) Part3 $(INPUT) /$(HDFS)/3-output
	@echo "hdfs dfs -cat /$(HDFS)/3-output/part*"

run4: $(OBJDIR)/Part4.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/4-output/
	hadoop jar $(JAR) Part4 $(INPUT) /$(HDFS)/4-output
	@echo "hdfs dfs -cat /$(HDFS)/4-output/part*"


$(OBJDIR)/%.class: %.java | $(OBJDIR)
	hadoop com.sun.tools.javac.Main *.java -d $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)/* $(JAR)
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/*-output/

