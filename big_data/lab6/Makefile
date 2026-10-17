HADOOP_CLASSPATH := ${JAVA_HOME}/lib/tools.jar
export HADOOP_CLASSPATH

INPUT=/user/adampap/stackOverflow
HDFS=user/bigdata06/stackoverflow
OBJDIR=build
JAR=StackOverflow.jar

TARGETS := run5 clean

.PHONY: all $(TARGETS)

all: $(TARGETS)

# Main run target
run5: $(OBJDIR)/StackOverflowAnalysis.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/output/
	hadoop jar $(JAR) StackOverflowAnalysis $(INPUT) /$(HDFS)/output
	@echo "hdfs dfs -cat /$(HDFS)/output/part*"

# Compile rule
$(OBJDIR)/%.class: %.java | $(OBJDIR)
	hadoop com.sun.tools.javac.Main *.java -d $(OBJDIR)

# Create build directory if missing
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Cleanup target
clean:
	rm -f $(OBJDIR)/* $(JAR)
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/output/
