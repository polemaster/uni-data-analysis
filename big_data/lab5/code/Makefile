HADOOP_CLASSPATH := ${JAVA_HOME}/lib/tools.jar
export HADOOP_CLASSPATH

OBJDIR=build
JAR=PageRank.jar
CLASS=PageRankDriver

N=4
K=10
B=0.8

INPUT_LINKS=/user/adampap/wikiSmall/links.txt
INPUT_DEADENDS=/user/adampap/wikiDeadEnds/deadEnds.txt
INPUT_TRAPS=/user/adampap/wikiSpiderTraps/spiderTraps.txt

HDFS_BASE=/user/bigdata06/pagerank-output

TARGETS := run-links run-deadends run-spidertraps clean

.PHONY: all $(TARGETS)

all: $(TARGETS)

# Compile all Java files to OBJDIR
build:
	mkdir -p $(OBJDIR)
	hadoop com.sun.tools.javac.Main *.java -d $(OBJDIR)
	jar -cvf $(JAR) -C $(OBJDIR)/ ./

# Run PageRank on links.txt
run-links: build
	hdfs dfs -rm -r -skipTrash -f $(HDFS_BASE)/links
	hadoop jar $(JAR) $(CLASS) $(INPUT_LINKS) $(HDFS_BASE)/links $(N) $(K) $(B)
	@echo "hdfs dfs -cat $(HDFS_BASE)/links/iter$(K)/part-*"

# Run PageRank on deadEnds.txt
run-deadends: build
	hdfs dfs -rm -r -skipTrash -f $(HDFS_BASE)/deadends
	hadoop jar $(JAR) $(CLASS) $(INPUT_DEADENDS) $(HDFS_BASE)/deadends $(N) $(K) $(B)
	@echo "hdfs dfs -cat $(HDFS_BASE)/deadends/iter$(K)/part-*"

# Run PageRank on spiderTraps.txt
run-spidertraps: build
	hdfs dfs -rm -r -skipTrash -f $(HDFS_BASE)/spidertraps
	hadoop jar $(JAR) $(CLASS) $(INPUT_TRAPS) $(HDFS_BASE)/spidertraps $(N) $(K) $(B)
	@echo "hdfs dfs -cat $(HDFS_BASE)/spidertraps/iter$(K)/part-*"

# Cleanup
clean:
	rm -rf $(OBJDIR) $(JAR)
	hdfs dfs -rm -r -skipTrash -f $(HDFS_BASE)/links
	hdfs dfs -rm -r -skipTrash -f $(HDFS_BASE)/deadends
	hdfs dfs -rm -r -skipTrash -f $(HDFS_BASE)/spidertraps

