export HADOOP_EXE='/usr/bin/hadoop'
export HADOOP_LIBPATH='/opt/cloudera/parcels/CDH/lib'
export HADOOP_STREAMING='hadoop-mapreduce/hadoop-streaming.jar'
# export PYSPARK_SUBMIT_ARGS="--master local[3]"

alias hfs="$HADOOP_EXE fs"
alias hjs="$HADOOP_EXE jar $HADOOP_LIBPATH/$HADOOP_STREAMING"

module load python/gcc/3.7.9
