package ai.skymind.apps

import java.io.File
import java.util
import java.util.Collections

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.{RDDTrainingApproach, Repartition, RepartitionStrategy}
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.{FileStatsStorage, InMemoryStatsStorage}
import org.deeplearning4j.api.storage.StatsStorageRouter
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.{DataSet, ExistingMiniBatchDataSetIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

object LstmSparkApp {

  def main(args: Array[String]): Unit = {
    val numClasses:Int      = 3
    val saveUpdater:Boolean = true
    val exportBasePath:File = new File(System.getProperty("user.home"), "data/nasa-turbofan/exports/ckpts")
    val modelPath:File = new File(exportBasePath.toString, "models")
    val modelFile:File = new File(modelPath, "dl4j_lstm.h5")

    /** Restore Network from Disk for Inference using ParallelInference */
    val net:ComputationGraph = ModelSerializer.restoreComputationGraph(modelFile.toString(), saveUpdater)
    println(net.summary())

    /** Define Spark Network + Training Master **/
    val confSpark:SparkConf = new SparkConf()
    confSpark.set("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
    confSpark.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    confSpark.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")
    confSpark.set("spark.locality.wait", "0")                         //Transfer Immediate to Executor rather than wait
    confSpark.setAppName("LSTM Spark")
    confSpark.setMaster("local[*]")
    val spark = SparkSession.builder().config(confSpark).getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    println("Spark Version: %s".format(spark.version))

    val dataSetObjectSize:Int  = 16   //number of examples in each DataSet (strings typically have size 1)
    val batchSizePerWorker:Int = 16
    val tmpDir:String = new File(exportBasePath, "spark").toString
    val tm = new ParameterAveragingTrainingMaster.Builder(dataSetObjectSize)
      .batchSizePerWorker(batchSizePerWorker)
      .workerPrefetchNumBatches(2)
      .saveUpdater(true)                                            //good default
      .averagingFrequency(5)                                        //save internal history state per worker
      .rddTrainingApproach(RDDTrainingApproach.Export)              //5-10 is good default
      .exportDirectory(tmpDir)                                      //for use with Export Approach
      .storageLevel(StorageLevel.MEMORY_ONLY_SER)                   //better performance for large DataSets
      .storageLevelStreams(StorageLevel.MEMORY_ONLY)                //for use with DIRECT Approach
      .repartionData(Repartition.Always)                            //for use with DIRECT, fitPaths(RDD<String>)
      .repartitionStrategy(RepartitionStrategy.ApproximateBalanced) //default for avoiding data skew
      .build()

    val sparkNet = new SparkComputationGraph(sc, net.getConfiguration(), tm)

    /** Add Listeners to Network **/
    /** Add Monitoring to the Network, further derive training Stats **/
    val listenerFreq = 1
    val fileStorage:StatsStorage = new FileStatsStorage(new File("net_monitoring.dl4j"))
    val uiServer:UIServer = UIServer.getInstance()
    uiServer.enableRemoteListener()                                 //for monitoring on separate JVM
    sparkNet.setListeners(fileStorage, Collections.singletonList(new StatsListener(null)))
    //uiServer.attach(fileStorage)                                  //After Completed Fitting for FileStorage
    sparkNet.setCollectTrainingStats(true)


    /** Load Existing DataSet Iterators **/
    val pattern: String = "dataset_iter_%d.bin"
    val exportIterPath:File = new File(exportBasePath, "iterators")
    val numEpochs:Int = 10
    for(epoch<- 1 to numEpochs) {
      sparkNet.fit(new File(exportIterPath, "train").toString)
      println("Iterating Epoch:%d".format(epoch))
    }

    /**Delete the temp training files, now that we are done with them**/
    tm.deleteTempFiles(sc)

    /** Evaluations: Samples for Small Data **/
    val queueSize:Int = 2
    val valIter:DataSetIterator  = loadAsyncIterator(new File(exportIterPath, "val"),  pattern, queueSize)
    val valDataSet = scala.collection.mutable.MutableList[DataSet]()
    while(valIter.hasNext()) { valDataSet+= valIter.next() }
    val valRdd:RDD[DataSet] = sc.parallelize(valDataSet)
    //Per Release 0.9.1 Bug
    val batchSizeEval:Int = 16
    //val eval:Evaluation = sparkNet.evaluate(valRdd)
    val eval:Evaluation = sparkNet.doEvaluation(valRdd, batchSizeEval, new Evaluation(numClasses))(0)
    println("Accuracy on Validation: %.2f".format(eval.accuracy()))

  }

  /**
    * Load Iterator that have been saved
    * @param path
    * @param pattern
    * @return
    */
  def loadAsyncIterator(path:File, pattern:String, queueSize:Int):DataSetIterator = {
    val existingIter  = new ExistingMiniBatchDataSetIterator(path, pattern)
    val asyncIter     = new AsyncDataSetIterator(existingIter, queueSize)
    return asyncIter
  }
}
