package ai.skymind.apps

import java.io.File
import java.util.Random

import org.datavec.api.records.listener.impl.LogRecordListener
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.records.reader.impl.transform.TransformProcessSequenceRecordReader
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.nd4j.linalg.dataset.api.preprocessor.{NormalizerMinMaxScaler, NormalizerStandardize}
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

object LstmIngestApp {

  def main(args: Array[String]): Unit = {

    val exportPath:File  = new File(System.getProperty("user.home"), "data/nasa-turbofan/exports/ckpts/iterators")
    val baseDir:File     = new File(System.getProperty("user.home"), "data/nasa-turbofan/exports/ckpts/data")
    val trainDir:File    = new File(baseDir, "train")
    val testDir:File     = new File(baseDir, "test")
    val numTrainRecs:Int = getRecordCount(trainDir.toString)
    val numTestRecs:Int  = getRecordCount(trainDir.toString)
    val splitRec:Int     = (numTrainRecs* 0.8).toInt
    val numClasses:Int   = 3

    /***Stage 1: Define Transformation Process to occur for DataSet Iterators **/
    /** Define Transformation process: to filter out unused parameter: id **/
    val builderFeatures:Schema.Builder = new Schema.Builder
    List("id", "cycle", "cycle_norm").map(col => builderFeatures.addColumnsString(col))
    (1 to  3).map(col => builderFeatures.addColumnString("setting%d".format(col)))
    (1 to 21).map(col => builderFeatures.addColumnString("s%d".format(col)))
    val schemaFeatures:Schema = builderFeatures.build()
    val jsonSchemaFeatures:String = schemaFeatures.toJson()
    val tpFeatures = new TransformProcess.Builder(schemaFeatures).removeColumns("id", "cycle_norm").build()
    println("%s".format(jsonSchemaFeatures))
    println("%s".format(tpFeatures.getFinalSchema))

    /** Define Transformation Process: to solve for Multi-Class Classification **/
    //Labels: RUL (Regression) | label1 (Binary Classification) | label2 (MultiClass Classification)
    //ID|CYCLE kept for proper tracking, configure for the type of output
    val builderLabels:Schema.Builder = new Schema.Builder
    List("id", "cycle", "RUL", "label1", "label2").map(col => builderLabels.addColumnInteger(col))
    val schemaLabels:Schema = builderLabels.build()
    val jsonSchemaLabels:String   = schemaLabels.toJson()
    val tpLabels = new TransformProcess.Builder(schemaLabels).removeColumns("id", "cycle", "RUL", "label1").build()
    println("%s".format(jsonSchemaLabels))
    println("%s".format(tpLabels.getFinalSchema))

    /**Stage 2: Define DataSetIterators on RecordReader w/Transformation Process**/
    //Assume Time Series Partition Split: Treating as an Ordered Split
    val header:Boolean = true
    val startRow:Int   = if (header) 1 else 0
    val startFile:Int  = 1
    val batchSize:Int  = 16
    val trainIter:DataSetIterator = createIterator(trainDir.toString, startFile,  splitRec, startRow,
      tpFeatures, tpLabels, batchSize, numClasses)
    val valIter:DataSetIterator = createIterator(trainDir.toString,   splitRec+1, numTrainRecs, startRow,
      tpFeatures, tpLabels, batchSize, numClasses)
    val testIter:DataSetIterator  = createIterator(testDir.toString,  startFile,  numTestRecs, startRow,
      tpFeatures, tpLabels, batchSize, numClasses)

    /**Stage 3: Define and Configure PreProcessors **/
    var preProcessor:NormalizerStandardize = new NormalizerStandardize()
    preProcessor.fitLabel(false)
    preProcessor.fit(trainIter)
    //set the cursor back to the beginning after capturing statistics
    //set the preprocessor on the iterator, to be called during next
    trainIter.reset()
    trainIter.setPreProcessor(preProcessor)
    valIter.setPreProcessor(preProcessor)
    testIter.setPreProcessor(preProcessor)
    println("TrainIter Batches Available: %B".format(trainIter.hasNext()))
    println("Num Features: %d | Num OutComes: %d | Async Support: %b | Reset Support: %b".format(
        trainIter.inputColumns, trainIter.totalOutcomes, trainIter.asyncSupported, trainIter.resetSupported)
    )
    /** Save Iterators **/
    saveIterators(trainIter, "train", exportPath.toString)
    saveIterators(valIter,   "val",   exportPath.toString)
    saveIterators(testIter,  "test",  exportPath.toString)
  }


  /**
    * Retrieves the number of Records before finding metadata
    * @param partDir partition Directory to get File Count from
    */
  def getRecordCount(partDir:String):Int = {
    val seed = new Random(11182017)
    val features:File = new File(partDir, "features")
    val filesInDir:FileSplit = new FileSplit(features, Array[String]("csv"), seed)
    val numRecords:Int  = filesInDir.length.toInt
    numRecords
  }


  /**
    * Save Iterator to File with pattern in respective folder
    * @param iterator DataSetIterator to save
    * @param phase    train/validation/test
    * @param path     String Path to export to
    */
  def saveIterators(iterator:DataSetIterator, phase:String, path:String) = {
    var index:Int = 0
    val filePath:File = new File(path, phase)
    filePath.mkdirs()
    while(iterator.hasNext()) {
      val ds:DataSet    = iterator.next()
      val pat: String   = "dataset_iter_%d.bin".format(index)
      val fileRes:File  = new File(filePath, pat)
      ds.save(fileRes)
      index+=1
    }
  }


  /**
    * Creates a Sequence DataSetIterator
    * Alt format: WritableConverter
    *
    * @param dirPath       partition directory Path (e.g. base_url/train)
    * @param startFile     index on to start numerical format from
    * @param stopFile      index up to start reading from (inclusive)
    * @param startRow      weather to skip for existent of a header
    * @param tpInputs      Transformation Process
    * @param tpTargets     Transformation Process
    * @param miniBatchSize Batch Size
    * @param numClasses    numClasses per DataSetIterator
    */
  @throws[Exception]
  def createIterator(dirPath: String,
                     startFile: Integer,
                     stopFile: Integer,
                     startRow: Integer,
                     tpInputs: TransformProcess,
                     tpTargets: TransformProcess,
                     miniBatchSize: Integer,
                     numClasses: Integer): DataSetIterator = {

    val regression: Boolean = if (numClasses == -1) true else false
    val partDir = new File(dirPath)
    val features = new File(partDir, "features")
    val labels = new File(partDir, "labels")

    //configure CSV Sequence Record Reader for Features
    val featuresPattern = features.getAbsolutePath + "/turbofan-engine-ts-%d.csv"
    val isFeatures = new NumberedFileInputSplit(featuresPattern, startFile, stopFile)
    val rrSeqFeatures = new CSVSequenceRecordReader(startRow)
    rrSeqFeatures.initialize(isFeatures)
    rrSeqFeatures.setListeners(new LogRecordListener)

    //configure CSV Sequence Record Reader for Labels
    val labelsPattern = labels.getAbsolutePath + "/turbofan-engine-ts-%d.csv"
    val isLabels = new NumberedFileInputSplit(labelsPattern, startFile, stopFile)
    val rrSeqLabels = new CSVSequenceRecordReader(startRow)
    rrSeqLabels.initialize(isLabels)
    rrSeqLabels.setListeners(new LogRecordListener)

    //Transform Process for Features & Labels
    //Time Series of Different Lengths between records, Same Size Inputs & Labels (Sequence)
    val tpRrSeqFeatures = new TransformProcessSequenceRecordReader(rrSeqFeatures, tpInputs)
    val tpRrSeqLabels = new TransformProcessSequenceRecordReader(rrSeqLabels, tpTargets)
    val iterator = new SequenceRecordReaderDataSetIterator(tpRrSeqFeatures, tpRrSeqLabels,
      miniBatchSize, numClasses, regression, AlignmentMode.ALIGN_END)

    iterator.setCollectMetaData(true)
    iterator
  }


}



