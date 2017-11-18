package ai.skymind.apps

import java.io.File
import java.util

import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.nn.api.{Layer, OptimizationAlgorithm, Updater}
import org.nd4j.linalg.learning.config.{Adam, Nadam}
//import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.layers.{DropoutLayer, GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator
import org.deeplearning4j.eval.{Evaluation, ROC}
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer

//import org.nd4j.linalg.learning.config.{Adam, Nadam}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.{MCXENT, NEGATIVELOGLIKELIHOOD}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.{DataSet, ExistingMiniBatchDataSetIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator


object LstmApp {

  /**
    * Main Function
    * @param args Arugments
    */
  def main(args: Array[String]): Unit = {
    /** Define Model Configuration and intialize Network **/
    val learningRate:Double = 1e-3
    val numFeatures:Int     = 25        //for first level input
    val numClasses:Int      = 3
    val dropoutRate:Double  = (1-0.2)   //.dropout is the fraction of activations to keep not drop
    val loss:LossFunction   = if (numClasses > 2) MCXENT else NEGATIVELOGLIKELIHOOD
    val act:Activation      = if (numClasses > 2) Activation.SOFTMAX else Activation.SIGMOID

    val confGraph:ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
      //initial hyper-parameters
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      //.updater(new Adam(learningRate))
      .learningRate(learningRate)
      .updater(new Adam())
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(0.5)
      .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
      .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
      .iterations(1)
      .seed(11182017)
      .graphBuilder()
      .addInputs("input")
      .addLayer("lstm1", new GravesLSTM.Builder()
        .nIn(numFeatures)
        .nOut(164)
        .activation(Activation.TANH)
        .weightInit(WeightInit.XAVIER)
        .dropOut(dropoutRate)
        //.dropOut(new Dropout(dropoutRate))
        .build(), "input")
      .addLayer("lstm2", new GravesLSTM.Builder()
        .nIn(164)
        .nOut(96)
        .activation(Activation.TANH)
        .weightInit(WeightInit.XAVIER)
        .dropOut(dropoutRate)
        //.dropOut(new Dropout(dropoutRate))
        .build(), "lstm1")
      .addLayer("lstm3", new GravesLSTM.Builder()
        .nIn(96)
        .nOut(48)
        .activation(Activation.TANH)
        .weightInit(WeightInit.XAVIER)
        .dropOut(dropoutRate)
        //.dropOut(new Dropout(dropoutRate))
        .build(), "lstm2")
      .addLayer("output", new RnnOutputLayer.Builder(loss)
        .activation(act)
        .weightInit(WeightInit.XAVIER)
        .nIn(48)
        .nOut(numClasses).build(),"lstm3")
      .setOutputs("output")
      .backpropType(BackpropType.TruncatedBPTT)
      .tBPTTForwardLength(50)
      .tBPTTBackwardLength(50)
      .pretrain(false).backprop(true)
      .build()

    /** Configure and Initialize Network **/
    val net = new ComputationGraph(confGraph)
    net.init()
    println(net.getConfiguration())
    println(net.summary())

    /** Load Existing DataSet Iterators **/
    val queueSize:Int   = 2
    val pattern: String = "dataset_iter_%d.bin"
    val exportBasePath:File = new File(System.getProperty("user.home"), "data/nasa-turbofan/exports/ckpts")
    val exportIterPath:File = new File(exportBasePath, "iterators")
    val trainIter:DataSetIterator = loadAsyncIterator(new File(exportIterPath, "train"), pattern, queueSize)
    val valIter:DataSetIterator   = loadAsyncIterator(new File(exportIterPath, "val"),   pattern, queueSize)
    val testIter:DataSetIterator  = loadAsyncIterator(new File(exportIterPath, "test"),  pattern, queueSize)

    /** Add Listeners to Network **/
    /** Add Monitoring to the Network **/
    val listenerFreq = 1
    val inMemoryStorage:StatsStorage = new InMemoryStatsStorage()
    val listeners:util.Collection[IterationListener] = new util.ArrayList[IterationListener]() {
      add(new StatsListener(inMemoryStorage, listenerFreq))
      add(new ScoreIterationListener(listenerFreq))
    }
    val uiServer:UIServer = UIServer.getInstance()
    uiServer.attach(inMemoryStorage)     //Attach storage instance to UI to be visualized
    net.setListeners(listeners)          //Set Listeners before training (On Traditional Network)

    /** Fit Network **/
    val numEpochs:Int = 10
    var iterCnt:Int   = 10
    for(epoch<-1 to numEpochs) {
      while (trainIter.hasNext()) {
        val ds: DataSet = trainIter.next()
        net.fit(ds)
        if(iterCnt %10 == 0)
        {
          //Evaluation on Batch
          val features:INDArray    = ds.getFeatureMatrix()
          val labels:INDArray      = ds.getLabels()
          val yhat:Array[INDArray] = net.output(false, features)
          //particular training batch scoring
          val eval:Evaluation      = new Evaluation(numClasses)
          eval.evalTimeSeries(labels, yhat(0))
          println("Evaluation on Training Batch:")
          println(eval.stats())
          //how is validation performing
          val evalIter:Evaluation  = net.evaluate(valIter)
          println("Evaluation on Validation Data:")
          println(eval.accuracy, eval.f1, eval.precision, eval.recall)
          println(eval.stats())
          valIter.reset()
        }
        iterCnt+=1
      }
      //After each epoch: reset the DataSetIterator for the next one
      trainIter.reset()
    }


    /** Finally Evaluate on the Test Set **/
    evaluateIterator(net, testIter)

    /** Serialize the Network for reuse and **/
    val saveUpdater:Boolean=true
    val modelPath:File = new File(System.getProperty("user.home"), "data/nasa-turbofan/exports/ckpts/models")
    val modelFile:File = new File(modelPath, "dl4j_lstm.h5")
    ModelSerializer.writeModel(net, modelFile, saveUpdater)
    val preProc:DataSetPreProcessor = trainIter.getPreProcessor()
    //ModelSerializer.addNormalizerToModel(trainIter.getPreProcessor())

    /** Restore Network from Disk for Inference using ParallelInference */
    //val netRestored:ComputationGraph = ModelSerializer.restoreComputationGraph(modelFile.toString(), saveUpdater)
    //ModelSerializer.restoreNormalizerFromFile()

  }

  /**
    * Evaluate on the DataSetIterator
    * @param net C omputationGraph
    * @param iter DataSetIterator
    */
  def evaluateIterator(net:ComputationGraph, iter:DataSetIterator): Unit = {
    val roc:ROC = new ROC()
    val eval:Evaluation = net.evaluate(iter)
    println("Evaluate model on DataSetIterator")
    println(eval.accuracy, eval.f1, eval.precision, eval.recall)
    println(eval.stats())
    iter.reset()
  }

  /**
    * Evaluate DataSet Batch iteratively
    * @param net ComputationGraph
    * @param iter DataSetIterator
    * @param numClasses numClasses
    */
  def evaluateOnBatch(net:ComputationGraph, iter:DataSetIterator, numClasses:Int): Unit = {
    val roc:ROC = new ROC()
    val eval:Evaluation = new Evaluation(numClasses)
    while(iter.hasNext()) {
      val ds:DataSet = iter.next()
      val features:INDArray = ds.getFeatureMatrix()
      val labels:INDArray   = ds.getLabels()
      val yhat:Array[INDArray] = net.output(false, features)
      println(eval.evalTimeSeries(labels, yhat(0)))
    }
    println(eval.stats())
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

  /**
    * Log Network Layer Parameters
    * @param layers Layers for the Network Configuration
    */
  def logNetworkLayers(layers: Array[Layer]): Unit = {
    println("Number of parameters by layer:")
    var totalNumParams = 0
    for (l <- layers) {
      val numParams = l.numParams
      totalNumParams += numParams
      println("\tLayer Name: %s => Number Params: %d".format(l.conf.getLayer.getLayerName, numParams))
    }
    println("Total number of Network parameters: %d".format(totalNumParams))
  }

}
