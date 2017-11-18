package ai.skymind.apps

import java.io.File

import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.storage.FileStatsStorage

object MonitorApp {
  def main(args: Array[String]): Unit = {
    val uiServer:UIServer = UIServer.getInstance()
    val statsStorage = new FileStatsStorage(new File("net_monitoring.dl4j"))
    uiServer.attach(statsStorage)
    //open training UI
  }
}
