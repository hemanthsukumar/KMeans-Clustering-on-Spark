// Hemanth Sukumar Vangala
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

object KMeans {
  // Define a type alias for a point (x, y) in 2D space
  type Point = (Double, Double)

  // Function to calculate Euclidean distance between two points x and y
  def distance(x: Point, y: Point): Double = {
    math.sqrt(math.pow(x._1 - y._1, 2) + math.pow(x._2 - y._2, 2))
  }

  // Function to find the closest centroid to a given point p from an array of centroids
  def closest_point(p: Point, cs: Array[Point]): Point = {
    cs.minBy(c => distance(p, c))  // Find the centroid with the minimum distance to point p
  }

  def main(args: Array[String]): Unit = {
    // Set up Spark configuration and context
    val conf = new SparkConf().setAppName("KMeans").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Read dataset of points from the input file (points-small.txt or points-large.txt)
    val points: RDD[Point] = sc.textFile(args(0)).map { line =>
      val parts = line.split(",")
      (parts(0).toDouble, parts(1).toDouble)  // Convert the input line to a (Double, Double) tuple
    }

    // Read initial centroids from the centroids.txt file
    var centroids: Array[Point] = sc.textFile(args(1)).map { line =>
      val parts = line.split(",")
      (parts(0).toDouble, parts(1).toDouble)  // Convert the input line to a (Double, Double) tuple
    }.collect()  // Collect centroids into an array for broadcasting

    // Perform 5 iterations of Lloyd's algorithm to update centroids
    for (i <- 1 to 5) {
      // Broadcast the centroids to all workers for efficient access
      val broadcastCentroids = sc.broadcast(centroids)

      // Assign each point to the closest centroid
      val clusters = points.map { p =>
        val cs = broadcastCentroids.value  // Retrieve the broadcasted centroids
        (closest_point(p, cs), p)  // (closest_centroid, point) pair
      }

      // Group points by their closest centroid and calculate the new centroids
      centroids = clusters.groupByKey()  // Group points by centroid
        .mapValues { points =>
          // Calculate the new centroid as the mean of all points in the cluster
          val count = points.size
          val sumX = points.map(_._1).sum
          val sumY = points.map(_._2).sum
          (sumX / count, sumY / count)  // Return the new centroid
        }.collect()  // Collect new centroids
        .map(_._2)  // Extract the centroid values from the (centroid, points) pairs

      // Destroy the broadcast variable after each iteration to free resources
      broadcastCentroids.destroy()
    }

    // Print the final centroids formatted to 2 decimal places
    centroids.foreach { case (x, y) =>
      println(f"\t$x%2.2f\t$y%2.2f")
    }

    // Stop the SparkContext to release resources
    sc.stop()
  }
}
