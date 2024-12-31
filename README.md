### Project Description: K-Means Clustering with Spark and Scala

The goal of this project is to implement one step of the **Lloyd's algorithm** for K-Means clustering using **Spark** and **Scala**. In K-Means clustering, the algorithm divides a set of points into `k` clusters based on the proximity to initial centroids. The process involves the following steps:

1. **Assignment Step**: Assign each point to the nearest centroid, forming `k` clusters.
2. **Update Step**: Recompute the centroids of each cluster by calculating the mean of the points assigned to each centroid.

#### **Dataset:**
- The points are random 2D points distributed in a plane, where each point belongs to a grid square defined by `(i*2+1, j*2+1)` to `(i*2+2, j*2+2)` for `0 ≤ i ≤ 9` and `0 ≤ j ≤ 9`, meaning the dataset has `k = 100` clusters.
- Initial centroids are given in `centroids.txt`, and the points are stored in files like `points-small.txt` (for testing in local mode) or `points-large.txt` (for testing in distributed mode).

#### **Task Overview:**
- **Objective**: Implement one step of the K-Means algorithm using Spark to compute the new centroids.
- **Repetitions**: The update step needs to be repeated **5 times** to compute the final centroids.
- **Broadcasting**: Use Spark's **broadcast variables** to efficiently send the centroids to worker nodes in each iteration.

### **Steps to Implement:**

1. **Read the Data:**
   - Read the `points-small.txt` or `points-large.txt` file to get the points for clustering.
   - Read the `centroids.txt` file to get the initial centroids.

2. **Broadcast the Centroids:**
   - The centroids need to be broadcast to all worker nodes before each map step so that each point can find its nearest centroid.
   
3. **Assign Points to Centroids:**
   - For each point `p`, calculate the Euclidean distance to each centroid, and assign the point to the centroid that minimizes this distance. This step is called the **Assignment Step**.

4. **Recompute New Centroids:**
   - After the points are assigned to centroids, recompute the centroids by taking the mean of all the points assigned to each centroid. This is the **Update Step**.

5. **Repeat for 5 Iterations:**
   - Perform the **Assignment** and **Update** steps 5 times to refine the centroids.

6. **Output:**
   - After 5 iterations, output the new centroids.

### **Main Logic in Code:**

```scala
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

case class Point(x: Double, y: Double)
case class Centroid(x: Double, y: Double)

object KMeans {
  def closestCentroid(p: Point, centroids: Array[Centroid]): Centroid = {
    centroids.minBy(c => distance(p, c))
  }

  def distance(p: Point, c: Centroid): Double = {
    math.sqrt(math.pow(p.x - c.x, 2) + math.pow(p.y - c.y, 2))
  }

  def computeNewCentroid(points: Iterable[Point]): Centroid = {
    val sum = points.foldLeft((0.0, 0.0)) { case ((sumX, sumY), point) =>
      (sumX + point.x, sumY + point.y)
    }
    val size = points.size
    Centroid(sum._1 / size, sum._2 / size)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("KMeans").getOrCreate()

    // Read points and initial centroids
    val pointsRDD = spark.read.textFile(args(0)).rdd.map { line =>
      val parts = line.split(",")
      Point(parts(0).toDouble, parts(1).toDouble)
    }
    val centroids = spark.read.textFile(args(1)).rdd.map { line =>
      val parts = line.split(",")
      Centroid(parts(0).toDouble, parts(1).toDouble)
    }.collect()

    var centroidsBroadcast: Broadcast[Array[Centroid]] = spark.sparkContext.broadcast(centroids)

    // Run for 5 iterations
    for (i <- 1 to 5) {
      // Broadcast centroids to workers
      centroidsBroadcast = spark.sparkContext.broadcast(centroids)

      // Assignment Step: Assign points to the closest centroid
      val assignedPoints = pointsRDD.map { p =>
        val closest = closestCentroid(p, centroidsBroadcast.value)
        (closest, p)
      }.groupByKey()

      // Update Step: Compute new centroids
      val newCentroids = assignedPoints.mapValues(computeNewCentroid).collect()

      // Extract new centroids and broadcast them
      val newCentroidsArray = newCentroids.map { case (centroid, _) => centroid }.toArray
      centroidsBroadcast = spark.sparkContext.broadcast(newCentroidsArray)

      // Collect and print new centroids
      println(s"New centroids after iteration $i: ${newCentroidsArray.mkString(", ")}")
    }

    spark.stop()
  }
}
```

### **Explanation of Code:**

1. **Data Structures:**
   - `Point` represents a point in the 2D plane.
   - `Centroid` represents a centroid, also in the 2D plane.

2. **Functions:**
   - `closestCentroid`: This function finds the nearest centroid for a given point by calculating the Euclidean distance to all centroids and selecting the one with the smallest distance.
   - `distance`: Computes the Euclidean distance between a point and a centroid.
   - `computeNewCentroid`: Computes the new centroid of a cluster by averaging the points in the cluster.

3. **Main Program Flow:**
   - Reads points and initial centroids from files.
   - Broadcasts the centroids to all worker nodes.
   - Repeats the **Assignment** and **Update** steps for 5 iterations.
   - Outputs the new centroids after each iteration.

### **Execution:**

1. **Build the Program:**
   - Compile the program on Expanse:
     ```bash
     run kmeans.build
     ```

2. **Run in Local Mode (Small Data):**
   - Run the program for the small dataset using the following command:
     ```bash
     sbatch kmeans.local.run
     ```
   - Verify the output in `kmeans.local.out` and compare it with `solution-small.txt`.

3. **Run in Distributed Mode (Large Data):**
   - Once the local run works correctly, run the program for the large dataset:
     ```bash
     sbatch kmeans.distr.run
     ```
   - Verify the output in `kmeans.distr.out` and compare it with `solution-large.txt`.

### **Final Output:**
- After 5 iterations, the new centroids are printed to the output.
- Make sure the output matches the results provided in the `solution-small.txt` and `solution-large.txt` files (order does not matter).

### **Notes:**
- The centroids are broadcasted to all workers in each iteration to ensure that each worker has access to the current centroids.
- The program is run both locally (for smaller datasets) and in a distributed manner on Expanse (for larger datasets).
