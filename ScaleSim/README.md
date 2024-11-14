* csv_detr.py &rarr;
  Imports DETR for Opbject Detection model from pretrained models collection of "trasformers" library.
  Then, create the .csv file with all convolutional layers' parameters, as required for ScaleSim topology.

* detr.csv &rarr;
  CSV file produced by csv_detr.py

* dse.py &rarr;
  1. Create the Design Space with different parameters for the Google TPU Architecture.
     The configurable properties are:
    * ArrayHeight and ArrayWidth, which are the Systollic array dimensions
    * IfmapSramSzkB and FilterSramSzkB, are the sizes of memory for input feature map and filter
    * OfmapSramSzkB, is the output feature map size
    * Dataflow, which can be weight/input/output static
    * Bandwidth, for the memory bandwidth in bytes
    * MemoryBanks, which is the number of memory banks
  2. Run ScaleSim for current TPU architecture instance and keep the total training latency, as fitness for the optimization algorithm.
  3. Use PyGAD genetic algorithm to find the optimal solution, which will be the TPU architecture with the best performance (smallest latency).
  
