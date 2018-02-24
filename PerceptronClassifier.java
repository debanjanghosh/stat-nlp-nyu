package nlp.assignments;

import java.io.IOException;
import java.util.List;
import java.util.Collection;
import java.util.ArrayList;
import java.util.Arrays;

import nlp.classify.*;
import nlp.math.DifferentiableFunction;
import nlp.math.DoubleArrays;
import nlp.math.GradientMinimizer;
import nlp.math.LBFGSMinimizer;
import nlp.math.SloppyMath;
import nlp.util.Counter;
import nlp.util.Indexer;
import nlp.util.Pair;

/**
 * Maximum entropy classifier for assignment 2. You will have to fill in the
 * code gaps marked by TODO flags. To test whether your classifier is
 * functioning correctly, you can invoke the main method of this class using
 * <p/>
 * java edu.berkeley.nlp.assignments.MaximumEntropyClassifier
 * <p/>
 * This will run a toy test classification.
 */
public class PerceptronClassifier<I, F, L> implements
		ProbabilisticClassifier<I, L> {

	/**
	 * Factory for training MaximumEntropyClassifiers.
	 */
	public static class Factory<I, F, L> implements
			ProbabilisticClassifierFactory<I, L> {

		double sigma;
		int iterations;
		FeatureExtractor<I, F> featureExtractor;
		
		private static IndexLinearizer indexLinearizer ;

		public ProbabilisticClassifier<I, L> trainClassifier(
				List<LabeledInstance<I, L>> trainingData) throws IOException 
		{
			// build data encodings so the inner loops can be efficient
			Encoding<F, L> encoding = buildEncoding(trainingData);
			indexLinearizer = buildIndexLinearizer(encoding);
			double[] initialWeights = buildInitialWeights(indexLinearizer);
			EncodedDatum[] data = encodeData(trainingData, encoding);
			// build a minimizer object
			
			int numLabels = encoding.getNumLabels() ;
			weights = trainIterations(initialWeights,numLabels,data);
			
			// build a classifer using these weights (and the data encodings)
			return new PerceptronClassifier<I, F, L>(weights, encoding,
					indexLinearizer, featureExtractor);
		}

		private int getMaxIndex(double[] weightFeatSum) 
		{
			// TODO Auto-generated method stub
			double max = 0 ;
			int index = -1 ;
			
			for ( int i = 0 ; i < weightFeatSum.length ; i++)
			{
				double weight = weightFeatSum[i] ;
				if ( weight >= max)
				{
					weight = max ;
					index = i ;
				}
			}
					
			return index ;
		}
		
		public double prediction(EncodedDatum inst, int classVal,double[]x) 
		{
			 double sum = 0.0;
			 int[] featIndexes = inst.featureIndexes ;
				
			 for ( int num = 0 ; num< featIndexes.length ; num++)
			 {
				int featIndex = inst.getFeatureIndex(num);
				double featValue = inst.getFeatureCount(num);
				int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, classVal) ;
				sum += x[linearFeatIndex] * featValue ;
			 }
			 int last = indexLinearizer.getLinearIndex(featIndexes.length-1, classVal) ;
			 sum += x[last];
			 return 1.0 / (1.0 + Math.exp(-sum));
		}
		
		private double[] trainIterations ( double[]x , int numLabels, EncodedDatum[] data )
		{
			double learningRatio = 0.2 ;
			
			double[] preds = new double[numLabels];
			
		//	for ( i)
			
			for ( int itr = 0 ; itr < 200 ; itr ++)

			{
			for ( int datumIdx = 0 ; datumIdx< data.length ; datumIdx++)
			{
				EncodedDatum inst = data[datumIdx] ;
			
				for (int i = 0; i < numLabels; i++) 
				{
					preds[i] = prediction(inst, i,x);
				}
			
				int actualClass = (int) inst.labelIndex ;
				
			    for (int i = 0; i < numLabels; i++) 
			    {
			         double actual = (i == actualClass) ? 1.0 : 0.0;
			         double delta = (actual - preds[i]) * preds[i] * (1 - preds[i]);
			         int[] featIndexes = inst.featureIndexes ;
						
					 for ( int num = 0 ; num< featIndexes.length ; num++)
			         {
							double featValue = inst.getFeatureCount(num);
							int linearFeatIndex = indexLinearizer.getLinearIndex(num, i) ;
					         x[linearFeatIndex] += learningRatio * delta * featValue;
			         }
				
					 int last = indexLinearizer.getLinearIndex(featIndexes.length-1, i) ;
					 x[last] +=learningRatio * delta;
				
			        //     this.weightAttribute[i][inst.numAttributes() - 1] += learningRatio * delta;
			    }
			}
			}
			return x ;
			
		}
		
		public double[] getVotesForInstance(EncodedDatum inst )
		{
			 double[] votes = new double[2];
			             
			 for (int i = 0; i < votes.length; i++)
			 {
			      votes[i] = prediction(inst, i,weights);
			  }
			         return votes;
		}
		
		private  double[] trainIterations1(double[] x, int numLabels, EncodedDatum[] data) 
		{
			// TODO Auto-generated method stub
			double learning_rate = 0.25 ;
			
			// TODO: compute the objective and its derivatives
			// TODO

			// logProb
			//during classification stage - we do this for a single data 
			//where the "weight" is already learn. In training sessions we 
			//loop through each data point and labels to calculate first a similar 
			//log weight feature sum
			
			double[] weightFeatSum = new double[numLabels] ;
			
			for ( int itr = 0 ;itr < 50 ; itr++)
			{
			
				for ( int datumIdx = 0 ; datumIdx< data.length ; datumIdx++)
				{
					EncodedDatum datum = data[datumIdx] ;
					int[] featIndexes = datum.featureIndexes ;
					
					for ( int labelIdx = 0 ; labelIdx < numLabels ; labelIdx++ )
					{
						
						double internalSum = 0d ;
						
						for ( int num = 0 ; num < featIndexes.length ; num++)
						{
							int featIndex = datum.getFeatureIndex(num);
							double featValue = datum.getFeatureCount(num);
							
							//this is a confusing stage!
							int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, labelIdx) ;
							
							internalSum += x[linearFeatIndex] * featValue ;
						}
						weightFeatSum[labelIdx] = internalSum ;
					}
					//for perceptron algorithm - we need to find the following
					//Y* = arg max Wt *feat 
					int maxPredictedLabel = getMaxIndex(weightFeatSum) ;
					if ( maxPredictedLabel == datum.labelIndex)
					{
						//no change for perceptron!
						System.out.println("here") ;
						
					}
					else
					{
						//adjust the weights 
						//w_old = w_old + learning rate * (target - predicted) * feature
						double diff = (datum.labelIndex - maxPredictedLabel ) ;
					   
						for ( int labelIdx = 0 ; labelIdx < numLabels ; labelIdx++ )
						{
							for ( int num = 0 ; num < featIndexes.length ; num++)
							{
								int featIndex = datum.getFeatureIndex(num);
								double featValue = datum.getFeatureCount(num);
								  
								
								int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, labelIdx) ;
								if ( labelIdx != maxPredictedLabel)
									x[linearFeatIndex] +=  diff * featValue * learning_rate;
								else
									x[linearFeatIndex] -=  diff * featValue * learning_rate;

							}
						}
						
					}
					
				
				}
			}
			
			
			return x;
		}

		private double[] buildInitialWeights(IndexLinearizer indexLinearizer) 
		{
			return DoubleArrays.constantArray(0.1,
					indexLinearizer.getNumLinearIndexes());
		}

		private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) 
		{
			return new IndexLinearizer(encoding.getNumFeatures(),
					encoding.getNumLabels());
		}

		private Encoding<F, L> buildEncoding(List<LabeledInstance<I, L>> data) throws IOException 
		{
			Indexer<F> featureIndexer = new Indexer<F>();
			Indexer<L> labelIndexer = new Indexer<L>();
			for (LabeledInstance<I, L> labeledInstance : data) 
			{
				L label = labeledInstance.getLabel();
				Counter<F> features1 = featureExtractor
						.extractFeatures(labeledInstance.getInput());
				
				//how about class conditioned features here 
				
				LabeledFeatureVector<F, L> labeledDatum = new BasicLabeledFeatureVector<F, L>(
						label, features1);
				labelIndexer.add(labeledDatum.getLabel());
				for (F feature : labeledDatum.getFeatures().keySet()) 
				{
					featureIndexer.add(feature);
				}
			}
			return new Encoding<F, L>(featureIndexer, labelIndexer);
		}

		private EncodedDatum[] encodeData(List<LabeledInstance<I, L>> data,
				Encoding<F, L> encoding) throws IOException 
		{
			EncodedDatum[] encodedData = new EncodedDatum[data.size()];
			
			for (int i = 0; i < data.size(); i++) 
			{
				LabeledInstance<I, L> labeledInstance = data.get(i);
				L label = labeledInstance.getLabel();
				Counter<F> features = featureExtractor
						.extractFeatures(labeledInstance.getInput());
				LabeledFeatureVector<F, L> labeledFeatureVector = new BasicLabeledFeatureVector<F, L>(
						label, features);
				encodedData[i] = EncodedDatum.encodeLabeledDatum(
						labeledFeatureVector, encoding);
			}
			return encodedData;
		}

		/**
		 * Sigma controls the variance on the prior / penalty term. 1.0 is a
		 * reasonable value for large problems, bigger sigma means LESS
		 * smoothing. Zero sigma is a special indicator that no smoothing is to
		 * be done.
		 * <p/>
		 * Iterations determines the maximum number of iterations the
		 * optimization code can take before stopping.
		 */
		public Factory(double sigma, int iterations,
				FeatureExtractor<I, F> featureExtractor) 
		{
			this.sigma = sigma;
			this.iterations = iterations;
			this.featureExtractor = featureExtractor;
		}
	}

	/**
	 * This is the MaximumEntropy objective function: the (negative) log
	 * conditional likelihood of the training data, possibly with a penalty for
	 * large weights. Note that this objective get MINIMIZED so it's the
	 * negative of the objective we normally think of.
	 */
	public static class ObjectiveFunction<F, L> implements
			DifferentiableFunction 
	{
		IndexLinearizer indexLinearizer;
		Encoding<F, L> encoding;
		EncodedDatum[] data;

		double sigma;

		double lastValue;
		double[] lastDerivative;
		double[] lastX;

		public int dimension() 
		{
			return indexLinearizer.getNumLinearIndexes();
		}

		public double valueAt(double[] x) 
		{
			ensureCache(x);
			return lastValue;
		}

		public double[] derivativeAt(double[] x) 
		{
			ensureCache(x);
			return lastDerivative;
		}

		private void ensureCache(double[] x) 
		{
			if (requiresUpdate(lastX, x)) 
			{
				Pair<Double, double[]> currentValueAndDerivative = calculate(x);
				lastValue = currentValueAndDerivative.getFirst();
				lastDerivative = currentValueAndDerivative.getSecond();
				lastX = x;
			}
		}

		private boolean requiresUpdate(double[] lastX, double[] x) 
		{
			if (lastX == null)
				return true;
			
			for (int i = 0; i < x.length; i++) 
			{
				if (lastX[i] != x[i])
					return true;
			}
			return false;
		}

		/**
		 * The most important part of the classifier learning process! This
		 * method determines, for the given weight vector x, what the (negative)
		 * log conditional likelihood of the data is, as well as the derivatives
		 * of that likelihood wrt each weight parameter.
		 */
		
		
		private Pair<Double, double[]> calculate(double[] x) 
		{
			double objective = 0.0;
			double[] derivatives = DoubleArrays.constantArray(0.0, dimension());
			
			double learning_rate = 0.25 ;
			
			// TODO: compute the objective and its derivatives
			// TODO

			// logProb
			//during classification stage - we do this for a single data 
			//where the "weight" is already learn. In training sessions we 
			//loop through each data point and labels to calculate first a similar 
			//log weight feature sum
			
			double[] weightFeatSum = new double[encoding.getNumLabels()] ;
			double[] weightFeatSumEachClass = new double[encoding.getNumLabels()] ;
			
			for ( int itr = 0 ;itr < 100 ; itr++)
			{
			
				for ( int datumIdx = 0 ; datumIdx< data.length ; datumIdx++)
				{
					EncodedDatum datum = data[datumIdx] ;
					int[] featIndexes = datum.featureIndexes ;
					
					for ( int labelIdx = 0 ; labelIdx < encoding.getNumLabels() ; labelIdx++ )
					{
						L label = encoding.getLabel(labelIdx);
						
						double internalSum = 0d ;
						
						for ( int num = 0 ; num < featIndexes.length ; num++)
						{
							int featIndex = datum.getFeatureIndex(num);
							double featValue = datum.getFeatureCount(num);
							
							//this is a confusing stage!
							int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, labelIdx) ;
							
							internalSum += x[linearFeatIndex] * featValue ;
						}
						weightFeatSum[labelIdx] = internalSum ;
					}
					//for perceptron algorithm - we need to find the following
					//Y* = arg max Wt *feat 
					int maxPredictedLabel = getMaxIndex(weightFeatSum) ;
					if ( maxPredictedLabel == datum.labelIndex)
					{
						//no change for perceptron!
					}
					else
					{
						//adjust the weights 
						//w_old = w_old + learning rate * (target - predicted) * feature
						double diff = (datum.labelIndex - maxPredictedLabel ) ;
						objective += 0.5 * diff * diff;
	                   
						for ( int labelIdx = 0 ; labelIdx < encoding.getNumLabels() ; labelIdx++ )
						{
							L label = encoding.getLabel(labelIdx);
							for ( int num = 0 ; num < featIndexes.length ; num++)
							{
								int featIndex = datum.getFeatureIndex(num);
								double featValue = datum.getFeatureCount(num);
								  
								
								int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, labelIdx) ;
								
								if ( labelIdx != maxPredictedLabel)
									x[linearFeatIndex] +=  diff * featValue * learning_rate;
								else
									x[linearFeatIndex] =  diff * featValue * learning_rate;

							}
						}
						
					}
					
				
				}
			}
			
			
			
			
			return new Pair<Double, double[]>(objective, x);
		}

		private int getMaxIndex(double[] weightFeatSum) 
		{
			// TODO Auto-generated method stub
			double max = 0 ;
			int index = -1 ;
			
			for ( int i = 0 ; i < weightFeatSum.length ; i++)
			{
				double weight = weightFeatSum[i] ;
				if ( weight >= max)
				{
					weight = max ;
					index = i ;
				}
			}
					
			return index ;
		}

		public ObjectiveFunction(Encoding<F, L> encoding, EncodedDatum[] data,
				IndexLinearizer indexLinearizer, double sigma) 
		{
			this.indexLinearizer = indexLinearizer;
			this.encoding = encoding;
			this.data = data;
			this.sigma = sigma;
		}
	}

	/**
	 * EncodedDatums are sparse representations of (labeled) feature count
	 * vectors for a given data point. Use getNumActiveFeatures() to see how
	 * many features have non-zero count in a datum. Then, use getFeatureIndex()
	 * and getFeatureCount() to retreive the number and count of each non-zero
	 * feature. Use getLabelIndex() to get the label's number.
	 */
	public static class EncodedDatum 
	{

		public static <F, L> EncodedDatum encodeDatum(
				FeatureVector<F> featureVector, Encoding<F, L> encoding) 
		{
			Counter<F> features = featureVector.getFeatures();
			Counter<F> knownFeatures = new Counter<F>();
			for (F feature : features.keySet()) 
			{
				if (encoding.getFeatureIndex(feature) < 0)
					continue;
				knownFeatures.incrementCount(feature,
						features.getCount(feature));
			}
			
			int numActiveFeatures = knownFeatures.keySet().size();
			int[] featureIndexes = new int[numActiveFeatures];
			double[] featureCounts = new double[knownFeatures.keySet().size()];
			int i = 0;
			
			for (F feature : knownFeatures.keySet()) 
			{
				int index = encoding.getFeatureIndex(feature);
				double count = knownFeatures.getCount(feature);
				featureIndexes[i] = index;
				featureCounts[i] = count;
				i++;
			}
			EncodedDatum encodedDatum = new EncodedDatum(-1, featureIndexes,
					featureCounts);
			return encodedDatum;
		}

		public static <F, L> EncodedDatum encodeLabeledDatum(
				LabeledFeatureVector<F, L> labeledDatum, Encoding<F, L> encoding) 
		{
			EncodedDatum encodedDatum = encodeDatum(labeledDatum, encoding);
			encodedDatum.labelIndex = encoding.getLabelIndex(labeledDatum
					.getLabel());
			return encodedDatum;
		}

		int labelIndex;
		int[] featureIndexes;
		double[] featureCounts;

		public int getLabelIndex() {
			return labelIndex;
		}

		public int getNumActiveFeatures() {
			return featureCounts.length;
		}

		public int getFeatureIndex(int num) {
			return featureIndexes[num];
		}

		public double getFeatureCount(int num) {
			return featureCounts[num];
		}

		public EncodedDatum(int labelIndex, int[] featureIndexes,
				double[] featureCounts) {
			this.labelIndex = labelIndex;
			this.featureIndexes = featureIndexes;
			this.featureCounts = featureCounts;
		}
	}

	/**
	 * The Encoding maintains correspondences between the various representations
	 * of the data, labels, and features. The external representations of labels
	 * and features are object-based. The functions getLabelIndex() and
	 * getFeatureIndex() can be used to translate those objects to integer
	 * representations: numbers between 0 and getNumLabels() or getNumFeatures()
	 * (exclusive). The inverses of this map are the getLabel() and getFeature()
	 * functions.
	 */
	public static class Encoding<F, L> {
		Indexer<F> featureIndexer;
		Indexer<L> labelIndexer;

		public int getNumFeatures() {
			return featureIndexer.size();
		}

		public int getFeatureIndex(F feature) {
			return featureIndexer.indexOf(feature);
		}

		public F getFeature(int featureIndex) {
			return featureIndexer.get(featureIndex);
		}

		public int getNumLabels() {
			return labelIndexer.size();
		}

		public int getLabelIndex(L label) {
			return labelIndexer.indexOf(label);
		}

		public L getLabel(int labelIndex) {
			return labelIndexer.get(labelIndex);
		}

		public Encoding(Indexer<F> featureIndexer, Indexer<L> labelIndexer) {
			this.featureIndexer = featureIndexer;
			this.labelIndexer = labelIndexer;
		}
	}

	/**
	 * The IndexLinearizer maintains the linearization of the two-dimensional
	 * features-by-labels pair space. This is because, while we might think
	 * about lambdas and derivatives as being indexed by a feature-label pair,
	 * the optimization code expects one long vector for lambdas and
	 * derivatives. To go from a pair featureIndex, labelIndex to a single
	 * pairIndex, use getLinearIndex().
	 */
	public static class IndexLinearizer {
		int numFeatures;
		int numLabels;

		public int getNumLinearIndexes() {
			return numFeatures * numLabels;
		}

		public int getLinearIndex(int featureIndex, int labelIndex) {
			return labelIndex + featureIndex * numLabels;
		}

		public int getFeatureIndex(int linearIndex) {
			return linearIndex / numLabels;
		}

		public int getLabelIndex(int linearIndex) {
			return linearIndex % numLabels;
		}

		public IndexLinearizer(int numFeatures, int numLabels) {
			this.numFeatures = numFeatures;
			this.numLabels = numLabels;
		}
	}

	private static double[] weights;
	private Encoding<F, L> encoding;
	private IndexLinearizer indexLinearizer;
	private FeatureExtractor<I, F> featureExtractor;

	/**
	 * Calculate the log probabilities of each class, for the given datum
	 * (feature bundle). Note that the weighted votes (referred to as
	 * activations) are *almost* log probabilities, but need to be normalized.
	 */
	private static <F, L> double[] getLogProbabilities(EncodedDatum datum,
			double[] weights, Encoding<F, L> encoding,
			IndexLinearizer indexLinearizer) 
	{
		// TODO: apply the classifier to this feature vector
		// TODO
		// TODO
		// TODO
		int datumFeatLength = datum.featureCounts.length ;
		int[] featIndexes = datum.featureIndexes ;
		double[] weightFeatSum = new double[encoding.getNumLabels()] ;
		
		
		for ( int labelIdx = 0 ; labelIdx < encoding.getNumLabels(); labelIdx++)
		{
			L label = encoding.getLabel(labelIdx);
			
			double internalSum = 0 ;
			for ( int datumIdx = 0 ; datumIdx < featIndexes.length ; datumIdx++)
			{
				int featIndex = featIndexes[datumIdx];
				F feature = encoding.getFeature(featIndex);
				int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, labelIdx) ;
				internalSum += weights[linearFeatIndex] * datum.getFeatureCount(datumIdx) ;
			}
			weightFeatSum[labelIdx] = internalSum ;
		}
		
		 return weightFeatSum;
		
/*
		// dummy code
		double[] logProbabilities = DoubleArrays.constantArray(
				Double.NEGATIVE_INFINITY, encoding.getNumLabels());
		logProbabilities[0] = 0.0;
		return logProbabilities;
		// end dummy code
*/
		// TODO
	}

	public Counter<L> getProbabilities(I input) throws IOException 
	{
		
		
		FeatureVector<F> featureVector = new BasicFeatureVector<F>(
				featureExtractor.extractFeatures(input));
		EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
				encoding);
		
		return getVotesForInstance(encodedDatum) ;
		
	//	Counter L = new Counter[L];
		
	//	return logProbabiltyArrayToProbabiltyCounter(votes);
		
		
	//	return getProbabilities(featureVector);
	}

	private Counter<L> getVotesForInstance(EncodedDatum encodedDatum) 
	{
		// TODO Auto-generated method stub
		 double[] votes = new double[encoding.getNumLabels()];
		Counter<L> probabiltyCounter = new Counter<L>();
			
         
		 for (int i = 0; i < votes.length; i++)
		 {
		      votes[i] = prediction(encodedDatum, i,weights);
		      L label = encoding.getLabel(i);
			  probabiltyCounter.setCount(label, votes[i]);
		  }
		 return probabiltyCounter;
	}

	public double prediction(EncodedDatum inst, int classVal,double[]x) 
	{
		 double sum = 0.0;
		 int[] featIndexes = inst.featureIndexes ;
			
		 for ( int num = 0 ; num< featIndexes.length ; num++)
		 {
			int featIndex = inst.getFeatureIndex(num);
			double featValue = inst.getFeatureCount(num);
			int linearFeatIndex = indexLinearizer.getLinearIndex(featIndex, classVal) ;
			sum += x[linearFeatIndex] * featValue ;
		 }
		 int last = indexLinearizer.getLinearIndex(featIndexes.length-1, classVal) ;
		 sum += x[last];
		 return 1.0 / (1.0 + Math.exp(-sum));
	}
	private Counter<L> getProbabilities(FeatureVector<F> featureVector) 
	{
		EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
				encoding);
		
//		double learned[] = {0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5, 0.5, 0.0, 0.0} ;
		
		
		double[] logProbabilities = getLogProbabilities(encodedDatum, weights,
				encoding, indexLinearizer);
		return logProbabiltyArrayToProbabiltyCounter(logProbabilities);
	}

	private Counter<L> logProbabiltyArrayToProbabiltyCounter(
			double[] logProbabilities) {
		Counter<L> probabiltyCounter = new Counter<L>();
		
		for (int labelIndex = 0; labelIndex < logProbabilities.length; labelIndex++) 
		{
			double logProbability = logProbabilities[labelIndex];
			double probability = Math.exp(logProbability);
			L label = encoding.getLabel(labelIndex);
			probabiltyCounter.setCount(label, probability);
		}
		return probabiltyCounter;
	}

	public L getLabel(I input) throws IOException {
		return getProbabilities(input).argMax();
	}

	public PerceptronClassifier(double[] weights, Encoding<F, L> encoding,
			IndexLinearizer indexLinearizer,
			FeatureExtractor<I, F> featureExtractor) 
	{
		this.weights = weights;
		this.encoding = encoding;
		this.indexLinearizer = indexLinearizer;
		this.featureExtractor = featureExtractor;
	}

	public static void main(String[] args) throws IOException 
	{
		// create datums
		LabeledInstance<String[], String> datum1 = new LabeledInstance<String[], String>(
				"cat", new String[] { "fuzzy", "claws", "small" });
		LabeledInstance<String[], String> datum2 = new LabeledInstance<String[], String>(
				"bear", new String[] { "fuzzy", "claws", "big" });
		LabeledInstance<String[], String> datum3 = new LabeledInstance<String[], String>(
				"cat", new String[] { "claws", "medium" });
		LabeledInstance<String[], String> datum4 = new LabeledInstance<String[], String>(
				"cat", new String[] { "claws", "small" });

		// create training set
		List<LabeledInstance<String[], String>> trainingData = new ArrayList<LabeledInstance<String[], String>>();
		trainingData.add(datum1);
		trainingData.add(datum2);
		trainingData.add(datum3);

		// create test set
		List<LabeledInstance<String[], String>> testData = new ArrayList<LabeledInstance<String[], String>>();
		testData.add(datum4);

		// build classifier
		FeatureExtractor<String[], String> featureExtractor = new FeatureExtractor<String[], String>() {
			public Counter<String> extractFeatures(String[] featureArray) {
				return new Counter<String>(Arrays.asList(featureArray));
			}
		};
		PerceptronClassifier.Factory<String[], String, String> perceptronClassifierFactory = new PerceptronClassifier.Factory<String[], String, String>(
				1.0, 20, featureExtractor);
		ProbabilisticClassifier<String[], String> perceptronEntropyClassifier = perceptronClassifierFactory
				.trainClassifier(trainingData);
		System.out.println("Probabilities on test instance: "
				+ perceptronEntropyClassifier.getProbabilities(datum4.getInput()));
		
		
	}
}
