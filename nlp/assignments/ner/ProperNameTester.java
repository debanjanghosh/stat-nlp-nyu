package nlp.assignments;

import java.util.HashMap;
import java.util.List;
import java.util.Collection;
import java.util.ArrayList;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

import nlp.assignments.ner.BiCharactergramLabelClassifier;
import nlp.assignments.ner.TriCharactergramLabelClassifier;
import nlp.assignments.ner.UniCharactergramLabelClassifier;
import nlp.assignments.ner.UnigramLabelClassifier;
import nlp.classify.*;
import nlp.langmodel.LanguageModel;
import nlp.scoring.PRF1Measure;
import nlp.scoring.RefTargetPair;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.Pair;
import nlp.util.TextUtil;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester 
{
	private static HashMap<Double, List<RefTargetPair>> eachCategoryMap;
	
	private static HashMap<String,Double> LabelToNumMap;

	private static PRF1Measure prf1Object;

	private static ArrayList<String> LabelToNumSet;
	
	private static FeatureExtractorFunctions featExtractObj ;
	
	

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> 
	{

		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 * @throws IOException 
		 */
		//original function
	/*	
		public Counter<String> extractFeatures(String name) 
		{
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			// add character unigram features
			
			for (int i = 0; i < characters.length; i++) 
			{
				char character = characters[i];
				features.incrementCount("UNI-" + character, 1.0);
			}
			// TODO : extract better features!
			// TODO
			// TODO
			// TODO
			return features;
		}
		
		
	*/	
	/*	
		public Counter<String> extractFeatures(String name) throws IOException 
		{
			featExtractObj =  new FeatureExtractorFunctions();
			
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			
			// add character unigram features
			Counter<String> gramCharFeatures = featExtractObj.getGramFeatures(name);
			features.incrementAll(gramCharFeatures);
			
				
			//now set features for unigrams 
			//using bigram may lowe down - becuase there not much information
			Counter<String> gramWordFeatures = featExtractObj.getWordFeatures(name);
			features.incrementAll(gramWordFeatures);
			
		//	Counter<String> bigramWordFeatures = featExtractObj.getBigramFeatures(name);
		//	features.incrementAll(bigramWordFeatures);
			
			//seq of POS?
		
	//		Counter<String> nonalpha = featExtractObj.getNonAlphaFeatures(name);
	//		features.incrementAll(nonalpha);
		
			
			//create a feature if it contains number
			//good for medicines and films 
			Counter<String> numberFeatures = featExtractObj.getNumberFeatures(name);
			features.incrementAll(numberFeatures);
			
			//create a feature to check if it contains and ending of company
			Counter<String> companyFeatures = featExtractObj.getCompanyFeatures(name);
			features.incrementAll(companyFeatures);
			
			//some words ends like places - check
			
			Counter<String> placeFeatures = featExtractObj.getPlaceFeatures(name);
			features.incrementAll(placeFeatures);
			
			//some words very common in medicines
			Counter<String> medicineFeatures = featExtractObj.getMedicineFeatures(name);
			features.incrementAll(medicineFeatures);
			
			//some common names
			Counter<String> peopleFeatures = featExtractObj.getPeopleFeatures(name);
			features.incrementAll(peopleFeatures);
			
			//some movie names
	//		Counter<String> movieFeatures = featExtractObj.getMovieFeatures(name);
	//		features.incrementAll(movieFeatures);
					
			
			Counter<String> planceEndsFeatures = featExtractObj.getPlacePatternFeatures(name);
				features.incrementAll(planceEndsFeatures);
			
			
				
		//	Counter<String> countryFeatures = featExtractObj.getCountryPatternFeatures(name);
		//		features.incrementAll(countryFeatures);
			
			
			
			//caps features
			Counter<String> capsFeatures = featExtractObj.getCAPSFeatures(name);
			features.incrementAll(capsFeatures);
		
			//punctuation features
			Counter<String> punchFeatures = featExtractObj.getPunctuationFeatures(name);
			features.incrementAll(punchFeatures);
		
			
			Counter<String> patSumFeatures = featExtractObj.getSummarizedPatternFeatures(name);
			features.incrementAll(patSumFeatures);
		
			Counter<String> patFeatures = featExtractObj.PatternFeatures(name);
			features.incrementAll(patFeatures);
		
			features.incrementCount("LENGTH1",Math.log(name.length()));
			
			String tokens[] = name.split("\\s++");
			
			features.incrementCount("LENGTH2",tokens.length);
			
			
			// TODO : extract better features!
			// TODO
			// TODO
			// TODO
			return features;
		}
	*/	
	


		public Counter<String> extractFeatures(String name) throws IOException 
		{
			featExtractObj =  new FeatureExtractorFunctions();
			
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			
			// add character unigram features
			Counter<String> gramCharFeatures = featExtractObj.getGramFeatures(name);
			features.incrementAll(gramCharFeatures);
			
			Counter<String> gramWordFeatures = featExtractObj.getWordFeatures(name);
			features.incrementAll(gramWordFeatures);
		
		
			Counter<String> capsFeatures = featExtractObj.getCAPSFeatures(name);
			features.incrementAll(capsFeatures);
			
			Counter<String> numberFeatures = featExtractObj.getNumberFeatures(name);
			features.incrementAll(numberFeatures);
			
			Counter<String> punchFeatures = featExtractObj.getPunctuationFeatures(name);
			features.incrementAll(punchFeatures);
		
			Counter<String> patSumFeatures = featExtractObj.getSummarizedPatternFeatures(name);
			features.incrementAll(patSumFeatures);
		
			Counter<String> patFeatures = featExtractObj.PatternFeatures(name);
			features.incrementAll(patFeatures);
			
			Counter<String> placeFeatures = featExtractObj.getPlaceFeatures(name);
			features.incrementAll(placeFeatures);
		
		
			Counter<String> planceEndsFeatures = featExtractObj.getPlacePatternFeatures(name);
			features.incrementAll(planceEndsFeatures);
	
			Counter<String> movieFeatures = featExtractObj.getMovieFeatures(name);
					features.incrementAll(movieFeatures);
					
			Counter<String> medicineFeatures = featExtractObj.getMedicineFeatures(name);
			features.incrementAll(medicineFeatures);
			
			Counter<String> medicineStartFeatures = featExtractObj.getMedicineStartFeatures(name);
			features.incrementAll(medicineStartFeatures);
			
			Counter<String> medicineEndFeatures = featExtractObj.getMedicineEndFeatures(name);
			features.incrementAll(medicineEndFeatures);
			
			
			
			Counter<String> companyFeatures = featExtractObj.getCompanyFeatures(name);
			features.incrementAll(companyFeatures);
					
			features.incrementCount("LENGTH1",Math.log(name.length()));
					
			Counter<String> countryFeatures = featExtractObj.getCountryPatternFeatures(name);
			features.incrementAll(countryFeatures);
				
			
		//	Counter<String> stopWords = featExtractObj.checkStopWords(name);
		//	features.incrementAll(stopWords);
			
		//	Counter<String> peopleFeatures = featExtractObj.getPeopleFeatures(name);
		//	features.incrementAll(peopleFeatures);
			
	//		Counter<String> romanFeatures = featExtractObj.checkRomanNumerals(name);
	//		features.incrementAll(romanFeatures);
		
		
			
			
			String tokens[] = name.split("\\s++");
					
			features.incrementCount("LENGTH2",tokens.length);
				
			features.incrementCount("BIAS", 1.0);
					
			return features;
		}	
	}
		
	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException 
	{
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		
		while (reader.ready()) 
		{
			
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
			
			if(!LabelToNumSet.contains(label) && label.length()>1)
			{
				LabelToNumSet.add(label);
			}
		}
		return labeledInstances;
	}

	private static void testClassifier(
			ProbabilisticClassifier<String, String> classifier,
			List<LabeledInstance<String, String>> testData, boolean verbose) throws IOException 
	{
	
		double numCorrect = 0.0;
		double numTotal = 0.0;
		
		List<RefTargetPair> refTargetPairList = null ;
		eachCategoryMap = new HashMap<Double,List<RefTargetPair>>() ;
		RefTargetPair pair = null ;
	
		
		for (LabeledInstance<String, String> testDatum : testData) 
		{
			String name = testDatum.getInput();
			String label = classifier.getLabel(name);
			double confidence = classifier.getProbabilities(name).getCount(
					label);
			
			double reference = (double)LabelToNumSet.indexOf(label) ;
			
			refTargetPairList = eachCategoryMap.get(reference);
			if ( null == refTargetPairList )
			{
				refTargetPairList = new ArrayList<RefTargetPair>();
			}
			double target = (double) LabelToNumSet.indexOf(testDatum.getLabel() );
			pair = new RefTargetPair(reference,target);
			refTargetPairList.add(pair);
			eachCategoryMap.put((double)reference, refTargetPairList);
	
			
			if (label.equals(testDatum.getLabel())) 
			{
				numCorrect += 1.0;
			}
			else
			{
				if (verbose)
				{
					// display an error
			//		System.err.println("Error:" + "\t" + name + "\t" +"guess=" + "\t" + label +"\t" + 
			//				 "gold=" + "\t" + testDatum.getLabel() + "\t" + "confidence=" + "\t" +
			//				 confidence);
					
				    System.err.println("Error: "+name+" guess="+label+" gold="+testDatum.getLabel()+" confidence="+confidence);
				         
				
					if (testDatum.getLabel().contains("movie"))
					{
		//				System.err.println(name + "\t" + "movie" +"\t" + label);
					}
				}
			}
			
			numTotal += 1.0;
		}
/*	
		prf1Object = new PRF1Measure();
		prf1Object.setPrecRecallObject(eachCategoryMap);
		
		for ( String label : LabelToNumSet)
		{
			String scoreString = prf1Object.calculatePRF1((double)LabelToNumSet.indexOf(label));
			String confMatrixString = prf1Object.getConfusionMatrix((double)LabelToNumSet.indexOf(label));
			System.out.println(label +"\t" + scoreString);
		}
	
		for ( String label : LabelToNumSet)
		{
			String scoreString = prf1Object.calculatePRF1((double)LabelToNumSet.indexOf(label));
			String confMatrixString = prf1Object.getConfusionMatrix((double)LabelToNumSet.indexOf(label));
			System.out.println(label +"\t" + confMatrixString);
		}

	*/	
		double accuracy = numCorrect / numTotal;
		System.out.println("numTotal: " + numTotal + " numCorrect: " + numCorrect + " Accuracy: " + accuracy);
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;
		boolean useValidation = true;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the model to use
		if (argMap.containsKey("-test")) {
			String testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) 
		{
			verbose = true;
		}
		LabelToNumSet = new ArrayList<String>();
		
		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train.txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test-label.txt");

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		if (model.equalsIgnoreCase("baseline")) {
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
					.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("n-gram")) 
		{
			// TODO: construct your n-gram model here
		} 
		else if (model.equalsIgnoreCase("maxent"))
		{
			// TODO: construct your maxent model here
			ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					1.0, 120, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
		}
		
		else if (model.equalsIgnoreCase("perceptron"))
		{
			// TODO: construct your maxent model here
			ProbabilisticClassifierFactory<String, String> factory = new PerceptronClassifier.Factory<String, String, String>(
					1.0, 50, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
		}
		
		//check the character/ngram LM 
		else if (model.equalsIgnoreCase("unigram-word"))
		{
			classifier = new UnigramLabelClassifier.Factory<String,String>().trainClassifier(trainingData);
		}
		
		else if (model.equalsIgnoreCase("unigram-char"))
		{
			classifier = new UniCharactergramLabelClassifier.Factory<String,String>().trainClassifier(trainingData);
		}
		
		else if (model.equalsIgnoreCase("bigram-char"))
		{
			classifier = new BiCharactergramLabelClassifier.Factory<String,String>().trainClassifier(trainingData);
		}
		
		else if (model.equalsIgnoreCase("trigram-char"))
		{
			classifier = new TriCharactergramLabelClassifier.Factory<String,String>().trainClassifier(trainingData);
		}
		
		
		
		else
		{
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Test classifier
		testClassifier(classifier, (useValidation ? validationData : testData),
				verbose);
	}
}
