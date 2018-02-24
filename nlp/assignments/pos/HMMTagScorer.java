package nlp.assignments.pos;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import nlp.assignments.MaximumEntropyClassifier;
import nlp.assignments.pos.POSTaggerTester.LabeledLocalTrigramContext;
import nlp.assignments.pos.POSTaggerTester.LocalTrigramContext;
import nlp.assignments.pos.POSTaggerTester.LocalTrigramScorer;
import nlp.assignments.pos.POSTaggerTester_maxent.HMMTagScorer.UnkFeatExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.Counters;

class HMMTagScorer implements LocalTrigramScorer 
{

	static final String START_WORD = "<S>";
	static final String STOP_WORD = "</S>";
	static final String START_TAG = "<S>";
	static final String STOP_TAG = "</S>";
	
	boolean restrictTrigrams; // if true, assign log score of
								// Double.NEGATIVE_INFINITY to illegal tag
								// trigrams.

	// Emission - P(wi|ti)
	CounterMap<String, String> tagsToWord = new CounterMap<String, String>();
	
	
	
	CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
	Counter<String> unknownWordTags = new Counter<String>();
	
	//map for trigram tags - needed for HMM (P(ti|ti-1, ti-2)
	
	CounterMap<String, String > seenTagTrigrams = new CounterMap<String,String>();
	CounterMap<String,String> seenTagBigrams = new CounterMap<String,String>();
	Counter<String> seenTagUnigrams = new Counter<String>();

	double N = 0.0 ;
	private ProbabilisticClassifier<String, String> classifier;
	private List<String> skipTagsList;
	private List<String> noTrainSymbols;
	
	public int getHistorySize() {
		return 2;
	}

	public Counter<String> getLogScoreCounter(
			LocalTrigramContext localTrigramContext) 
	{
		int position = localTrigramContext.getPosition();
		String word = localTrigramContext.getWords().get(position);
		List<String> words =localTrigramContext.getWords() ;
		Counter<String> tagCounter = unknownWordTags;
		
		//this is inside the trellis 
		//looping into all possible tag sets - so not a "set" of tags are allotted
		//we are only returning the scores from count
		String previousPreviousTag = localTrigramContext.getPreviousPreviousTag();
		String previousTag = localTrigramContext.getPreviousTag();
		
	//	String beforeWord = getBeforeWord(words,position);
	//	String afterWord = getAfterWord(words,position);
		
	//	word = beforeWord + "|" + word + "|" + afterWord ;
		
		boolean seen = false ;
		
		if (wordsToTags.keySet().contains(word)) 
		{
			//word is not unknown!
			seen = true ;
			tagCounter = wordsToTags.getCounter(word);
		}
		
		Set<String> allowedFollowingTags = allowedFollowingTags(
				tagCounter.keySet(),
				localTrigramContext.getPreviousPreviousTag(),
				localTrigramContext.getPreviousTag());
		
		Counter<String> logScoreCounter = new Counter<String>();
		
		
		//check if the word is a given symbol like ","
		
	/*	
		if(noTrainSymbols.contains(word))
		{
			if (!restrictTrigrams || allowedFollowingTags.isEmpty()
					|| allowedFollowingTags.contains(word))
				logScoreCounter.setCount(word, 0.0);
	
				return logScoreCounter ;
		}
	*/		
		
		if (word.equalsIgnoreCase("\""))
		{
			if (!restrictTrigrams || allowedFollowingTags.isEmpty()
					|| allowedFollowingTags.contains(word))
				logScoreCounter.setCount("``", 0.0);
	
				return logScoreCounter ;
		}
		
		if (word.equalsIgnoreCase("#"))
		{
			if (!restrictTrigrams || allowedFollowingTags.isEmpty()
					|| allowedFollowingTags.contains(word))
				logScoreCounter.setCount("$", 0.0);
	
				return logScoreCounter ;
		}
		
		if (word.equalsIgnoreCase("'"))
		{
			if (!restrictTrigrams || allowedFollowingTags.isEmpty()
					|| allowedFollowingTags.contains(word))
				logScoreCounter.setCount("POS", 0.0);
	
				return logScoreCounter ;
		}
		if (word.equalsIgnoreCase("-"))
		{
			if (!restrictTrigrams || allowedFollowingTags.isEmpty()
					|| allowedFollowingTags.contains(word))
				logScoreCounter.setCount("HYPH", 0.0);
	
				return logScoreCounter ;
		}
		
		
		
		for (String tag : tagCounter.keySet()) 
		{
		//	double logScore = Math.log(tagCounter.getCount(tag));
			
			//we do something different than above
			double probEmissionScore = 0.0 ;
			double probInterpolateTransScore = 0.0 ;
			double logScore = 0.0 ;
			
	//		if (word.equalsIgnoreCase("\"") && tag.equalsIgnoreCase("``"))
	//		{
	//			System.out.println("check for quote");
	//		}
			
			if(seen)
			{
				probEmissionScore = tagsToWord.getCount(tag, word);
				if ( probEmissionScore == 0.0 )
				{
					//flag an error?
					System.out.println("check here, word is absent "+ word + " "+tag);
				}
			}
			else
			{
				//check the upper case/lower case
				String lowerWord = word.toLowerCase() ;
				if (position == 0 && tagsToWord.getCount(tag, lowerWord)>0) 
				{
		//			probEmissionScore = tagsToWord.getCount(tag, lowerWord);
				} 
				else
				{
					//maxent classifier 
					try 
					{
						if(skipTagsList.contains(tag))
						{
						//	probEmissionScore = 0.001 ;
							continue ; //check
						}
						else
						{
							probEmissionScore = classifier.getProbabilities(word).getCount(tag);
						}
					}
					catch (IOException e) 
					{
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
		//
		//		probEmissionScore = unknownWordTags.getCount(tag);
			}
				
				if(seenTagTrigrams.containsKey(makeBigramString(previousPreviousTag,previousTag)))
				{
					probInterpolateTransScore = getInterpolatedScore(previousPreviousTag,previousTag,tag,tagCounter);
				}
				else if ( seenTagBigrams.containsKey(previousTag))
				{
					probInterpolateTransScore = getInterpolatedScore(previousTag,tag,tagCounter);
					
				}
				else
				{
					probInterpolateTransScore = getProbScore(tag,tagCounter) ;
				}
			
				logScore = (probInterpolateTransScore) * (probEmissionScore) ;
				logScore = Math.log(logScore);
				
			//	probInterpolateTransScore = getInterpolatedScore(previousPreviousTag,previousTag,tag,tagCounter);
				
			//	logScore = Math.log(probInterpolateTransScore) + Math.log(probEmissionScore) ;
				
			
			if (!restrictTrigrams || allowedFollowingTags.isEmpty()
					|| allowedFollowingTags.contains(tag))
				logScoreCounter.setCount(tag, logScore);
		}
		
		return logScoreCounter;
	}
	
	
	
	private double getProbScore(String tag, 
			Counter<String> tagCounter) {
		// TODO Auto-generated method stub
		return tagCounter.getCount(tag);
	}

	private double getInterpolatedScore(
			String previousTag, String tag,  Counter<String> tagCounter) 
	{
		// TODO Auto-generated method stub
		double lambda1 = 0.97 ;
		double lambda2 = 0.03;
		
		double score1 = lambda1 * seenTagBigrams.getCount(previousTag, tag) ;
		double score2 = lambda2 * seenTagUnigrams.getCount(tag)  ;
		
		return score1+score2;
	}


	private double getInterpolatedScore(String previousPreviousTag,
			String previousTag, String tag,  Counter<String> tagCounter) 
	{
		// TODO Auto-generated method stub
		double lambda1 = .74 ;
		double lambda2 = 0.24 ;
		double lambda3 = 0.02 ;
		
		double score1 = lambda1 * seenTagTrigrams.getCount(makeBigramString(previousPreviousTag,previousTag), tag) ;
		double score2 = lambda2 * seenTagBigrams.getCount(previousTag, tag) ;
		double score3 = lambda3 * tagCounter.getCount(tag)  ;
		
		return score1+score2+score3;
	}

	private Set<String> allowedFollowingTags(Set<String> tags,
			String previousPreviousTag, String previousTag) 
	{
		Set<String> allowedTags = new HashSet<String>();
		for (String tag : tags) 
		{
			String trigramString = makeBigramString(previousPreviousTag,
					previousTag);
			if (seenTagTrigrams.getCount(trigramString,tag)>0) 
			{
				allowedTags.add(tag);
			}
		}
		return allowedTags;
	}
/*
	private String makeTrigramString(String previousPreviousTag,
			String previousTag, String currentTag) {
		return previousPreviousTag + " " + previousTag + " " + currentTag;
	}
*/
	/*
	 * public void train(List<LabeledLocalTrigramContext>
	 * labeledLocalTrigramContexts) { // collect word-tag counts for
	 * (LabeledLocalTrigramContext labeledLocalTrigramContext :
	 * labeledLocalTrigramContexts) { String word =
	 * labeledLocalTrigramContext.getCurrentWord(); String tag =
	 * labeledLocalTrigramContext.getCurrentTag(); if
	 * (!wordsToTags.keySet().contains(word)) { // word is currently
	 * unknown, so tally its tag in the unknown tag counter
	 * unknownWordTags.incrementCount(tag, 1.0); }
	 * wordsToTags.incrementCount(word, tag, 1.0);
	 * seenTagTrigrams.add(makeTrigramString
	 * (labeledLocalTrigramContext.getPreviousPreviousTag(),
	 * labeledLocalTrigramContext.getPreviousTag(),
	 * labeledLocalTrigramContext.getCurrentTag())); } wordsToTags =
	 * Counters.conditionalNormalize(wordsToTags); unknownWordTags =
	 * Counters.normalize(unknownWordTags); }
	 */
	public void train(
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) 
	{
		// collect word-tag counts
		for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) 
		{
			
			String previousPreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
			String previousTag = labeledLocalTrigramContext.getPreviousTag();
			
			String word = labeledLocalTrigramContext.getCurrentWord();
			String tag = labeledLocalTrigramContext.getCurrentTag();
			
			if (!wordsToTags.keySet().contains(word)) 
			{
				// word is currently unknown, so tally its tag in the
				// unknown tag counter
				unknownWordTags.incrementCount(tag, 1.0);
			}
			
			//for building HMM
			tagsToWord.incrementCount(tag, word, 1.0);
			//for building HMM
			
		//	if (word.equalsIgnoreCase("\"") )//&& tag.equalsIgnoreCase("``"))
		//	{
		//		System.out.println("check for quote");
		//	}
			
			
			wordsToTags.incrementCount(word, tag, 1.0);
			seenTagTrigrams.incrementCount(makeBigramString(
					labeledLocalTrigramContext.getPreviousPreviousTag(),
					labeledLocalTrigramContext.getPreviousTag()),
					labeledLocalTrigramContext.getCurrentTag(),1.0);
			
			seenTagBigrams.incrementCount(
					labeledLocalTrigramContext.getPreviousTag(),
					labeledLocalTrigramContext.getCurrentTag(),1.0);
			
		//	seenTagBigrams.incrementCount(
		//			labeledLocalTrigramContext.getPreviousPreviousTag(),
		//			labeledLocalTrigramContext.getPreviousTag(),1.0);
			
		//	seenTagUnigrams.incrementCount(labeledLocalTrigramContext.getPreviousPreviousTag(), 1.0);
		//	seenTagUnigrams.incrementCount(labeledLocalTrigramContext.getPreviousTag(), 1.0);
			seenTagUnigrams.incrementCount(labeledLocalTrigramContext.getCurrentTag(), 1.0);
				
			
		}
		
		for ( String tag : tagsToWord.keySet())
		{
	//		System.out.println(tag +"\t" + tagsToWord.getCounter(tag).totalCount());
			
		}
		
		//training via maxent?
		maxEntTrainer(labeledLocalTrigramContexts);
		//get a sense of tagger/statistics
//		for ( String tag : tagsToWord.keySet())
//		{
//			System.out.println(tag +"\t" + tagsToWord.getCounter(tag).keySet().size());
			
//		}
		
	}
	
	public void maxEntTrainer(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts)
	{
		List<LabeledInstance<String, String>> trainingDataForMaxent =
				new ArrayList<LabeledInstance<String, String>>() ;
		
		double start_timing = System.currentTimeMillis();
			
		CounterMap<String,String> seenWords = new CounterMap<String,String>();
		
		for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) 
		{
			String word = labeledLocalTrigramContext.getCurrentWord();
			String tag = labeledLocalTrigramContext.getCurrentTag();

	//		List<String> words = labeledLocalTrigramContext.getWords();
	//		int position = labeledLocalTrigramContext.getPosition() ;
	//		String beforeWord = getBeforeWord(words,position);
	//		String afterWord = getAfterWord(words,position);
			
			
			//we can filter the uncommon tags - but first run everything!
			//note - running for all words make it slow
			//check for fewer/unpopular words
			if (seenWords.getCount(tag, word) > 0)
			{
				continue;
			}
			else
			{
				seenWords.incrementCount(tag, word, 1.0);
			 }
			
			
		//	if ( seenTagUnigrams.getCount(word) > 10)
		//	{
		//		continue ;
		//	}
			if(skipTagsList.contains(tag))
			{
				continue ; //check
			}
		
		//the next 3 lines is the change	
		//	String prevPreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
		///	String previousTag = labeledLocalTrigramContext.getPreviousTag();
	//		word = beforeWord + "|" + word + "|" + afterWord ;
			
			trainingDataForMaxent.add(new LabeledInstance<String, String>(tag, word));
		}
		
		double before_timing = System.currentTimeMillis();
		 System.out.println("max ent training will start: " + " time took: " + (before_timing-start_timing));
			
		
		//fire a probabilistic classifier with new features
		  ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(1.0, 20, new POSUnknownFeatureExtractor());
		 try {
			classifier = factory.trainClassifier(trainingDataForMaxent);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double stop_timing = System.currentTimeMillis();
			
		 
		 System.out.println("max ent training finished: " + " time took: " + (stop_timing-before_timing));
		 
		  
		 
	}

	private String getAfterWord(List<String> words, int position) 
	{
		// TODO Auto-generated method stub
		if ( position == words.size()-1)
		{
			return STOP_WORD ;
		}
		else
		{
			return words.get(position+1);
		}
		
	}

	private String getBeforeWord(List<String> words, int position) 
	{
		// TODO Auto-generated method stub
		if ( position == 0 )
		{
			return START_WORD ;
		}
		else
		{
			return words.get(position-1) ;
		}
		
		
	}

	public void validate(
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) 
	{
		// no tuning for this dummy model!
		//lets tune as per the paper - TnT
		//for trigram!
		validateForTrigram(labeledLocalTrigramContexts) ;
		validateForBigram(labeledLocalTrigramContexts) ;
		
		normalize();
		
	}
	
	private void normalize() 
	{
		// TODO Auto-generated method stub
		wordsToTags = Counters.conditionalNormalize(wordsToTags);
		unknownWordTags = Counters.normalize(unknownWordTags);
	
		//normalize HMM
		
		tagsToWord = Counters.conditionalNormalize(tagsToWord);
		seenTagUnigrams = Counters.normalize(seenTagUnigrams);
		seenTagBigrams = Counters.conditionalNormalize(seenTagBigrams);
		seenTagTrigrams = Counters.conditionalNormalize(seenTagTrigrams);
		
	}

	public void validateForBigram(
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) 
	{
		double lambda1 = 0.0 ;
		double lambda2 = 0.0 ;
		
		double totalTokens = wordsToTags.totalCount();
		
		
		for( LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts )
		{
			String t2 = labeledLocalTrigramContext.getPreviousTag() ;
			String t3 = labeledLocalTrigramContext.getCurrentTag() ;
			
			//we need to get the count of f(t2,t3)
			//we need to get the count of f(t3)
			
			double bigramCount = seenTagBigrams.getCounter(t2).totalCount() ;
			double unigramCount = 0.0 ;
			
			unigramCount = seenTagUnigrams.getCount(t2) ;
			
			double case2 = (bigramCount-1)/(unigramCount-1) ;
			
			unigramCount = seenTagUnigrams.getCount(t3) ;
			
		//	double N = wordsToTags.totalCount();
			double case3 = (unigramCount-1)/(totalTokens-1) ;
			
			List<Double> localList = new ArrayList<Double>();
			localList.add(case2);
			localList.add(case3);
			
			java.util.Collections.sort(localList);
			if ( localList.get(1) == case2)
			{
				lambda2 = lambda2 + bigramCount ;
			}
			else if ( localList.get(1) == case3)
			{
				lambda1 = lambda1 + bigramCount ;
			}
		}
		
		double total = lambda1 + lambda2  ;
		lambda1 = lambda1/total ;
		lambda2 = lambda2/total ;
		
		System.out.println(" validation/normalization done for bigrams ") ;
		System.out.println(" validation/normalization scores: " + "lambda1: " + lambda1) ;
		System.out.println(" validation/normalization scores: " + "lambda2: " + lambda2) ;
		
	}

	
	public void validateForTrigram(
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) 
	{
		double lambda1 = 0.0 ;
		double lambda2 = 0.0 ;
		double lambda3 = 0.0 ;
		
		double totalTokens = wordsToTags.totalCount();
		
		
		for( LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts )
		{
			String t1 = labeledLocalTrigramContext.getPreviousPreviousTag() ;
			String t2 = labeledLocalTrigramContext.getPreviousTag() ;
			String t3 = labeledLocalTrigramContext.getCurrentTag() ;
			
			//we need to get the count of f(t1,t2,t3) 
			//we need to get the count of f(t2,t3)
			//we need to get the count of f(t3)
			
			double trigramCount = seenTagTrigrams.getCount(makeBigramString(t1,t2),t3)  ;
			double bigramCount = seenTagTrigrams.getCounter(makeBigramString(t1,t2)).totalCount();
			double case1 = (trigramCount-1)/(bigramCount-1) ;
			
			
			bigramCount = seenTagTrigrams.getCounter(makeBigramString(t2,t3)).totalCount() ;
			double unigramCount = 0.0 ;
			
			unigramCount = seenTagUnigrams.getCount(t2) ;
			
			double case2 = (bigramCount-1)/(unigramCount-1) ;
			
			unigramCount = seenTagUnigrams.getCount(t3) ;
			
		//	double N = wordsToTags.totalCount();
			double case3 = (unigramCount-1)/(totalTokens-1) ;
			
			List<Double> localList = new ArrayList<Double>();
			localList.add(case1);
			localList.add(case2);
			localList.add(case3);
			
			java.util.Collections.sort(localList);
			if ( localList.get(2) == case1)
			{
				lambda3 = lambda3 + trigramCount ;
			}
			else if ( localList.get(2) == case2)
			{
				lambda2 = lambda2 + trigramCount ;
			}
			else if ( localList.get(2) == case3)
			{
				lambda1 = lambda1 + trigramCount ;
			}
		}
		
		double total = lambda1 + lambda2 + lambda3 ;
		lambda1 = lambda1/total ;
		lambda2 = lambda2/total ;
		lambda3 = lambda3/total ;
		
		System.out.println(" validation/normalization done for trigrams ") ;
		System.out.println(" validation/normalization scores: " + "lambda1: " + lambda1) ;
		System.out.println(" validation/normalization scores: " + "lambda2: " + lambda2) ;
		System.out.println(" validation/normalization scores: " + "lambda3: " + lambda3) ;
		
	}

	private boolean startStopTag(String tag) 
	{
		// TODO Auto-generated method stub
		if ( tag.equalsIgnoreCase(START_TAG) || tag.equalsIgnoreCase(STOP_TAG))
		{
			return true ;
		}
		return false;
	}

	public HMMTagScorer(boolean restrictTrigrams) 
	{
		this.restrictTrigrams = restrictTrigrams;
		String[] skipTags = {".", "SYM", "-RRB-", "-LRB-", ",", "$",  "#", "$", "TO", "MD",
				"</S>", "PRP$", "WP", "WP$", "WDT", "PDT", "RBS", "WRB", "POS", "HYPH", "NFP", "AFX",
				"(", ")",":","DT","CC","PRP", "''","``","\""};
		
		skipTagsList = Arrays.asList(skipTags);
		
		String[] symbols = {".", ":", STOP_WORD, "``", "''", ",", "$", "#"}; 
		noTrainSymbols = Arrays.asList(symbols);
	}
	
	private String makeBigramString(String previousTag, String currentTag) {
		return  previousTag + "_" + currentTag;
	}
	
}
