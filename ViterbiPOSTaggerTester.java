package nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

import javax.xml.stream.events.EndDocument;

import nlp.io.PennTreebankReader;
import nlp.ling.Tree;
import nlp.ling.Trees;
import nlp.util.*;

/**
 * Harness for POS Tagger project.
 */
public class ViterbiPOSTaggerTester {

	static final String START_WORD = "<S>";
	static final String STOP_WORD = "</S>";
	static final String START_TAG = "<S>";
	static final String STOP_TAG = "</S>";

	/**
	 * Tagged sentences are a bundling of a list of words and a list of their
	 * tags.
	 */
	static class TaggedSentence {
		List<String> words;
		List<String> tags;

		public int size() {
			return words.size();
		}

		public List<String> getWords() {
			return words;
		}

		public List<String> getTags() {
			return tags;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int position = 0; position < words.size(); position++) {
				String word = words.get(position);
				String tag = tags.get(position);
				sb.append(word);
				sb.append("_");
				sb.append(tag);
			}
			return sb.toString();
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof TaggedSentence))
				return false;

			final TaggedSentence taggedSentence = (TaggedSentence) o;

			if (tags != null ? !tags.equals(taggedSentence.tags)
					: taggedSentence.tags != null)
				return false;
			if (words != null ? !words.equals(taggedSentence.words)
					: taggedSentence.words != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = (words != null ? words.hashCode() : 0);
			result = 29 * result + (tags != null ? tags.hashCode() : 0);
			return result;
		}

		public TaggedSentence(List<String> words, List<String> tags) {
			this.words = words;
			this.tags = tags;
		}
	}

	/**
	 * States are pairs of tags along with a position index, representing the
	 * two tags preceding that position. So, the START state, which can be
	 * gotten by State.getStartState() is [START, START, 0]. To build an
	 * arbitrary state, for example [DT, NN, 2], use the static factory method
	 * State.buildState("DT", "NN", 2). There isnt' a single final state, since
	 * sentences lengths vary, so State.getEndState(i) takes a parameter for the
	 * length of the sentence.
	 */
	static class State {

		private static transient Interner<State> stateInterner = new Interner<State>(
				new Interner.CanonicalFactory<State>() {
					public State build(State state) {
						return new State(state);
					}
				});

		private static transient State tempState = new State();

		public static State getStartState() {
			return buildState(START_TAG, START_TAG, 0);
		}

		public static State getStopState(int position) {
			return buildState(STOP_TAG, STOP_TAG, position);
		}

		public static State buildState(String previousPreviousTag,
				String previousTag, int position) {
			tempState.setState(previousPreviousTag, previousTag, position);
			return stateInterner.intern(tempState);
		}

		public static List<String> toTagList(List<State> states) {
			List<String> tags = new ArrayList<String>();
			if (states.size() > 0) {
				tags.add(states.get(0).getPreviousPreviousTag());
				for (State state : states) {
					tags.add(state.getPreviousTag());
				}
			}
			return tags;
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public State getNextState(String tag) {
			return State.buildState(getPreviousTag(), tag, getPosition() + 1);
		}

		public State getPreviousState(String tag) {
			return State.buildState(tag, getPreviousPreviousTag(),
					getPosition() - 1);
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof State))
				return false;

			final State state = (State) o;

			if (position != state.position)
				return false;
			if (previousPreviousTag != null ? !previousPreviousTag
					.equals(state.previousPreviousTag)
					: state.previousPreviousTag != null)
				return false;
			if (previousTag != null ? !previousTag.equals(state.previousTag)
					: state.previousTag != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = position;
			result = 29 * result
					+ (previousTag != null ? previousTag.hashCode() : 0);
			result = 29
					* result
					+ (previousPreviousTag != null ? previousPreviousTag
							.hashCode() : 0);
			return result;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getPosition() + "]";
		}

		int position;
		String previousTag;
		String previousPreviousTag;

		private void setState(String previousPreviousTag, String previousTag,
				int position) {
			this.previousPreviousTag = previousPreviousTag;
			this.previousTag = previousTag;
			this.position = position;
		}

		private State() {
		}

		private State(State state) {
			setState(state.getPreviousPreviousTag(), state.getPreviousTag(),
					state.getPosition());
		}
	}

	/**
	 * A Trellis is a graph with a start state an an end state, along with
	 * successor and predecessor functions.
	 */
	static class Trellis<S> {
		S startState;
		S endState;
		CounterMap<S, S> forwardTransitions;
		CounterMap<S, S> backwardTransitions;

		/**
		 * Get the unique start state for this trellis.
		 */
		public S getStartState() {
			return startState;
		}

		public void setStartState(S startState) {
			this.startState = startState;
		}

		/**
		 * Get the unique end state for this trellis.
		 */
		public S getEndState() {
			return endState;
		}

		public void setStopState(S endState) {
			this.endState = endState;
		}

		/**
		 * For a given state, returns a counter over what states can be next in
		 * the markov process, along with the cost of that transition. Caution:
		 * a state not in the counter is illegal, and should be considered to
		 * have cost Double.NEGATIVE_INFINITY, but Counters score items they
		 * don't contain as 0.
		 */
		public Counter<S> getForwardTransitions(S state) {
			return forwardTransitions.getCounter(state);

		}

		/**
		 * For a given state, returns a counter over what states can precede it
		 * in the markov process, along with the cost of that transition.
		 */
		public Counter<S> getBackwardTransitions(S state) {
			return backwardTransitions.getCounter(state);
		}

		public void setTransitionCount(S start, S end, double count) {
			forwardTransitions.setCount(start, end, count);
			backwardTransitions.setCount(end, start, count);
		}

		public Trellis() {
			forwardTransitions = new CounterMap<S, S>();
			backwardTransitions = new CounterMap<S, S>();
		}
	}

	/**
	 * A TrellisDecoder takes a Trellis and returns a path through that trellis
	 * in which the first item is trellis.getStartState(), the last is
	 * trellis.getEndState(), and each pair of states is connected in the
	 * trellis.
	 */
	static interface TrellisDecoder<S> {
		List<S> getBestPath(Trellis<S> trellis);
	}

	static class GreedyDecoder<S> implements TrellisDecoder<S> 
	{
		public List<S> getBestPath(Trellis<S> trellis) 
		{
			List<S> states = new ArrayList<S>();
			S currentState = trellis.getStartState();
			states.add(currentState);
			while (!currentState.equals(trellis.getEndState())) 
			{
				Counter<S> transitions = trellis
						.getForwardTransitions(currentState);
				S nextState = transitions.argMax();
				states.add(nextState);
				currentState = nextState;
			}
			return states;
		}
	}
	
	static class HMMDecoder<S> implements TrellisDecoder<S> 
	{
		//an inefficient HMM - 
		public List<S> getBestPath(Trellis<S> trellis) 
		{
			List<S> states = new ArrayList<S>();
			S currentState = trellis.getStartState();
			states.add(currentState);
			while (!currentState.equals(trellis.getEndState())) 
			{
				Counter<S> transitions = trellis
						.getForwardTransitions(currentState);
				S nextState = transitions.argMax();
				states.add(nextState);
				currentState = nextState;
			}
			return states;
		}
	
		
    
       
	}
	
	static class ViterbiCheckDecoder<S> implements TrellisDecoder<S> 
	{

		@Override
		public List<S> getBestPath(Trellis<S> trellis) 
		{
			// TODO Auto-generated method stub
			 List<S> states = new ArrayList<S>();
             S currentState = trellis.getStartState();
             Counter<S> viterbi = new Counter<S>();
             viterbi.setCount(currentState, 0);
             Map<S, S> backPointer = new HashMap<S, S>();
             backPointer.put(currentState, currentState);
             while (!currentState.equals(trellis.getEndState())) 
             {
                     Counter<S> forwardTransitions = trellis.getForwardTransitions(currentState);
                     recursive(trellis, viterbi, backPointer, forwardTransitions);
//                   currentState = (S)forwardTransitions.keySet().iterator().next();
                     for (S forwardState : forwardTransitions.keySet()) 
                     {
                             currentState = forwardState;
                             recursive(trellis, viterbi, backPointer, trellis.getForwardTransitions(currentState));
                     }
             }
             
             S backState = trellis.getEndState();
             S startState = trellis.getStartState();
         	StringBuffer ret = new StringBuffer();
    		
             while (!backState.equals(startState)) 
             {
                     states.add(0, backState);
                     ret.append(backState.toString()+" --> ");
                     backState = backPointer.get(backState);
//                   System.out.println(backState);
             }
             states.add(0, startState);
             ret.append(startState);
             System.out.println(ret.toString());
             
             return states;
		}
		
		 void recursive(Trellis<S> trellis, Counter<S> viterbi,
                 Map<S, S> backPointer, Counter<S> forwardTransitions) 
                 {
         for (S forwardState : forwardTransitions.keySet()) 
         {
                 Counter<S> backwardTransitions = trellis.getBackwardTransitions(forwardState);
                 double max = Double.NEGATIVE_INFINITY;
                 double temp = 0;//Double.NEGATIVE_INFINITY;
                 S backwardMax = null;
                 for (S backwardState : backwardTransitions.keySet()) 
                 {
                         double temp1 = viterbi.containsKey(backwardState) ? viterbi.getCount(backwardState) : Double.NEGATIVE_INFINITY;
                         double temp2 = backwardTransitions.getCount(backwardState);
//                       temp = viterbi.getCount(backwardState) + backwardTransitions.getCount(backwardState);
                         temp = temp1 + temp2;
//                       System.out.println(backwardState + ", " + forwardState + ", " + temp1 + ", " + temp2);
                         if (temp > max)
                          {
                                         max = temp;
                                         backwardMax = backwardState;
                     		}
                 }
                 
                 viterbi.setCount(forwardState, max);
                 backPointer.put(forwardState, backwardMax);
//                       System.out.println(forwardState + ", " + backwardMax);
         }
 }
		
	}
	
	static class ViterbiDecoder<S> implements TrellisDecoder<S> 
	{
		//viterbi decoder - 
		
		public List<S> getBestPath(Trellis<S> trellis) 
		{
			List<S> states = new ArrayList<S>();
			S currentState = trellis.getStartState();
			states.add(currentState);
			
			//we need to keep the max score for each state
			Counter<S> viterbiMax = new Counter<S>();
			viterbiMax.setCount(currentState, 0);
			
			//we also need to keep the state-to-state map that gave the "max" probability till this point
			Map<S,S> statePerStateMaxMap = new HashMap<S,S>();
			//init with current state
			statePerStateMaxMap.put(currentState, currentState);
            
			while (!currentState.equals(trellis.getEndState())) 
			{
				//for t = 1 we need to calculate separately so that
				//we use <S> or the zeroth state
				Counter<S> forwardTransitions = trellis
						.getForwardTransitions(currentState);
			
				//this is t=1
				recursion(trellis,forwardTransitions,viterbiMax,statePerStateMaxMap);
				
				//now do t = 2...n
				for ( S fwdState : forwardTransitions.keySet())
				{
					currentState = fwdState ;
					forwardTransitions = trellis
							.getForwardTransitions(currentState);
					
					recursion(trellis,forwardTransitions,viterbiMax,statePerStateMaxMap);
					
				}
				
				
			}
			
			//now backtrack all states till we hit the first state
			S lastState = trellis.getEndState();
			S firstState = trellis.getStartState();
			
			StringBuffer ret = new StringBuffer();
			while (!lastState.equals(firstState) )
			{
				states.add(0,lastState);
				ret.append(lastState.toString()+" --> ");
				lastState = statePerStateMaxMap.get(lastState);
			
			}
			states.add(0,firstState);
			ret.append(firstState.toString());
	//		System.out.println(ret);
			return states;
		}
		
		private void recursion(Trellis<S> trellis,
				Counter<S> forwardTransitions, Counter<S> viterbiMax,
				Map<S, S> statePerStateMaxMap) 
		{
			// TODO Auto-generated method stub
			for ( S fwdState : forwardTransitions.keySet() )
			{
					Counter<S> backwardStates =	trellis.getBackwardTransitions(fwdState);
					
					double localMaxScore = Double.NEGATIVE_INFINITY ;
					S maxBwdState = null ;
					
					for ( S bwdState : backwardStates.keySet())
					{
						//we need two scores (previous max and current emission/transmission)
						double score1 = backwardStates.getCount(bwdState);
						double score2 = viterbiMax.getCount(bwdState) ;
						double totalScore = score1+score2 ;
						if ( totalScore > localMaxScore )
						{
							localMaxScore = totalScore ;
							maxBwdState = bwdState ;
						}
					}
					statePerStateMaxMap.put(fwdState, maxBwdState);
					viterbiMax.setCount(fwdState,localMaxScore);
			}
		}



		private Double calcMaxPreviousState(Trellis<S> trellis, Entry<S, Double> fwdTrans, Counter<S> viterbiMax) 
		{
			// TODO Auto-generated method stub
			Counter<S> backwardTrans = trellis.getBackwardTransitions((S) fwdTrans) ;
			
			double maxScore = Double.NEGATIVE_INFINITY ;
			double localScore = 0.0 ;
			S backwardMax = null;
			for ( S backTran : backwardTrans.keySet())
			{
				double score1 = backwardTrans.getCount(backTran);
				double score2 = viterbiMax.getCount(backTran) ;
				if ( score2 == 0 )
				{
					score2 = Double.NEGATIVE_INFINITY ;
				}
				localScore = score1+score2 ;
				if ( localScore > maxScore )
				{
					maxScore = localScore ;
					backwardMax = backTran ;
				}
			}
			viterbiMax.incrementCount(backwardMax, maxScore);
			
			return null;
		}
	
		
    
       
	}
	
	
	

	static class POSTagger {

		LocalTrigramScorer localTrigramScorer;
		TrellisDecoder<State> trellisDecoder;

		// chop up the training instances into local contexts and pass them on
		// to the local scorer.
		public void train(List<TaggedSentence> taggedSentences) {
			localTrigramScorer
					.train(extractLabeledLocalTrigramContexts(taggedSentences));
		}

		// chop up the validation instances into local contexts and pass them on
		// to the local scorer.
		public void validate(List<TaggedSentence> taggedSentences) {
			localTrigramScorer
					.validate(extractLabeledLocalTrigramContexts(taggedSentences));
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				List<TaggedSentence> taggedSentences) {
			List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			for (TaggedSentence taggedSentence : taggedSentences) {
				localTrigramContexts
						.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
			}
			return localTrigramContexts;
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				TaggedSentence taggedSentence) {
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			List<String> words = new BoundedList<String>(
					taggedSentence.getWords(), START_WORD, STOP_WORD);
			List<String> tags = new BoundedList<String>(
					taggedSentence.getTags(), START_TAG, STOP_TAG);
			for (int position = 0; position <= taggedSentence.size() + 1; position++) {
				labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(
						words, position, tags.get(position - 2), tags
								.get(position - 1), tags.get(position)));
			}
			return labeledLocalTrigramContexts;
		}

		/**
		 * Builds a Trellis over a sentence, by starting at the state State, and
		 * advancing through all legal extensions of each state already in the
		 * trellis. You should not have to modify this code (or even read it,
		 * really).
		 */
		private Trellis<State> buildTrellis(List<String> sentence) {
			Trellis<State> trellis = new Trellis<State>();
			trellis.setStartState(State.getStartState());
			State stopState = State.getStopState(sentence.size() + 2);
			trellis.setStopState(stopState);
			Set<State> states = Collections.singleton(State.getStartState());
			for (int position = 0; position <= sentence.size() + 1; position++) {
				Set<State> nextStates = new HashSet<State>();
				for (State state : states) {
					if (state.equals(stopState)) {
						continue;
					}

					LocalTrigramContext localTrigramContext = new LocalTrigramContext(
							sentence, position, state.getPreviousPreviousTag(),
							state.getPreviousTag());
					Counter<String> tagScores = localTrigramScorer
							.getLogScoreCounter(localTrigramContext);
					for (String tag : tagScores.keySet()) {
						double score = tagScores.getCount(tag);
						State nextState = state.getNextState(tag);
						trellis.setTransitionCount(state, nextState, score);
						nextStates.add(nextState);
					}
				}
				// System.out.println("States: "+nextStates);
				states = nextStates;
			}
			return trellis;
		}

		// to tag a sentence: build its trellis and find a path through that
		// trellis
		public List<String> tag(List<String> sentence) 
		{
			Trellis<State> trellis = buildTrellis(sentence);
			List<State> states = trellisDecoder.getBestPath(trellis);
			List<String> tags = State.toTagList(states);
			tags = stripBoundaryTags(tags);
			return tags;
		}

		/**
		 * Scores a tagging for a sentence. Note that a tag sequence not
		 * accepted by the markov process should receive a log score of
		 * Double.NEGATIVE_INFINITY.
		 */
		public double scoreTagging(TaggedSentence taggedSentence) {
			double logScore = 0.0;
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(taggedSentence);
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				Counter<String> logScoreCounter = localTrigramScorer
						.getLogScoreCounter(labeledLocalTrigramContext);
				String currentTag = labeledLocalTrigramContext.getCurrentTag();
				if (logScoreCounter.containsKey(currentTag)) {
					logScore += logScoreCounter.getCount(currentTag);
				} else {
					logScore += Double.NEGATIVE_INFINITY;
				}
			}
			return logScore;
		}

		private List<String> stripBoundaryTags(List<String> tags) {
			return tags.subList(2, tags.size() - 2);
		}

		public POSTagger(LocalTrigramScorer localTrigramScorer,
				TrellisDecoder<State> trellisDecoder) {
			this.localTrigramScorer = localTrigramScorer;
			this.trellisDecoder = trellisDecoder;
		}
	}

	/**
	 * A LocalTrigramContext is a position in a sentence, along with the
	 * previous two tags -- basically a FeatureVector.
	 */
	static class LocalTrigramContext {
		List<String> words;
		int position;
		String previousTag;
		String previousPreviousTag;

		public List<String> getWords() {
			return words;
		}

		public String getCurrentWord() {
			return words.get(position);
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getCurrentWord() + "]";
		}

		public LocalTrigramContext(List<String> words, int position,
				String previousPreviousTag, String previousTag) {
			this.words = words;
			this.position = position;
			this.previousTag = previousTag;
			this.previousPreviousTag = previousPreviousTag;
		}
	}

	/**
	 * A LabeledLocalTrigramContext is a context plus the correct tag for that
	 * position -- basically a LabeledFeatureVector
	 */
	static class LabeledLocalTrigramContext extends LocalTrigramContext {
		String currentTag;

		public String getCurrentTag() {
			return currentTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
		}

		public LabeledLocalTrigramContext(List<String> words, int position,
				String previousPreviousTag, String previousTag,
				String currentTag) {
			super(words, position, previousPreviousTag, previousTag);
			this.currentTag = currentTag;
		}
	}

	/**
	 * LocalTrigramScorers assign scores to tags occuring in specific
	 * LocalTrigramContexts.
	 */
	static interface LocalTrigramScorer {
		/**
		 * The Counter returned should contain log probabilities, meaning if all
		 * values are exponentiated and summed, they should sum to one. For
		 * efficiency, the Counter can contain only the tags which occur in the
		 * given context with non-zero model probability.
		 */
		Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext);

		void train(List<LabeledLocalTrigramContext> localTrigramContexts);

		void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
	}

	/**
	 * The MostFrequentTagScorer gives each test word the tag it was seen with
	 * most often in training (or the tag with the most seen word types if the
	 * test word is unseen in training. This scorer actually does a little more
	 * than its name claims -- if constructed with restrictTrigrams = true, it
	 * will forbid illegal tag trigrams, otherwise it makes no use of tag
	 * history information whatsoever.
	 */
	static class MostFrequentTagScorer implements LocalTrigramScorer {

		boolean restrictTrigrams; // if true, assign log score of
									// Double.NEGATIVE_INFINITY to illegal tag
									// trigrams.

		//map for trigram tags - needed for HMM (P(ti|ti-1, ti-2)
		CounterMap<String, String> trigramTagCounter = new CounterMap<String, String>();
		CounterMap<String, String> bigramTagCounter = new CounterMap<String, String>() ;

		// Emission - P(wi|ti)
		CounterMap<String, String> tagsToWord = new CounterMap<String, String>();
		
		
		CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
		Counter<String> unknownWordTags = new Counter<String>();
		Set<String> seenTagTrigrams = new HashSet<String>();
		
		Set<String> seenTagBigrams = new HashSet<String>();
		
		

		
		public int getHistorySize() {
			return 2;
		}

		public Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext) 
		{
			int position = localTrigramContext.getPosition();
			String word = localTrigramContext.getWords().get(position);
			Counter<String> tagCounter = unknownWordTags;
			if (wordsToTags.keySet().contains(word)) 
			{
				tagCounter = wordsToTags.getCounter(word);
			}
			
			Set<String> allowedFollowingTags = allowedFollowingTags(
					tagCounter.keySet(),
					localTrigramContext.getPreviousPreviousTag(),
					localTrigramContext.getPreviousTag());
			
			Counter<String> logScoreCounter = new Counter<String>();
			for (String tag : tagCounter.keySet()) 
			{
				double logScore = Math.log(tagCounter.getCount(tag));
				if (!restrictTrigrams || allowedFollowingTags.isEmpty()
						|| allowedFollowingTags.contains(tag))
					logScoreCounter.setCount(tag, logScore);
			}
			
			return logScoreCounter;
		}

		private Set<String> allowedFollowingTags(Set<String> tags,
				String previousPreviousTag, String previousTag) {
			Set<String> allowedTags = new HashSet<String>();
			for (String tag : tags) {
				String trigramString = makeTrigramString(previousPreviousTag,
						previousTag, tag);
				if (seenTagTrigrams.contains((trigramString))) {
					allowedTags.add(tag);
				}
			}
			return allowedTags;
		}

		private String makeTrigramString(String previousPreviousTag,
				String previousTag, String currentTag) {
			return previousPreviousTag + " " + previousTag + " " + currentTag;
		}
		
		

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
				trigramTagCounter.incrementCount(previousPreviousTag+"_"+previousTag,tag,1.0);
				bigramTagCounter.incrementCount(previousTag,tag,1.0);
				tagsToWord.incrementCount(tag, word, 1.0);
				//for building HMM
				
				wordsToTags.incrementCount(word, tag, 1.0);
				seenTagTrigrams.add(makeTrigramString(
						labeledLocalTrigramContext.getPreviousPreviousTag(),
						labeledLocalTrigramContext.getPreviousTag(),
						labeledLocalTrigramContext.getCurrentTag()));
				
						
			}
			wordsToTags = Counters.conditionalNormalize(wordsToTags);
			unknownWordTags = Counters.normalize(unknownWordTags);
		
			//normalize HMM
			tagsToWord = Counters.conditionalNormalize(tagsToWord);
			
			trigramTagCounter = Counters.conditionalNormalize(trigramTagCounter);
			
			bigramTagCounter = Counters.conditionalNormalize(bigramTagCounter);
			
		/*	
			for (String prevPreviousTrigram : trigramTagCounter.keySet()) 
			{
				trigramTagCounter.getCounter(prevPreviousTrigram).normalize();
			}
			for (String previousBigram : bigramTagCounter.keySet()) 
			{
				bigramTagCounter.getCounter(previousBigram).normalize();
			}
		*/	
		}

		public void validate(
				List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			// no tuning for this dummy model!
		}

		public MostFrequentTagScorer(boolean restrictTrigrams) {
			this.restrictTrigrams = restrictTrigrams;
		}
	}
	
	static class HMMTagScorer implements LocalTrigramScorer {

		boolean restrictTrigrams; // if true, assign log score of
									// Double.NEGATIVE_INFINITY to illegal tag
									// trigrams.

		//map for trigram tags - needed for HMM (P(ti|ti-1, ti-2)
		CounterMap<String, String> trigramTagCounter = new CounterMap<String, String>();
		CounterMap<String, String> bigramTagCounter = new CounterMap<String, String>() ;

		// Emission - P(wi|ti)
		CounterMap<String, String> tagsToWord = new CounterMap<String, String>();
		
		
		CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
		Counter<String> unknownWordTags = new Counter<String>();
		Set<String> seenTagTrigrams = new HashSet<String>();
		Set<String> seenTagBigrams = new HashSet<String>();

		
		
		public int getHistorySize() {
			return 2;
		}

		public Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext) 
		{
			int position = localTrigramContext.getPosition();
			String word = localTrigramContext.getWords().get(position);
			Counter<String> tagCounter = unknownWordTags;
			
			//this is inside the trellis 
			//looping into all possible tag sets - so not a "set" of tags are allotted
			//we are only returning the scores from count
			String previousPreviousTag = localTrigramContext.getPreviousPreviousTag();
			String previousTag = localTrigramContext.getPreviousTag();
			
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
			for (String tag : tagCounter.keySet()) 
			{
			//	double logScore = Math.log(tagCounter.getCount(tag));
				
				//we do something different than above
				double probEmissionScore = 0.0 ;
				double probInterpolateTransScore = 0.0 ;
				double logScore = 0.0 ;
				
				if(seen)
				{
					probEmissionScore = tagsToWord.getCount(tag, word);
					if ( probEmissionScore == 0.0 )
					{
						//flag an error?
						System.out.println("check here, word is absent "+ word + " "+tag);
					}
					
					if(allowedFollowingTags.contains(tag))
					{
						probInterpolateTransScore = getInterpolatedScore(previousPreviousTag,previousTag,tag,tagCounter);
					}
					else if ( seenTagBigrams.contains(makeBigramString(previousTag,tag)))
					{
						probInterpolateTransScore = getInterpolatedScore(previousTag,tag,tagCounter);
						
					}
					else
					{
						probInterpolateTransScore = getProbScore(tag,tagCounter) ;
					}
					
					logScore = Math.log(probInterpolateTransScore) + Math.log(probEmissionScore) ;
					
				}
				else
				{
					logScore = Math.log(tagCounter.getCount(tag));
					
				}
			
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
			double lambda1 = 1.0 ;
			double lambda2 = 0.0 ;
			
			double score1 = lambda1 * bigramTagCounter.getCount(previousTag, tag) ;
			double score2 = lambda2 * tagCounter.getCount(tag)  ;
			
			return score1+score2;
		}


		private double getInterpolatedScore(String previousPreviousTag,
				String previousTag, String tag,  Counter<String> tagCounter) 
		{
			// TODO Auto-generated method stub
			double lambda1 = 1.0 ;
			double lambda2 = 0.0 ;
			double lambda3 = 0.0 ;
			
			double score1 = lambda1 * trigramTagCounter.getCount(previousPreviousTag+"_"+previousTag, tag) ;
			double score2 = lambda2 * bigramTagCounter.getCount(previousTag, tag) ;
			double score3 = lambda3 * tagCounter.getCount(tag)  ;
			
			return score1+score2+score3;
		}

		private Set<String> allowedFollowingTags(Set<String> tags,
				String previousPreviousTag, String previousTag) 
		{
			Set<String> allowedTags = new HashSet<String>();
			for (String tag : tags) 
			{
				String trigramString = makeTrigramString(previousPreviousTag,
						previousTag, tag);
				if (seenTagTrigrams.contains((trigramString))) 
				{
					allowedTags.add(tag);
				}
			}
			return allowedTags;
		}

		private String makeTrigramString(String previousPreviousTag,
				String previousTag, String currentTag) {
			return previousPreviousTag + " " + previousTag + " " + currentTag;
		}

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
				trigramTagCounter.incrementCount(previousPreviousTag+"_"+previousTag,tag,1.0);
				bigramTagCounter.incrementCount(previousTag,tag,1.0);
				tagsToWord.incrementCount(tag, word, 1.0);
				//for building HMM
				
				wordsToTags.incrementCount(word, tag, 1.0);
				seenTagTrigrams.add(makeTrigramString(
						labeledLocalTrigramContext.getPreviousPreviousTag(),
						labeledLocalTrigramContext.getPreviousTag(),
						labeledLocalTrigramContext.getCurrentTag()));
				
				seenTagBigrams.add(makeBigramString(
						labeledLocalTrigramContext.getPreviousTag(),
						labeledLocalTrigramContext.getCurrentTag()));
			}
			wordsToTags = Counters.conditionalNormalize(wordsToTags);
			unknownWordTags = Counters.normalize(unknownWordTags);
		
			//normalize HMM
			tagsToWord = Counters.conditionalNormalize(tagsToWord);
			for (String prevPreviousTrigram : trigramTagCounter.keySet()) 
			{
				trigramTagCounter.getCounter(prevPreviousTrigram).normalize();
			}
			for (String previousBigram : bigramTagCounter.keySet()) 
			{
				bigramTagCounter.getCounter(previousBigram).normalize();
			}
			
		}

		public void validate(
				List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			// no tuning for this dummy model!
		}

		public HMMTagScorer(boolean restrictTrigrams) {
			this.restrictTrigrams = restrictTrigrams;
		}
		
		private String makeBigramString(String previousTag, String currentTag) {
			return  previousTag + " " + currentTag;
		}
		
	}
	

	private static List<TaggedSentence> readTaggedSentences(String path,
			boolean hasTags) throws Exception {
		List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
		BufferedReader reader = new BufferedReader(new FileReader(path));
		String line = "";
		List<String> words = new LinkedList<String>();
		List<String> tags = new LinkedList<String>();
		while ((line = reader.readLine()) != null) {
			if (line.equals("")) {
				taggedSentences.add(new TaggedSentence(new BoundedList<String>(
						words, START_WORD, STOP_WORD), new BoundedList<String>(
						tags, START_WORD, STOP_WORD)));
				words = new LinkedList<String>();
				tags = new LinkedList<String>();
			} else {
				String[] fields = line.split("\\s+");
				words.add(fields[0]);
				tags.add(hasTags ? fields[1] : "");
			}
		}
		System.out.println("Read " + taggedSentences.size() + " sentences.");
		return taggedSentences;
	}

	private static void labelTestSet(POSTagger posTagger,
			List<TaggedSentence> testSentences, String path) throws Exception {
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		for (TaggedSentence sentence : testSentences) {
			List<String> words = sentence.getWords();
			List<String> guessedTags = posTagger.tag(words);
			for (int i = 0; i < words.size(); i++) {
				writer.write(words.get(i) + "\t" + guessedTags.get(i) + "\n");
			}
			writer.write("\n");
		}
		writer.close();
	}

	private static void evaluateTagger(POSTagger posTagger,
			List<TaggedSentence> taggedSentences,
			Set<String> trainingVocabulary, boolean verbose) 
	{
		double numTags = 0.0;
		double numTagsCorrect = 0.0;
		double numUnknownWords = 0.0;
		double numUnknownWordsCorrect = 0.0;
		int numDecodingInversions = 0;
		for (TaggedSentence taggedSentence : taggedSentences) 
		{
			List<String> words = taggedSentence.getWords();
			List<String> goldTags = taggedSentence.getTags();
			List<String> guessedTags = posTagger.tag(words);
			for (int position = 0; position < words.size() - 1; position++) 
			{
				String word = words.get(position);
				String goldTag = goldTags.get(position);
				String guessedTag = guessedTags.get(position);
				if (guessedTag.equals(goldTag))
				{
					numTagsCorrect += 1.0;
				}
				numTags += 1.0;
				if (!trainingVocabulary.contains(word)) {
					if (guessedTag.equals(goldTag))
						numUnknownWordsCorrect += 1.0;
					numUnknownWords += 1.0;
				}
			}
			double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
			double scoreOfGuessedTagging = posTagger
					.scoreTagging(new TaggedSentence(words, guessedTags));
			if (scoreOfGoldTagging > scoreOfGuessedTagging) {
				numDecodingInversions++;
				if (verbose)
					System.out
							.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
			}
			if (verbose)
				System.out.println(alignedTaggings(words, goldTags,
						guessedTags, true) + "\n");
		}
		System.out.println("Tag Accuracy: " + (numTagsCorrect / numTags)
				+ " (Unknown Accuracy: "
				+ (numUnknownWordsCorrect / numUnknownWords)
				+ ")  Decoder Suboptimalities Detected: "
				+ numDecodingInversions);
	}

	// pretty-print a pair of taggings for a sentence, possibly suppressing the
	// tags which correctly match
	private static String alignedTaggings(List<String> words,
			List<String> goldTags, List<String> guessedTags,
			boolean suppressCorrectTags) {
		StringBuilder goldSB = new StringBuilder("Gold Tags: ");
		StringBuilder guessedSB = new StringBuilder("Guessed Tags: ");
		StringBuilder wordSB = new StringBuilder("Words: ");
		for (int position = 0; position < words.size(); position++) {
			equalizeLengths(wordSB, goldSB, guessedSB);
			String word = words.get(position);
			String gold = goldTags.get(position);
			String guessed = guessedTags.get(position);
			wordSB.append(word);
			if (position < words.size() - 1)
				wordSB.append(' ');
			boolean correct = (gold.equals(guessed));
			if (correct && suppressCorrectTags)
				continue;
			guessedSB.append(guessed);
			goldSB.append(gold);
		}
		return goldSB + "\n" + guessedSB + "\n" + wordSB;
	}

	private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2,
			StringBuilder sb3) {
		int maxLength = sb1.length();
		maxLength = Math.max(maxLength, sb2.length());
		maxLength = Math.max(maxLength, sb3.length());
		ensureLength(sb1, maxLength);
		ensureLength(sb2, maxLength);
		ensureLength(sb3, maxLength);
	}

	private static void ensureLength(StringBuilder sb, int length) {
		while (sb.length() < length) {
			sb.append(' ');
		}
	}

	private static Set<String> extractVocabulary(
			List<TaggedSentence> taggedSentences) {
		Set<String> vocabulary = new HashSet<String>();
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			vocabulary.addAll(words);
		}
		return vocabulary;
	}

	public static void main(String[] args) throws Exception {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		boolean verbose = false;
		boolean useValidation = true;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// Whether or not to print the individual errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Read in data
		System.out.print("Loading training sentences...");
		List<TaggedSentence> trainTaggedSentences = readTaggedSentences(
				basePath + "/en-wsj-train.pos", true);
		Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
		System.out.println("done.");
		
		System.out.print("Loading in-domain dev sentences...");
		List<TaggedSentence> devInTaggedSentences = readTaggedSentences(
				basePath + "/en-wsj-dev.pos", true);
		
	//	List<TaggedSentence> devInTaggedSentences = readTaggedSentences(
	//			basePath + "/en-temp.pos", true);
		
		
		System.out.println("done.");
		
		System.out.print("Loading out-of-domain dev sentences...");
		List<TaggedSentence> devOutTaggedSentences = readTaggedSentences(
				basePath + "/en-web-weblogs-dev.pos", true);
		System.out.println("done.");
		
		System.out.print("Loading out-of-domain blind test sentences...");
		List<TaggedSentence> testSentences = readTaggedSentences(basePath
				+ "/en-web-test.blind", false);
		System.out.println("done.");

		// Construct tagger components
		// TODO : improve on the MostFrequentTagScorer
	//	LocalTrigramScorer localTrigramScorer = new MostFrequentTagScorer(false);
		//First using a HMM tag scores
		LocalTrigramScorer localTrigramScorer = new HMMTagScorer(false);
		
		// TODO : improve on the GreedyDecoder
	//	TrellisDecoder<State> trellisDecoder = new GreedyDecoder<State>();
		TrellisDecoder<State> trellisDecoder = new ViterbiDecoder<State>();
	//	TrellisDecoder<State> trellisDecoder = new ViterbiCheckDecoder<State>();
		

		// Train tagger
		POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);
		posTagger.train(trainTaggedSentences);

		// Optionally tune hyperparameters on dev data
		posTagger.validate(devInTaggedSentences);

		// Test tagger
		System.out.println("Evaluating on in-domain data:.");
		evaluateTagger(posTagger, devInTaggedSentences, trainingVocabulary,
				verbose);
		
		System.out.println("Evaluating on out-of-domain data:.");
	
		evaluateTagger(posTagger, devOutTaggedSentences, trainingVocabulary,
				verbose);
	//	labelTestSet(posTagger, testSentences, basePath + "/en-web-test.tagged");
	}
}
