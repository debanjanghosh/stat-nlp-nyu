package nlp.assignments.pos;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nlp.io.PennTreebankReader;
import nlp.ling.Tree;
import nlp.ling.Trees;
import nlp.util.*;

import nlp.assignments.*;
import nlp.assignments.pos.POSTaggerTester.TaggedSentence;
import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;

/**
 * @author Dan Klein
 */
public class POSTaggerTester_maxent {

  static final String START_WORD = "<S>";
  static final String STOP_WORD = "</S>";
  static final String START_TAG = "<S>";
  static final String STOP_TAG = "</S>";
  static final String UNK = "<UNK>";
  
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
      if (this == o) return true;
      if (!(o instanceof TaggedSentence)) return false;

      final TaggedSentence taggedSentence = (TaggedSentence) o;

      if (tags != null ? !tags.equals(taggedSentence.tags) : taggedSentence.tags != null) return false;
      if (words != null ? !words.equals(taggedSentence.words) : taggedSentence.words != null) return false;

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
   * States are pairs of tags along with a position index, representing the two
   * tags preceding that position.  So, the START state, which can be gotten by
   * State.getStartState() is [START, START, 0].  To build an arbitrary state,
   * for example [DT, NN, 2], use the static factory method
   * State.buildState("DT", "NN", 2).  There isnt' a single final state, since
   * sentences lengths vary, so State.getEndState(i) takes a parameter for the
   * length of the sentence.
   */
  static class State {

    private static transient Interner<State> stateInterner = new Interner<State>(new Interner.CanonicalFactory<State>() {
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

    public static State buildState(String previousPreviousTag, String previousTag, int position) {
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
      return State.buildState(tag, getPreviousPreviousTag(), getPosition() - 1);
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof State)) return false;

      final State state = (State) o;

      if (position != state.position) return false;
      if (previousPreviousTag != null ? !previousPreviousTag.equals(state.previousPreviousTag) : state.previousPreviousTag != null) return false;
      if (previousTag != null ? !previousTag.equals(state.previousTag) : state.previousTag != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = position;
      result = 29 * result + (previousTag != null ? previousTag.hashCode() : 0);
      result = 29 * result + (previousPreviousTag != null ? previousPreviousTag.hashCode() : 0);
      return result;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getPosition() + "]";
    }

    int position;
    String previousTag;
    String previousPreviousTag;

    private void setState(String previousPreviousTag, String previousTag, int position) {
      this.previousPreviousTag = previousPreviousTag;
      this.previousTag = previousTag;
      this.position = position;
    }

    private State() {
    }

    private State(State state) {
      setState(state.getPreviousPreviousTag(), state.getPreviousTag(), state.getPosition());
    }
  }

  /**
   * A Trellis is a graph with a start state an an end state, along with
   * successor and predecessor functions.
   */
  static class Trellis <S> {
    S startState;
    S endState;
    CounterMap<S, S> forwardTransitions;
    CounterMap<S, S> backwardTransitions;

    List<String> sentence;
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
     * For a given state, returns a counter over what states can be next in the
     * markov process, along with the cost of that transition.  Caution: a state
     * not in the counter is illegal, and should be considered to have cost
     * Double.NEGATIVE_INFINITY, but Counters score items they don't contain as
     * 0.
     */
    public Counter<S> getForwardTransitions(S state) {
      return forwardTransitions.getCounter(state);

    }


    /**
     * For a given state, returns a counter over what states can precede it in
     * the markov process, along with the cost of that transition.
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
   * A TrellisDecoder takes a Trellis and returns a path through that trellis in
   * which the first item is trellis.getStartState(), the last is
   * trellis.getEndState(), and each pair of states is conntected in the
   * trellis.
   */
  static interface TrellisDecoder <S> {
    List<S> getBestPath(Trellis<S> trellis);
  }

  static class ViterbiDecoder<S> implements TrellisDecoder<S> {
	  public List<S> getBestPath(Trellis<S> trellis) {
		  Set<S> prev_states = new HashSet<S>();
		  
		  Map<S, Double> prev_states_cost = new HashMap<S, Double>();
		  Map<S, List<S>> prev_states_path = new HashMap<S, List<S>>();
		  
		  // Start off!
		  S start_state = trellis.getStartState();
		  List<S> start_path = new ArrayList<S>();
		  start_path.add(start_state);
		  prev_states.add(start_state);
		  prev_states_cost.put(start_state, 0.0);
		  prev_states_path.put(start_state, start_path);
		  

		  while (!prev_states.contains(trellis.getEndState())){

			  if (prev_states.isEmpty()){ // DEBUG
		//		  System.out.println("ERROR: Infinite looping");
			//	  for (S state : prev_states)
			//		  System.out.print(state+" ");
			//	  System.out.println();
			//	  System.out.println(trellis.sentence);
				  System.exit(0);
			  }
			  
			  Set<S> next_states = new HashSet<S>();
			  Map<S, Double> next_states_cost = new HashMap<S, Double>();
			  Map<S, List<S>> next_states_path = new HashMap<S, List<S>>();			 
			  
			 for (S prev_state : prev_states) {
				 Counter<S> transitions = trellis.getForwardTransitions(prev_state);
				 Double prev_state_cost = prev_states_cost.get(prev_state);
//				 System.out.println("=="+prev_state+"==");//DEBUG
//				 System.out.println(transitions);
				 
				 for (S next_state : transitions.keySet()){		
					 if (transitions.getCount(next_state) == Double.NEGATIVE_INFINITY && transitions.getCount(transitions.argMax()) != Double.NEGATIVE_INFINITY) {
						 continue;
					 }
					 if (!next_states.contains(next_state)){
						 List<S> prev_state_path = new ArrayList<S>(prev_states_path.get(prev_state));
						 prev_state_path.add(next_state);
						 next_states.add(next_state);
						 next_states_cost.put(next_state, prev_state_cost+transitions.getCount(next_state));
						 next_states_path.put(next_state, prev_state_path);
					 } else {
						 if (prev_state_cost+transitions.getCount(next_state) > next_states_cost.get(next_state)){
							 next_states_cost.put(next_state, prev_state_cost+transitions.getCount(next_state));
							 List<S> prev_state_path = new ArrayList<S>(prev_states_path.get(prev_state));
							 prev_state_path.add(next_state);
							 
							 next_states_path.put(next_state, prev_state_path);
						 }
					 }
				 }
			 }
			 
			 prev_states_path = next_states_path;
			 prev_states_cost = next_states_cost;
			 prev_states = next_states;
			 
		  }
		  
		  if (prev_states.size() != 1){
			  System.out.println("SIZE ERROR");
			  System.exit(0);
		  }
		  
		  List<S> result_list = prev_states_path.get(trellis.getEndState());
		  result_list.add(trellis.getEndState());
//		  for (S state : result_list) {
//			  System.out.print(state + " ");
//		  }
//		  System.out.println();
//		  
//		  System.out.println(trellis.sentence);
		  return result_list;
	  }
  }
  
  static class GreedyDecoder <S> implements TrellisDecoder<S> {
    public List<S> getBestPath(Trellis<S> trellis) {
      List<S> states = new ArrayList<S>();
      S currentState = trellis.getStartState();
      states.add(currentState);
      //System.out.println(trellis.getEndState()); // DEBUG
      
      while (!currentState.equals(trellis.getEndState())) {
        Counter<S> transitions = trellis.getForwardTransitions(currentState);
        
        S nextState = transitions.argMax();
        
        // BEGIN DEBUG
        if (nextState == null){	
        	System.out.println(states.get(states.size()-2));
        	System.out.println(trellis.getForwardTransitions(states.get(states.size()-2)));
        	for (String word : trellis.sentence){
        		System.out.print(word + " ");
        	}
        	System.out.println();
        	
        	for (S state : states){
        		System.out.print(state+" ");
        	}
        	System.out.println();
        	
        	System.out.println(trellis.getEndState());
        	System.out.println(currentState);
        }
        // END DEBUG
        
        states.add(nextState);
        currentState = nextState;
      }

      for (S state : states){ // DEBUG
    	  System.out.print(state+" ");
      }
      System.out.println();
      System.out.println(trellis.sentence);
      // END DEBUG
      
      return states;
    }
  }

  static class POSTagger {

    LocalTrigramScorer localTrigramScorer;
    TrellisDecoder<State> trellisDecoder;

    // chop up the training instances into local contexts and pass them on to the local scorer.
    public void train(List<TaggedSentence> taggedSentences) {
      localTrigramScorer.train(extractLabeledLocalTrigramContexts(taggedSentences));
    }

    // chop up the validation instances into local contexts and pass them on to the local scorer.
    public void validate(List<TaggedSentence> taggedSentences) {
      localTrigramScorer.validate(extractLabeledLocalTrigramContexts(taggedSentences));
    }

    private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(List<TaggedSentence> taggedSentences) {
      List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
      for (TaggedSentence taggedSentence : taggedSentences) {
        localTrigramContexts.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
      }
      return localTrigramContexts;
    }

    private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(TaggedSentence taggedSentence) {
      List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
      List<String> words = new BoundedList<String>(taggedSentence.getWords(), START_WORD, STOP_WORD);
      List<String> tags = new BoundedList<String>(taggedSentence.getTags(), START_TAG, STOP_TAG);
      for (int position = 0; position <= taggedSentence.size() + 1; position++) {
        labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(words, position, tags.get(position - 2), tags.get(position - 1), tags.get(position)));
      }
      return labeledLocalTrigramContexts;
    }

    /**
     * Builds a Trellis over a sentence, by starting at the state State, and
     * advancing through all legal extensions of each state already in the
     * trellis.  You should not have to modify this code (or even read it,
     * really).
     * @throws IOException 
     */
    private Trellis<State> buildTrellis(List<String> sentence)  {
    	
    	double start = System.currentTimeMillis();
    	double tally = 0.0;
    	
      Trellis<State> trellis = new Trellis<State>();
      trellis.sentence = sentence; // DEBUG
      trellis.setStartState(State.getStartState());
      State stopState = State.getStopState(sentence.size() + 2);
      trellis.setStopState(stopState);
      Set<State> states = Collections.singleton(State.getStartState());
      for (int position = 0; position <= sentence.size() + 1; position++) {
        Set<State> nextStates = new HashSet<State>();
        for (State state : states) {
          if (state.equals(stopState))
            continue;
          LocalTrigramContext localTrigramContext = new LocalTrigramContext(sentence, position, state.getPreviousPreviousTag(), state.getPreviousTag());
          
          double temp_start = System.currentTimeMillis();
          Counter<String> tagScores = localTrigramScorer.getLogScoreCounter(localTrigramContext);
          double temp_end = System.currentTimeMillis();
          tally += temp_end-temp_start;
          
          for (String tag : tagScores.keySet()) {
            double score = tagScores.getCount(tag);
            State nextState = state.getNextState(tag);
            trellis.setTransitionCount(state, nextState, score);
            nextStates.add(nextState);
          }
        }
//        System.out.println("States: "+nextStates);
        states = nextStates;
      }
      
      double stop = System.currentTimeMillis();
      //System.out.println(stop-start);
      //System.out.println(tally);
      return trellis;
    }

    // to tag a sentence: build its trellis and find a path through that trellis
    public List<String> tag(List<String> sentence) {
  //  	System.out.println(sentence);
      Trellis<State> trellis = buildTrellis(sentence);
      List<State> states = trellisDecoder.getBestPath(trellis);
      List<String> tags = State.toTagList(states);
      tags = stripBoundaryTags(tags);
      return tags;
    }

    /**
     * Scores a tagging for a sentence.  Note that a tag sequence not accepted
     * by the markov process should receive a log score of
     * Double.NEGATIVE_INFINITY.
     */
    public double scoreTagging(TaggedSentence taggedSentence) {
      double logScore = 0.0;
      List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(taggedSentence);
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        Counter<String> logScoreCounter = localTrigramScorer.getLogScoreCounter(labeledLocalTrigramContext);
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

    public POSTagger(LocalTrigramScorer localTrigramScorer, TrellisDecoder<State> trellisDecoder) {
      this.localTrigramScorer = localTrigramScorer;
      this.trellisDecoder = trellisDecoder;
    }
  }

  /**
   * A LocalTrigramContext is a position in a sentence, along with the previous
   * two tags -- basically a FeatureVector.
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
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "]";
    }

    public LocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag) {
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
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
    }

    public LabeledLocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag, String currentTag) {
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
     * values are exponentiated and summed, they should sum to one.  For
     * efficiency, the Counter can contain only the tags which occur in the
     * given context with non-zero model probability.
     * @throws IOException 
     */
    Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) ;

    void train(List<LabeledLocalTrigramContext> localTrigramContexts);

    void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
  }

  /**
   * Implements a HMM Tag Scorer where transition is linear combination
   * empirical conditional trigram+bigram+unigram distribution
   * 
   * Emission probability for known words are the good-turing discounted 
   * probability. Emission probability for unknown words is determined by
   * Maxent classifier that extracts feature from the unknown word.
   * 
   * Also implements illegal tag filtering
   * 
   * @author Min
   *
   */
  
  static class HMMTagScorer implements LocalTrigramScorer {
	  
	  boolean restrictTrigram ;
	  
	  CounterMap<List<String>, String> emp_trigram_tag_cond = new CounterMap<List<String>, String>();
	  CounterMap<String, String> emp_bigram_tag_cond = new CounterMap<String, String>();
	  Counter<String> emp_unigram_tag_distr = new Counter<String>();
	  
	  CounterMap<String, String> word_given_tag_cond = new CounterMap<String, String>();
	  CounterMap<String, Double> word_given_tag_freq_cond = new CounterMap<String, Double>();
	  
	  CounterMap<String, String> tag_given_word_cond = new CounterMap<String, String>();
	  
	  Set<String> seen_tag_trigrams = new HashSet<String>();
	  
	  ProbabilisticClassifier<String, String> unk_classif = null;
	  
	  double lambda1 = 0.7;
	  double lambda2 = 0.2;
		  
	  String[] symbols = {".", ":", STOP_WORD, "``", "''", ",", "$", "#"}; 

	  public HMMTagScorer(boolean restrictTrigram){
		  this.restrictTrigram = restrictTrigram;
		  
	  }
	  
	  private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
		  Set<String> allowedTags = new HashSet<String>();
		  for (String tag : tags) {
			  String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
			  if (seen_tag_trigrams.contains((trigramString))) {
				  allowedTags.add(tag);
			  }
		  }
		  return allowedTags;
	  }
	    	  
	  public Counter<String> getLogScoreCounter(LocalTrigramContext local_trigram_context) 
	  {
		  
		  double start_timing = System.currentTimeMillis();
		  
		  int cur_pos = local_trigram_context.getPosition();
		  String cur_word = local_trigram_context.getCurrentWord();
		  String prev_tag = local_trigram_context.getPreviousTag();
		  String prev_prev_tag = local_trigram_context.getPreviousPreviousTag();
		  
		  List<String> prev_tags = new ArrayList<String>();
		  prev_tags.add(prev_prev_tag);
		  prev_tags.add(prev_tag);
		  
		  Counter<String> log_score = new Counter<String>();
		  
		  Set<String> all_known_tags = tag_given_word_cond.getCounter(cur_word).keySet();
		  if (all_known_tags.isEmpty())
		  {
			  all_known_tags = emp_unigram_tag_distr.keySet();
		  }
		  //Set<String> allowed_trigrams = allowedFollowingTags(all_tags, prev_prev_tag, prev_tag);
		  Set<String> allowed_trigrams = new HashSet<String>();
		  
		  double mid_early = System.currentTimeMillis();
		  
		  boolean symbol_skip = false;
		  for (String symb : symbols)
		  {
			  if (symb.equals(cur_word))
			  {
				  symbol_skip = true;

				  if (!restrictTrigram || allowed_trigrams.isEmpty() || allowed_trigrams.contains(symb)){
					  log_score.setCount(symb, 0);
					  break;
				  }
			  }
		  }
		  
		  double mid = System.currentTimeMillis();
		  
	//	  for (String tag : all_known_tags)
	//	  {
		
//			  double score = word_given_tag_cond.getCount(tag, UNK);
		  
		//	  System.out.println(tag + "\t" + score );
//		  }
		  if (!symbol_skip) 
		  {
			  for (String tag : all_known_tags)
			  {
				  // transmission_prob
				  double transmission_prob = lambda1*emp_trigram_tag_cond.getCount(prev_tags, tag);
				  transmission_prob += lambda2*emp_bigram_tag_cond.getCount(prev_tag, tag);
				  transmission_prob += (1-lambda1-lambda2)*emp_unigram_tag_distr.getCount(tag);

				  // emission probability
				  double emission_prob = word_given_tag_cond.getCount(tag, cur_word);
				  if (emission_prob == 0)
				  {
					  String lower_word = cur_word.toLowerCase();
					  // check to ensure  
					  if (cur_pos == 0 && word_given_tag_cond.getCount(tag, lower_word)>0 && false) 
					  {
						  emission_prob = word_given_tag_cond.getCount(tag, lower_word);
					  } 
					  else
					  {					  
						  double multiplier = word_given_tag_cond.getCount(tag, UNK);
						  if (multiplier <= 0.0000001) // DEBUG
							  emission_prob = 0.0; 
						  else
						  {
							  
							  System.out.println(cur_word + "\t" + "UNK?"  + "\t" + tag);
							  
							  try
							  {
								emission_prob = unk_classif.getProbabilities(cur_word).getCount(tag);
							  } 
							  catch (IOException e) 
							  {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						  }
					  }
				  }

				  if (!restrictTrigram || allowed_trigrams.isEmpty() || allowed_trigrams.contains(tag))
				  {
					  log_score.setCount(tag, Math.log(transmission_prob)+Math.log(emission_prob));
				  }
			  }
		  }
		  
//		  if ((cur_word == "." && log_score.argMax() != ".") ||
//			  (cur_word == STOP_WORD && log_score.argMax() != STOP_WORD)){ // DEBUG
//			  System.out.println(cur_word);
//			  System.out.println(log_score);
//			  System.out.println("ERROR");
//			  System.exit(0);
//		  }

		  if (log_score.keySet() == null)
		  { // DEBUG
			  System.out.println("BAD: "+local_trigram_context);
			  System.exit(1);
		  }
		  
		  double end_timing = System.currentTimeMillis(); // DEBUG
		  double total_time = end_timing-start_timing;
		  if (total_time > 1000)
		  {
			  System.out.println("TOO LONG:"+total_time);
			  System.out.println(local_trigram_context);
			  System.out.println(mid - mid_early);
			  System.out.println(end_timing - mid);
		  }
		  
		  return log_score;
		  
	  }
	  
	  public void validate(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
		  
		  int num_of_sample = labeledLocalTrigramContexts.size();
		  System.out.println("Tuning on "+labeledLocalTrigramContexts.size()+" held-out samples.");
		  
		  int max_so_far = Integer.MIN_VALUE;
		  double max_lambda1 = 0.0;
		  double max_lambda2 = 0.0;
		  for (double i=0.5 ; i<=1.0; i+=0.05){
			  for (double j=0.0; j<=1.0-i; j+=0.05){
				  lambda1 = i;
				  lambda2 = j;
				  
				  int corr_label = 0;
				  for (LabeledLocalTrigramContext local_tri_cont : labeledLocalTrigramContexts){
					  String guessed_label = getLogScoreCounter(local_tri_cont).argMax();
					  String true_label = local_tri_cont.getCurrentTag();
					  if (guessed_label == true_label)
						  corr_label+=1;
				  }
	//		  System.out.println("L1 = "+i+"  L2 = "+j+" :: "+corr_label);
				  if (corr_label > max_so_far){
					  max_so_far = corr_label;
					  max_lambda1 = i;
					  max_lambda2 = j;
				  }
			  }
		  }
		  lambda1 = max_lambda1;
		  lambda2 = max_lambda2;  
		  System.out.println("L1 = "+lambda1+"  L2 = "+lambda2+" :: ");
	  }
	  
	  class UnkFeatExtractor implements FeatureExtractor<String, String> {
		  
		  public Counter<String> extractFeatures(String word){
			  
			  Counter<String> features = new Counter<String>();
			  
			  String[] exploded_word = word.split("");
			  int length = exploded_word.length;
			  
			  for (int i=0; i<exploded_word.length; i+=1){
				  features.incrementCount("UNI--"+exploded_word[i], 1.0);
			  }
			  
			  for (int i=1; i<exploded_word.length; i+=1){ // DEBUG COMMENTTED
				  features.incrementCount("BI--"+exploded_word[i-1]+exploded_word[i], 1.0);				  
			  }
			  
//			  for (int i=2; i<exploded_word.length; i+=1){
//				  features.incrementCount("TRI--"+exploded_word[i-2]+exploded_word[i-1]+exploded_word[i], 1.0);
//			  }
			  
			  //features.incrementCount("<LEN>", length);
			  
			  // FS Special:
			  //features.incrementCount("<H-T>"+exploded_word[0]+exploded_word[length-1], 1.0);
			  

			  // FS B:
			  if (exploded_word[0].equals(exploded_word[0].toUpperCase()))
				  features.incrementCount("<CAP>", 1.0);
				  
			  if (word.matches(".*\\d+.*")) // DEBUG commented
				  features.incrementCount("<NUM>", 1.0);			  
			  
			  if (word.matches(".*\\W+.*"))
				  features.incrementCount("<SYMB>", 1.0);
			  
			  if (exploded_word.length > 2){
				  features.incrementCount("LAST3-"+exploded_word[length-3]+exploded_word[length-2]+exploded_word[length-1], 1.0);
				  features.incrementCount("FIRST3-"+exploded_word[0]+exploded_word[1]+exploded_word[2], 1.0);
			  }
			  if (exploded_word.length > 1){
				  features.incrementCount("LAST2-"+exploded_word[length-2]+exploded_word[length-1], 1.0);
			  	  features.incrementCount("FIRST2-"+exploded_word[0]+exploded_word[1], 1.0);
			  }
			  
			  
			  features.incrementCount("LAST-"+exploded_word[length-1], 1.0);
			  features.incrementCount("FIRST-"+exploded_word[0], 1.0);
			  	  
			  return features;
		  }
		  
	  }
	  
	  /**
	   * collects 
	   * 1. empirical trigram tag counts
	   * 2. empirical bigram tag counts
	   * 3. empirical unigram tag counts
	   * 4. count of word given tag
	   * 4.5. count of tag given word
	   * 5. keep track of seenTrigrams
	   * 6. train unknown-Maxent weight
	   */
	  public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts){
		  
		  // empirical distribution mining
		  for (LabeledLocalTrigramContext trigramContext : labeledLocalTrigramContexts)
		  {
			  String prev_prev_tag = trigramContext.getPreviousPreviousTag();
			  String prev_tag = trigramContext.getPreviousTag();
			  String cur_tag = trigramContext.getCurrentTag();
			  String cur_word = trigramContext.getCurrentWord();
			  
			  List<String> prev_two_tags = new ArrayList<String>();
			  prev_two_tags.add(prev_prev_tag);
			  prev_two_tags.add(prev_tag);
			  
			  emp_trigram_tag_cond.incrementCount(prev_two_tags, cur_tag, 1);
			  emp_bigram_tag_cond.incrementCount(prev_tag, cur_tag, 1);
			  emp_unigram_tag_distr.incrementCount(cur_tag,1);
			  
			  word_given_tag_cond.incrementCount(cur_tag, cur_word, 1);
			  tag_given_word_cond.incrementCount(cur_word, cur_tag, 1);
			  
			  seen_tag_trigrams.add(makeTrigramString(prev_prev_tag, prev_tag, cur_tag));
			  
			  }
		  
		  emp_trigram_tag_cond.normalize();
		  emp_bigram_tag_cond.normalize();
		  emp_unigram_tag_distr.normalize();
		  
		  // run good-turing smoothing on word_given_tag_cond
		  Set<String> all_tags = emp_unigram_tag_distr.keySet();

		  // first, find freq of freq
		  for (String tag : all_tags) 
		  {
			  Counter<String> word_count = word_given_tag_cond.getCounter(tag);
			  
			  for (String word : word_count.keySet())
			  {
				  word_given_tag_freq_cond.incrementCount(tag, word_count.getCount(word), 1);
			  }
		  }

		  // set GT counts, set unknown count
		  for (String tag : all_tags)
		  {
			  Counter<String> word_count = word_given_tag_cond.getCounter(tag);

			  double old_total_count = word_count.totalCount();
			  for (String word : word_count.keySet())
			  {
				  double old_count = word_count.getCount(word);
				  double new_count = old_count;
				  double new_normalized_count = 0;
				  
				  Double kplus1_count = word_given_tag_freq_cond.getCount(tag, new Double(old_count+1));
				  Double k_count = word_given_tag_freq_cond.getCount(tag, new Double(old_count));
				  
				  if (kplus1_count != 0) 
				  {
					  new_count = (old_count+1)*kplus1_count/k_count;
					  if (new_count > old_count)
						  new_count = old_count;
				  }
				  
				  new_normalized_count = new_count/old_total_count;
				  
				  word_given_tag_cond.setCount(tag, word, new_normalized_count);
			  }
			  
			  double new_total_count = word_given_tag_cond.getCounter(tag).totalCount();
			  if (new_total_count > 1.00000001){
				  System.out.println("ERROR: bad probability distr") ;
				  System.exit(0);
			  }
			  
			  System.out.println("tag: "+tag+"  smoothed UNK mass: "+(1-new_total_count));
			  word_given_tag_cond.setCount(tag, UNK, 1-new_total_count);
		  }
		  // end GT smoothing
		  
		  // BEGIN Unknown-Maxent Training
		  System.out.print("Training Maxent ...");
		  List<LabeledInstance<String, String>> labeled_inst_train_data = new ArrayList<LabeledInstance<String, String>>();
		  int max_num_samples = labeledLocalTrigramContexts.size();
		  System.out.print("on at most "+max_num_samples+" samples ...");
		  CounterMap<String, String> seen_words = new CounterMap<String, String>();
		  int num_of_samples = 0;
		  
		  for (LabeledLocalTrigramContext label_tri_cont : labeledLocalTrigramContexts)
		  {
			  String cur_tag = label_tri_cont.getCurrentTag();
			  String cur_word = label_tri_cont.getCurrentWord();
			  if (seen_words.getCount(cur_tag, cur_word) > 0){
				  continue;
			  } else {
				  seen_words.incrementCount(cur_tag, cur_word, 1.0);
			  }
			  
			  String[] skip_tag = {".", "SYM", "-RRB-", "-LRB-", ",", "$", "``", "''", "#", "$", "TO", "MD", "</S>", "PRP$", "EX", "WP", "WP$", "WDT", "PDT", "RBS", "WRB", "POS"};
			  boolean skip_flag = false;
			  for (int i=0; i<skip_tag.length; i+=1)
			  {
				  if (cur_tag == skip_tag[i] || word_given_tag_cond.getCount(cur_tag, UNK) <= 0.0000001){ // DEBUG -min
					  skip_flag = true;
					  break;
				  }
			  }
			  if (skip_flag) continue;
			  
			  num_of_samples += 1;
			  labeled_inst_train_data.add(new LabeledInstance<String, String>(label_tri_cont.getCurrentTag(), label_tri_cont.getCurrentWord()));
			  if (num_of_samples == max_num_samples)
				  break;
		  }
		  System.out.print("on "+num_of_samples+" samples ...\n");
		  
		  ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(1.0, 20, new UnkFeatExtractor());
		  try {
			unk_classif = factory.trainClassifier(labeled_inst_train_data);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		  System.out.println("done");
		  
		  
		  
	  }
	  
	  public String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
		  return previousPreviousTag + " " + previousTag + " " + currentTag;
	  }
  }
  
  /**
   * The MostFrequentTagScorer gives each test word the tag it was seen with
   * most often in training (or the tag with the most seen word types if the
   * test word is unseen in training.  This scorer actually does a little more
   * than its name claims -- if constructed with restrictTrigrams = true, it
   * will forbid illegal tag trigrams, otherwise it makes no use of tag history
   * information whatsoever.
   */
  static class MostFrequentTagScorer implements LocalTrigramScorer {

    boolean restrictTrigrams; // if true, assign log score of Double.NEGATIVE_INFINITY to illegal tag trigrams.

    CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
    Counter<String> unknownWordTags = new Counter<String>();
    Set<String> seenTagTrigrams = new HashSet<String>();

    public int getHistorySize() {
      return 2;
    }

    public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
      int position = localTrigramContext.getPosition();
      String word = localTrigramContext.getWords().get(position);
      Counter<String> tagCounter = unknownWordTags;
      if (wordsToTags.keySet().contains(word)) {
        tagCounter = wordsToTags.getCounter(word);
      }
      Set<String> allowedFollowingTags = allowedFollowingTags(tagCounter.keySet(), localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
      Counter<String> logScoreCounter = new Counter<String>();
      for (String tag : all_tags) { // CHANGED! tagCounter.keySet()
        double logScore = Math.log(tagCounter.getCount(tag));
        if (!restrictTrigrams || allowedFollowingTags.isEmpty() || allowedFollowingTags.contains(tag))
          logScoreCounter.setCount(tag, logScore);
      }
      return logScoreCounter;
    }

    private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
      Set<String> allowedTags = new HashSet<String>();
      for (String tag : tags) {
        String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
        if (seenTagTrigrams.contains((trigramString))) {
          allowedTags.add(tag);
        }
      }
      return allowedTags;
    }

    public String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
      return previousPreviousTag + " " + previousTag + " " + currentTag;
    }

    Set<String> all_tags = new HashSet<String>();
    
    public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      // collect word-tag counts
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        String word = labeledLocalTrigramContext.getCurrentWord();
        String tag = labeledLocalTrigramContext.getCurrentTag();
        
        all_tags.add(tag);
        
        if (!wordsToTags.keySet().contains(word)) {
          // word is currently unknown, so tally its tag in the unknown tag counter
          unknownWordTags.incrementCount(tag, 1.0);
        }
        wordsToTags.incrementCount(word, tag, 1.0);
        seenTagTrigrams.add(makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()));
      }
      wordsToTags = Counters.conditionalNormalize(wordsToTags);
      unknownWordTags = Counters.normalize(unknownWordTags);
    }

    public void validate(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      // no tuning for this dummy model!
    }

    public MostFrequentTagScorer(boolean restrictTrigrams) {
      this.restrictTrigrams = restrictTrigrams;
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
  private static List<TaggedSentence> readTaggedSentences(String path, int low, int high) {
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(path, low, high);
    List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
    Trees.TreeTransformer<String> treeTransformer = new Trees.EmptyNodeStripper();
    for (Tree<String> tree : trees) {
      tree = treeTransformer.transformTree(tree);
      List<String> words = new BoundedList<String>(new ArrayList<String>(tree.getYield()), START_WORD, STOP_WORD);
      List<String> tags = new BoundedList<String>(new ArrayList<String>(tree.getPreTerminalYield()), START_TAG, STOP_TAG);
      taggedSentences.add(new TaggedSentence(words, tags));
    }
    return taggedSentences;
  }

  private static void evaluateTagger(POSTagger posTagger, List<TaggedSentence> taggedSentences, Set<String> trainingVocabulary, boolean verbose) {
    double numTags = 0.0;
    double numTagsCorrect = 0.0;
    double numUnknownWords = 0.0;
    double numUnknownWordsCorrect = 0.0;
    int numDecodingInversions = 0;
    
    CounterMap<String, String> confusion_mat = new CounterMap<String, String>();
    int num_corr_sent = 0;
    int total_sent_num = taggedSentences.size();
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      List<String> goldTags = taggedSentence.getTags();
      List<String> guessedTags = posTagger.tag(words);
      for (int position = 0; position < words.size() - 1; position++) {
        String word = words.get(position);
        String goldTag = goldTags.get(position);
        String guessedTag = guessedTags.get(position);
        if (guessedTag.equals(goldTag))
          numTagsCorrect += 1.0;
        else {
        	confusion_mat.incrementCount(goldTag, guessedTag, 1.0);
        }
        numTags += 1.0;
        
        if (!trainingVocabulary.contains(word)) {
          if (guessedTag.equals(goldTag))
            numUnknownWordsCorrect += 1.0;
          numUnknownWords += 1.0;
        }
      }
      if (numTagsCorrect == numTags)
    	  num_corr_sent += 1;
      
      double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
      double scoreOfGuessedTagging = posTagger.scoreTagging(new TaggedSentence(words, guessedTags));
      if (scoreOfGoldTagging > scoreOfGuessedTagging) {
        numDecodingInversions++;
        if (verbose) System.out.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
      }
      if (verbose) System.out.println(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
    }
    System.out.println("Confusion Matrix");
    System.out.println(confusion_mat);
    System.out.println("Tag Accuracy: " + (numTagsCorrect / numTags) + " (Unknown Accuracy: " + (numUnknownWordsCorrect / numUnknownWords) + ")  Decoder Suboptimalities Detected: " + numDecodingInversions + " Sentence Accuracy: "+(num_corr_sent/total_sent_num));
    
    
  }

  // pretty-print a pair of taggings for a sentence, possibly suppressing the tags which correctly match
  private static String alignedTaggings(List<String> words, List<String> goldTags, List<String> guessedTags, boolean suppressCorrectTags) {
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

  private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2, StringBuilder sb3) {
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

  private static Set<String> extractVocabulary(List<TaggedSentence> taggedSentences) {
    Set<String> vocabulary = new HashSet<String>();
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      vocabulary.addAll(words);
    }
    return vocabulary;
  }

  public static void main(String[] args) throws Exception {
    // Parse command line flags and arguments
    Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

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

    // Whether to use the validation or test set
    if (argMap.containsKey("-test")) {
      String testString = argMap.get("-test");
      if (testString.equalsIgnoreCase("test"))
        useValidation = false;
    }
    System.out.println("Testing on: " + (useValidation ? "validation" : "test"));

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
    //LocalTrigramScorer localTrigramScorer = new MostFrequentTagScorer(false);
   
//	LocalTrigramScorer localTrigramScorer = new HMMTagScorer(false);
    
    LocalTrigramScorer localTrigramScorer = (LocalTrigramScorer) new BetterUnknownTagScorer(false);
    
    
    
    
    // TODO : improve on the GreedyDecoder
    TrellisDecoder<State> trellisDecoder = new ViterbiDecoder<State>();
    //TrellisDecoder<State> trellisDecoder = new GreedyDecoder<State>();

    // Train tagger
    POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);
    posTagger.train(trainTaggedSentences);
    posTagger.validate(devInTaggedSentences); // DEBUG COMMENT
    
    System.out.println("Evaluating on in-domain data:.");
    
    double start_timing = System.currentTimeMillis();
	
	evaluateTagger(posTagger, devInTaggedSentences, trainingVocabulary,
			verbose);
	
	double stop_timing = System.currentTimeMillis();
	
	
	System.out.println("Evaluated on in-domain data:." + " time " + (stop_timing-start_timing));
	
	
	System.out.println("Evaluating on out-of-domain data:.");
	
	start_timing = System.currentTimeMillis();
		
	evaluateTagger(posTagger, devOutTaggedSentences, trainingVocabulary,
			verbose);
	
	stop_timing = System.currentTimeMillis();
		
	System.out.println("Evaluating on out-of-domain data:." + " time " + (stop_timing-start_timing));
		
	
	
		
    // Test tagger
 //   evaluateTagger(posTagger, testSentences, trainingVocabulary, verbose);
  }
}
