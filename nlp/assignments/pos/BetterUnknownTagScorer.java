package nlp.assignments.pos;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.Counters;
import nlp.util.Pair;

import nlp.assignments.pos.POSTaggerTester.LabeledLocalTrigramContext;
import nlp.assignments.pos.POSTaggerTester.LocalTrigramContext;
import nlp.assignments.pos.POSTaggerTester.LocalTrigramScorer;

class BooleanTag<F, S> extends Pair<F, S> {

        public BooleanTag(F first, S second) {
                super(first, second);
        }

        /**
         * 
         */
        private static final long serialVersionUID = 1L;
        
}
public class BetterUnknownTagScorer implements LocalTrigramScorer {

        boolean restrictTrigrams; // if true, assign log score of
                                                                // Double.NEGATIVE_INFINITY to illegal tag
                                                                // trigrams.

        CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
        CounterMap<String, String> tagsOfWords = new CounterMap<String, String>();
        Counter<String> unknownWordTags = new Counter<String>();
        CounterMap<String, String> seenTagTrigrams = new CounterMap<String, String>();
        CounterMap<String, String> seenTagBigrams = new CounterMap<String, String>();
        Counter<String> seenTagUnigrams = new Counter<String>();
        CounterMap<String, String> suffixMaps = new CounterMap<String, String>();
        Set<String> unknownWords = new HashSet<String>();
        double[] lambdas;
        public int getHistorySize() {
                return 2;
        }

        private final int SUFFIXLEN = 10;
        private double theta = 0;
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
                        
                        String bigramString = makeBigramString(localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
                        double likelihood = 0;
                        if (wordsToTags.keySet().contains(word)) 
                        {
                                likelihood = tagsOfWords.getCount(tag, word);
                        }
                        else
                        {
                              likelihood = unknownWordTags.getCount(tag);
                                likelihood = getUnknownWordProbability(tag, word);
                                if (likelihood == 0 || likelihood >=1)
                                {
                                	//System.out.println("here");
                                	likelihood = 0.0001 ;
                                }
                                
                        }
                        
                        double prior = lambdas[0] * seenTagTrigrams.getCount(bigramString, tag)
                                        + lambdas[1] * seenTagBigrams.getCount(localTrigramContext.previousTag, tag)
                                        + lambdas[2] * seenTagUnigrams.getCount(tag);
                        double logScore = Math.log(likelihood)
                                + Math.log(prior);
                        if (likelihood == 0 || prior == 0) {
//                              System.out.println("the word is" + word);
                                if (!unknownWords.contains(word)) {
//                                      System.out.println("the word is" + word);
                                }
                                unknownWords.add(word);
                        }
//                      System.out.println("tagsOfWords count is " + tagsOfWords.getCount(tag, word) + ", and seenTagTrigrams count is " + prior);
                        if (!restrictTrigrams || allowedFollowingTags.isEmpty()
                                        || allowedFollowingTags.contains(tag))
                                logScoreCounter.setCount(tag, logScore);
                        else {
                                System.err.println("Cannot get logScore!");
                        }
                }
                return logScoreCounter;
        }

        
        private double getUnknownWordProbability(String tag, String unknownWord) 
        {
//              if (unknownWord.startsWith("[A-Z]")) {
//                      if (tag.contains("NNP")) {
//                              return .99;
//                      }
//              }
                if (unknownWord.contains("-")) {
                        if (tag.contains("JJ")) {
                                return .66;
                        } else if (tag.contains("NN")) {
                                return .33;
                        }
                }

                int wordLen = unknownWord.length();
//              Counter<String> pTag = new Counter<String>();
                        double p = 0;
                        for (int i = 0; i < (SUFFIXLEN > wordLen ? wordLen : SUFFIXLEN); i++) 
                        {
                               if (0 == i)
                               {
                                        p = suffixMaps.getCount(unknownWord.substring(wordLen - 1), tag) / ( 1 + theta);
                                        continue;
                                }
                                double temp1 = suffixMaps.getCount(unknownWord.substring(wordLen -1 - i, wordLen - 1), tag);
//                              double temp2 = suffixMaps.getCount(unknownWord.substring(wordLen - i, wordLen - 1), tag);
                                p = (temp1 + theta * p) / (1 + theta);
                        }
//                      System.out.println("p is " + p);
//                      pTag.setCount(tag, p);
                return p;
        }
        
        private Set<String> allowedFollowingTags(Set<String> tags,
                        String previousPreviousTag, String previousTag) {
                Set<String> allowedTags = new HashSet<String>();
                for (String tag : tags) {
                        String bigramString = makeBigramString(previousPreviousTag, previousTag);
                        if (seenTagTrigrams.getCount(bigramString, tag) > 0) {
                                allowedTags.add(tag);
                        }
                }
                return allowedTags;
        }
        
        private String makeBigramString(String previousTag, String currentTag) {
                return previousTag + " " + currentTag;
        }

        public void train(
                        List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
                // collect word-tag counts
            LocalTrigramScorer localTrigramScorer =  new BetterUnknownTagScorer(false);
                for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
                        String word = labeledLocalTrigramContext.getCurrentWord();
                        String tag = labeledLocalTrigramContext.getCurrentTag();
                        if (!wordsToTags.containsKey(word)) {
                                unknownWordTags.incrementCount(tag, 1.0);
                        }
                        wordsToTags.incrementCount(word, tag, 1.0);
                        tagsOfWords.incrementCount(tag, word, 1.0);
                        seenTagTrigrams.incrementCount(makeBigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag()), labeledLocalTrigramContext.getCurrentTag(), 1.0);
                        seenTagBigrams.incrementCount(labeledLocalTrigramContext.previousTag, labeledLocalTrigramContext.currentTag, 1.0);
                        seenTagUnigrams.incrementCount(tag, 1.0);
                        
                        int wordLen = word.length();
                        for (int i = 0; i < (SUFFIXLEN > wordLen ? wordLen : SUFFIXLEN); i++) 
                        {
                                suffixMaps.incrementCount(word.substring(wordLen - 1 - i, wordLen - 1), tag, 1.0);
                        }
                }
                calThetas();
        }
        
        private void calThetas() {
                double total = tagsOfWords.totalCount();
                double size = tagsOfWords.size();
//              Counter<String> pTag = new Counter<String>();
                double sum = 0;
                for (String tag : tagsOfWords.keySet()) {
                        double count = tagsOfWords.getCounter(tag).totalCount();
//                      pTag.setCount(tag, count / total);
                        sum += Math.pow(count / total - 1 / size, 2);
                }
                theta = Math.sqrt(1 / (size) * sum);
                System.out.println("theta is " + theta);
        }

        public void validate(
                        List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
                lambdas = deletedInterpolation(labeledLocalTrigramContexts);
                for (int i = 0; i < lambdas.length; i++) {
                        System.out.println("lambda" + (i+1) + " is " + lambdas[i]);
                }
                normailize();
        }
        
        private void normailize() {
                wordsToTags = Counters.conditionalNormalize(wordsToTags);
                tagsOfWords = Counters.conditionalNormalize(tagsOfWords); // 46 tags
                unknownWordTags = Counters.normalize(unknownWordTags);
                seenTagTrigrams = Counters.conditionalNormalize(seenTagTrigrams);
                seenTagBigrams = Counters.conditionalNormalize(seenTagBigrams);
                seenTagUnigrams = Counters.normalize(seenTagUnigrams);
                suffixMaps = Counters.conditionalNormalize(suffixMaps);
                System.out.println("suffixMaps size is " + suffixMaps.size());
        }
        
        private double[] deletedInterpolation(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
                double totalTokens = wordsToTags.totalCount();
                double lambda1 = 0, lambda2 = 0, lambda3 = 0;
                double temp1 = 0, temp2 = 0, temp3 = 0;
                for (LabeledLocalTrigramContext labeledLocalTrigramContext: labeledLocalTrigramContexts) {
                        String currentTag = labeledLocalTrigramContext.getCurrentTag();
                        String previousPreviousTag = labeledLocalTrigramContext.previousPreviousTag;
                        String previousTag = labeledLocalTrigramContext.previousTag;
                        String previousBigramString = makeBigramString(previousPreviousTag, previousTag);
                        double count = seenTagTrigrams.getCount(previousBigramString, currentTag);
                        if (count > 0) {
                                if (totalTokens != 1) {
                                        temp3 = (tagsOfWords.getCounter(currentTag).totalCount() - 1) / (totalTokens - 1);
                                } else {
                                        temp3 = 0;
                                }
                                temp2 = tagsOfWords.getCounter(previousTag).totalCount();
                                if (temp2 != 1) {
                                        String currentBigramString = makeBigramString(previousTag, currentTag);
                                        temp2 = (seenTagTrigrams.getCounter(currentBigramString).totalCount() - 1) / (temp2 - 1);
                                } else {
                                        temp2 = 0;
                                }
                                temp1 = seenTagTrigrams.getCounter(previousBigramString).totalCount();
                                if (temp1 != 1) {
                                        temp1 = (count - 1) / (temp1 - 1);
                                } else {
                                        temp1 = 0;
                                }
                                int index = max(temp1, temp2, temp3);
                                if (index == 1) {
                                        lambda1 += count;
                                } else if (index == 2) {
                                        lambda2 += count;
                                } else if (index == 3) {
                                        lambda3 += count;
                                } else {
                                        throw new RuntimeException("index is out of bound here!!!");
                                }
                        }
                }
                double total = lambda1 + lambda2 + lambda3;
                lambda1 /= total;
                lambda2 /= total;
                lambda3 /= total;
                
                return new double[]{lambda1, lambda2, lambda3};
        }
        
        private int max(double d1, double d2, double d3) {
                int index = 0;
                if (d1 > d2) {
                        if (d1 >= d3) {
                                index = 1; 
                        } else {
                                index = 3;
                        }
                } else {
                        if (d2 >= d3) {
                                index = 2;
                        } else {
                                index = 3;
                        }
                }
                return index;
        }

        public BetterUnknownTagScorer(boolean restrictTrigrams) {
                this.restrictTrigrams = restrictTrigrams;
        }
}