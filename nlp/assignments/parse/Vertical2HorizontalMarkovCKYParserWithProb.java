package nlp.assignments.parse;

import java.util.ArrayList;
import java.util.List;

import nlp.assignments.parse.PCFGParserTester.Grammar;
import nlp.assignments.parse.PCFGParserTester.Lexicon;
import nlp.assignments.parse.PCFGParserTester.TreeAnnotations;
import nlp.assignments.parse.PCFGParserTester.UnaryClosure;
import nlp.ling.Tree;

//public class CollinsMarkovCKYParser extends CollinsCKYParser
public class Vertical2HorizontalMarkovCKYParserWithProb extends CKYParserWithProb

{

	public Vertical2HorizontalMarkovCKYParserWithProb()
	{
		 markov = VERTICAL2_HORIZONTAL_MARKOV ;
	}
	
	public void training(List<Tree<String>> trainTrees)
	{
		System.out.println("Starting Vertical2/Horizontal markovization Training Process....");

		ArrayList<Tree<String>> aTrees = new ArrayList<Tree<String>>();
		for (Tree<String> tree : trainTrees) 
		{
			aTrees.add(Vertical2HorizontalTreeAnnotation.annotateTreeVerticalMarkov(tree));
		}

		lexicon = new Lexicon(aTrees);
		grammar = new Grammar(aTrees);

		System.out.println("Vertical2/Horizontal markovization Training Process Done.");
	//	debugPrintln("Grammar:");
		// System.out.println(grammar);
		uc = new UnaryClosure(grammar);
	}
	
	
}
