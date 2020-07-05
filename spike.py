import spacy
import neuralcoref
from collections import defaultdict
import itertools
from graphviz import Digraph

from spacy.symbols import VERB, root

def getVerb(t):
	cycle = set()
	while True:
		if t.pos == VERB:
			return t
		if t.dep == root:
			return None
		if t in cycle:
			return None
		cycle.add(t)
		t = t.head

def extract_relations(doc,nlp):

	relation_triplets = []
	relation_doublets = set()

	spans = []

	for sent in doc.sents:
		objs = [ t for t in sent if "obj" in t.dep_]
		
		for o in objs:
			st = list(o.subtree)
			size = len(st)
			if size > 1:
				span = doc[st[0].i : st[size-1].i+1]
				spans.append(span)
		
	spans = spacy.util.filter_spans(spans)

	with doc.retokenize() as retokenizer:
		for span in spans:
			retokenizer.merge(span)
	

	doc = nlp.get_pipe('neuralcoref')(doc)

	for sent in doc.sents:

		verbs = [ t for t in sent if t.pos == VERB ]

		objs = [ t for t in sent if "obj" in t.dep_]
		subjs = [ t for t in sent if "subj" in  t.dep_]

		tokenToChunk = {}
		for chunk in doc.noun_chunks:
			tokenToChunk[chunk.root] = chunk

		subjsMapped = defaultdict(list)
		objsMapped = defaultdict(list)

		for t in subjs:
			verb = getVerb(t)
			if not verb == None:
				subjsMapped[t.head].append(t)
		for t in objs:
			verb = getVerb(t)
			if not verb == None:
				objsMapped[verb].append(t)

		discovered_objs = set(objs)

		mapped_objs = set()

		for v in verbs:
			for subj,obj1 in itertools.product(subjsMapped[v],objsMapped[v]):
				mapped_objs.add(obj1)
				if subj in tokenToChunk:
					if tokenToChunk[subj]._.is_coref:
						subjTxt = tokenToChunk[subj]._.coref_cluster.main.text
						
					else:
						subjTxt = tokenToChunk[subj].text
				else:
					if subj._.in_coref:
						
						subjTxt = subj._.coref_clusters[0].main.text
					else:
						subjTxt = subj.text

				if obj1 in tokenToChunk:
					if tokenToChunk[obj1]._.is_coref:
						objTxt = tokenToChunk[obj1]._.coref_cluster.main.text
					else:
						objTxt = tokenToChunk[obj1].text
				else:
					if obj1._.in_coref:
						objTxt = obj1._.coref_clusters[0].main.text
					else:
						objTxt = obj1.text
						
				verb_phrase = v.text

				for child in list(v.children):
					if child.dep_ == "attr":
						verb_phrase += " " + child.text

				out = [ subjTxt, verb_phrase, objTxt ]
				relation_triplets.append(out)
				r1 = (subjTxt, verb_phrase)
				r2 = (verb_phrase, objTxt)
				relation_doublets.add(r1)
				relation_doublets.add(r2)
				

		diff = discovered_objs - mapped_objs
		if diff:
			for entity in diff:
				og_entity = entity
				relation = ""
				while entity.head != entity:
					entity=entity.head
				
				for word in list(entity.subtree):
					if word == og_entity:
						break
					elif not "subj" in word.dep_:
						relation+= " " + word.text
				
				children = list(entity.children)
				for child in children:
					if "subj" in child.dep_:
						if child in tokenToChunk:
							if tokenToChunk[child]._.is_coref:
								subjTxt = tokenToChunk[child]._.coref_cluster.main.text
								
							else:
								subjTxt = tokenToChunk[child].text
						else:
							if child._.in_coref:
								subjTxt = child._.coref_clusters[0].main.text
							else:
								subjTxt = child.text
						
						out = [ subjTxt, entity.text, og_entity.text ]
						relation_triplets.append(out)
						r1 = (subjTxt, relation)
						r2 = (relation, og_entity.text)
						relation_doublets.add(r1)
						relation_doublets.add(r2)
						break
					
					
				


	# return relation_triplets
	return relation_doublets
				

if __name__ == "__main__":

	nlp = spacy.load("en_core_web_sm")
	merge_ents = nlp.create_pipe("merge_entities")
	merge_nps = nlp.create_pipe("merge_noun_chunks")
	nlp.add_pipe(merge_nps)
	nlp.add_pipe(merge_ents)
	neuralcoref.add_to_pipe(nlp)
	
	text = "In physics, sound is a vibration that propagates as an acoustic wave, through a transmission medium such as a gas, liquid or solid. In human physiology and psychology, sound is the reception of such waves and their perception by the brain."

	doc = nlp(text)
	relations = extract_relations(doc,nlp)
	u = Digraph('G', filename='cluster.gv')
	u.attr(size='6,6')
	# for relation in relations:
	# 	if not relation[0] == relation[2]:
	# 		u.edge(relation[0],relation[1])
	# 		u.edge(relation[1],relation[2])
	for relation in relations:
		u.edge(relation[0],relation[1])
	u.view()
	

		