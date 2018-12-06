import spacy
import Tokenizer
import ConstitutencyParser
import DependencyParser
import LemmatizerMorphAnalyzer
import NPChunker
import POSTagger
import PreverbIdentifier
import HuWordToVec


def main():
    # english_spacy_test()
    # hu_spacy_vector_test()
    # english_spacy_word_vector()
    # hu_spacy_word_vector_dumb()   # deprecated
    hu_spacy_word_vector_smart()


def hu_spacy_vector_test():
    nlp = create_spacy_hu()
    doc = nlp("Az alma messze esett a fától")
    print('The vector is: ')
    print(doc.vector)


def hu_spacy_word_vector_smart():
    nlp = create_spacy_hu()
    doc = nlp('alma körte')
    for token in doc:
        print(token.vector)
    print('Doc.vector: ')
    print(doc.vector)


def hu_spacy_word_vector_dumb():
    import numpy as np

    nlp = create_spacy_hu()

    vectors = spacy.vectors.Vectors(shape=(1, 4))
    vector = np.array([1, 1, 1, 1])
    alma_id = nlp.vocab.strings['alma']
    vectors.add(alma_id, vector=vector)

    nlp.vocab.vectors = vectors

    doc = nlp('almák')
    for token in doc:
        print(token.has_vector)
    print(doc.vector)


def english_spacy_word_vector():
    nlp = spacy.load('en')
    apple = nlp.vocab['apple']
    print(apple.vector)
    kastely = nlp.vocab['kastély']
    print(kastely.vector)


def english_spacy_test():
    nlp = spacy.load("en")
    doc = nlp("The beautiful castle waits for no one except to bang them")

    print(doc.vector)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)
        token.tag_ = 'NOUN'


def create_spacy_hu():
    nlp = spacy.blank('hu')
    nlp.tokenizer = Tokenizer.HuTokenizer(nlp.vocab)

    morph_analyzer = LemmatizerMorphAnalyzer.HuLemmaMorph(nlp)
    nlp.add_pipe(morph_analyzer)

    constitutency_parser = ConstitutencyParser.ConstitutencyParser(nlp)
    nlp.add_pipe(constitutency_parser)

    dependency_parser = DependencyParser.DependencyParser(nlp)
    nlp.add_pipe(dependency_parser)

    np_chunker = NPChunker.NPChunker(nlp)
    nlp.add_pipe(np_chunker)

    POS_analyzer = POSTagger.HuPOSTagger(nlp)
    nlp.add_pipe(POS_analyzer)

    preverb_identifier = PreverbIdentifier.PreverbIdentifier(nlp)
    nlp.add_pipe(preverb_identifier)

    hu_word_to_vec = HuWordToVec.HUWordToVec()
    nlp.add_pipe(hu_word_to_vec)

    return nlp

if __name__ == '__main__':
    main()
