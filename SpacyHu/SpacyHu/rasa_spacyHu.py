
import Tokenizer
import ConstitutencyParser
import DependencyParser
import LemmatizerMorphAnalyzer
import NPChunker
import POSTagger
import PreverbIdentifier

import spacy
from spacy.tokens import Doc
import urllib
import xml.etree.ElementTree as ET
from rasa_nlu.components import Component


class HuSpacyTagger(Component):
    name = "hu_spacy_tagger"
    provides = ["spacy_doc"]

    def __init__(self):
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

        self.nlp = nlp

    def process(self, message, **kwargs):
        message.set("spacy_doc", nlp(message.text))


if __name__ == "__main__":
    comp = HuSpacyTagger()
