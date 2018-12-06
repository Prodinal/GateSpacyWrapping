import spacy
from spacy.tokens import Doc, Span, Token
import urllib
import xml.etree.ElementTree as ET
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class NER(BaseSpacyHuComponent):

    def __init__(
            self,
            nlp,
            label='NER',
            url='http://localhost:8000/process?run='):

        necessary_modules = ['QT', 'ML3-PosLem-hfstcode',
                             'huntag3-NER-pipe-hfstcode']
        super().__init__(nlp, label, url, necessary_modules)
        # Token.set_extension('NERBIO1', default='')

    def get_token_by_idx(self, idx, doc):
        for token in doc:
            if token.idx == idx:
                return token

    def get_value_from_annotation(self, annotation, attr_name):
        for child in annotation.getchildren():
            if child.find('Name').text == attr_name:
                return child.find('Value').text

    def __call__(self, doc):
        text = urllib.parse.quote_plus(doc.text)
        result = urllib.request.urlopen(self.url + text).read()
        annotationset = ET.fromstring(result).find('AnnotationSet')
        for annotation in annotationset.getchildren():
            if annotation.get('Type') != 'Token':
                continue

            word_index = int(annotation.get('StartNode'))
            token = self.get_token_by_idx(word_index, doc)
            ner_tag = self.get_value_from_annotation(annotation, 'NER-BIO1')
            ner_tag = ner_tag.strip()
            if ner_tag != 'O':     # this is a capital o letter
                ner_tag_parts = ner_tag.split('-')
                label = doc.vocab.strings[ner_tag_parts[1]]
                if ner_tag_parts[0] == '1':
                    doc.ents = (list(doc.ents) +
                                [Span(doc, token.i,
                                      token.i+1,
                                      label=label)])
                elif ner_tag_parts[0] == 'B':
                    self.ner_begin_idx = token.i
                elif ner_tag_parts[0] == 'E':
                    if self.ner_begin_idx is None:
                        raise Exception('Found end of ner,'
                                        'when looking for beginning')
                    else:
                        span = Span(doc,
                                    self.ner_begin_idx,
                                    token.i+1,
                                    label=label)
                        span.merge()
                        doc.ents = (list(doc.ents) + [span])
                        self.ner_begin_idx = None

        return doc

if __name__ == '__main__':
    from Tokenizer import HuTokenizer
    debug_text = u'A kék New York elrepült a sima Berlin mellett'
    nlp = spacy.blank('en')
    nlp.tokenizer = HuTokenizer(nlp.vocab)
    ner = NER(nlp)
    nlp.add_pipe(ner, last=True)

    doc = nlp(debug_text)
    for span in doc.ents:
        print(span.text + " " + span.label_)
