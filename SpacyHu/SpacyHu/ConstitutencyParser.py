import spacy
from spacy.tokens import Doc, Span, Token
import urllib
import xml.etree.ElementTree as ET
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class ConstitutencyParser(BaseSpacyHuComponent):

    def __init__(self,
                 nlp,
                 label='ConsParser',
                 url='http://localhost:8000/process?run='):
        necessary_modules = ['QT', 'ML3-PosLem-hfstcode', 'ML3-Cons']
        super().__init__(nlp, label, url, necessary_modules)
        Token.set_extension('cons', default='')

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
            token._.cons = self.get_value_from_annotation(annotation, 'cons')

        return doc

if __name__ == '__main__':
    from Tokenizer import HuTokenizer

    debug_text = u'A kék alma nagyon gyorsan elrepült a sima köcsög mellett'
    nlp = spacy.blank('en')
    remote_url = 'http://hlt.bme.hu/chatbot/gate/process?run='
    nlp.tokenizer = HuTokenizer(nlp.vocab, url=remote_url)
    constitutency_parser = ConstitutencyParser(nlp, url=remote_url)
    nlp.add_pipe(constitutency_parser, last=True)

    doc = nlp(debug_text)
    result_cons_tree = []
    for token in doc:
        print('Token is: ' + token.text)
        print('Cons is: ' + token._.cons)
        result_cons_tree.append(token._.cons)
    print("".join(result_cons_tree))
