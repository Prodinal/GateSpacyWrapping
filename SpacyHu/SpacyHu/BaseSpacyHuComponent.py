

class BaseSpacyHuComponent():
    def __init__(self, nlp, label, url, necessary_modules):
        self.nlp = nlp
        self.label = nlp.vocab.strings[label]
        self.url = url + ','.join(necessary_modules) + '&text='
