import nltk
import spacy
from typing import List
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
import string


class ConjunctsHandler:
    CONJ_IND = ('and', '&', ',')
    PREP_MEANS = ['via', 'by', 'through']
    CONJ_DEPS = ('conj', 'appos')

    def __init__(self, doc: spacy.tokens.doc.Doc):
        self.sent = doc
        self.all_verbs = []

    def get_conjuncts(self, verb: spacy.tokens.token.Token):
        """

        Args:
              verb: one of the verbs in the row of verbs which we want to get conjuncts for

        Returns all the verb conjuncts for the cases when verbs come one by one with the following separators:
        'and', '&', ','.
        Output:
        A tuple of 2 elements:
        - the main verb (the first verb in the row)
        - the list of all of the verbs in the row, including the main one.
        e.g 'We re-engineer & launch a new global e-commerce platform', kw='e-commerce platform',
        input='launch', output=(reengineer, [reengineer, launch])
        or 'I drive roadmap and strategy for e-commerce platform', kw='e-commerce platform',
        input-'strategy', output=(roadmap, [roadmap, strategy])

        """
        self.main_verb = verb
        if verb.conjuncts:
            if verb.dep_ == 'conj':
                if self.sent[verb.i - 1].text in self.CONJ_IND or (
                        verb.i > 2 and self.sent[verb.i - 3:verb.i].text == 'as well as'):
                    if (self.sent[verb.i - 2].pos_ == 'VERB'
                            or self.sent[verb.i - 2].text[-3:] == 'ing'
                            or (self.sent[verb.i - 2].text == ','
                                and (self.sent[verb.i - 3].pos_ == 'VERB'
                                     or self.sent[verb.i - 2].text[-3:] == 'ing'))):
                        for conj in verb.conjuncts:
                            if conj.dep_ != 'conj':
                                if (conj.pos_ == 'VERB'
                                    or conj.text[-3:] == 'ing'
                                        or all(list(map(lambda x: x.text[-3:] == 'ing'
                                                    and x.i > conj.i, conj.conjuncts)))):
                                    self.all_verbs = []
                                    for word in conj.conjuncts:
                                        if ((word.pos_ == 'VERB'
                                            or word.text[-3:] == 'ing')
                                                and word.i < verb.i
                                                and (self.sent[word.i + 1].text in self.CONJ_IND
                                                     or self.sent[word.i + 1] in conj.conjuncts)):
                                            self.all_verbs.append(word)
                                    self.all_verbs.append(verb)
                                    if (self.sent[conj.i + 1].text in self.CONJ_IND
                                            and (conj.pos_ == 'VERB'
                                                 or conj.text[-3:] == 'ing')):
                                        self.main_verb = conj
                                        self.all_verbs = [self.main_verb] + self.all_verbs
                                    else:
                                        self.main_verb = self.all_verbs[0]
                                    return self  # return self.main_verb, self.all_verbs
            else:
                if self.sent[verb.i + 1].text in self.CONJ_IND:
                    self.all_verbs = [verb]
                    for word in verb.conjuncts[:-1]:
                        if ((word.pos_ == 'VERB' or word.text[-3:] == 'ing')
                                and (self.sent[word.i + 1].text in self.CONJ_IND
                                     or self.sent[word.i + 1] in verb.conjuncts)):
                            self.all_verbs.append(word)
                    self.all_verbs.append(verb.conjuncts[-1])
                    self.main_verb = verb
                    return self  # verb, self.verbs

        elif (self.sent[verb.i - 1].text in self.CONJ_IND) and (
                self.sent[verb.i - 2].pos_ == 'VERB' or self.sent[verb.i - 2].text[-3:] == 'ing' or (
                self.sent[verb.i - 2].text == ',' and (
                self.sent[verb.i - 3].pos_ == 'VERB' or self.sent[verb.i - 2].text[-3:] == 'ing'))):
            if verb.head == self.sent[verb.i - 3] or verb.head == self.sent[verb.i - 2]:
                self.main_verb = verb.head
                self.all_verbs = [self.main_verb, verb]
                return self  # self.main_verb, self.all_verbs

    def get_chunks(self, token: spacy.tokens.token.Token) -> spacy.tokens.token.Token:
        lefts = list(token.lefts)
        if lefts:
            if self.sent[lefts[0].i - 1].text in self.CONJ_IND:
                return (self.get_chunks(self.sent[lefts[0].i - 2])
                        if self.sent[lefts[0].i - 2].text != ','
                        else self.get_chunks(self.sent[lefts[0].i - 3]))
            else:
                return token
        else:
            if self.sent[token.i - 1].text in self.CONJ_IND:
                return (self.get_chunks(self.sent[token.i - 2])
                        if self.sent[token.i - 2].text != ','
                        else self.get_chunks(self.sent[token.i - 3]))
            else:
                return token

    def get_main_token(self, token: spacy.tokens.token.Token,
                       kw_span: spacy.tokens.span.Span) -> spacy.tokens.token.Token:
        """

        Args:
            token: the main (rightmost) token of the keyword
            kw_span: the span of the keyword tokens to the left of its head

        Returns the token of the first element of the enumeration which the keyword is part of.

        """
        while token.dep_ in self.CONJ_DEPS:
            if token.head.pos_ != 'VERB' and token.head not in kw_span:
                token = token.head
                kw_span = self.sent[token.i:token.i + 1]
            else:
                break

        if token.head.text in self.PREP_MEANS + ['for']:
            return token
        if (token.head.text in ('including', 'like')
                or (token.head.text == 'as'
                    and ('such' in list(map(lambda x: x.text, token.head.lefts))))):
            incl = token.head
            if incl.head.pos_ == 'VERB':
                if ((self.sent[incl.i - 1].pos_ == 'PROPN'
                     or (self.sent[incl.i - 1].text == ','
                         and self.sent[incl.i - 2].pos_ == 'PROPN'))
                        and (self.sent[incl.i - 1].head.text == 'for'
                             or self.sent[incl.i - 2].head.text == 'for')):
                    prep = (self.sent[incl.i - 1].head
                            if self.sent[incl.i - 1].head.text == 'for'
                            else self.sent[incl.i - 2].head)
                    return prep.head
                if incl.i > 1:
                    if self.sent[incl.i - 1].text != ',' and self.sent[incl.i - 1].dep_ in self.CONJ_DEPS:
                        return self.get_main_token(self.sent[incl.i - 1], self.sent[incl.i - 1:incl.i])
                    elif self.sent[incl.i - 1].text == ',' and self.sent[incl.i - 2].dep_ in self.CONJ_DEPS:
                        return self.get_main_token(self.sent[incl.i - 2], self.sent[incl.i - 2:incl.i - 1])
                    else:
                        return self.sent[incl.i - 1] if self.sent[incl.i - 1].text != ',' else self.sent[incl.i - 2]
            else:
                return (self.get_main_token(incl.head, self.sent[incl.head.i:incl.head.i + 1])
                        if incl.head != incl
                        else token)
        if (token.head.pos_ == 'ADP'
                and token.head.text not in self.PREP_MEANS
                and token.head.text != 'for'
                and (token.head.head.dep_ in self.CONJ_DEPS
                     or token.head.text in ('including', 'like')
                     or (token.head.text == 'as'
                         and ('such' in list(map(lambda x: x.text, token.head.lefts)))))):
            if self.sent[token.head.head.i - 1].text == 'and' and self.sent[token.head.head.i - 2].pos_ == 'NOUN':
                return token
            if token.head.dep_ != 'ROOT' and token.head.head.pos_ not in ['VERB', 'AUX']:
                return self.get_main_token(token.head.head, self.sent[token.head.head.i:token.head.head.i + 1])
        elif (token.head.pos_ == 'ADP'
              and token.head.text not in self.PREP_MEANS
              and token.head.text != 'for'
              and token.head.head.dep_ == 'pobj'
              and (self.get_chunks(token.head.head.head.head)
                   or token.head.head.head.head.dep_ in self.CONJ_DEPS)
                and token.head.head.head.head.pos_ not in ['VERB', 'AUX']):
            return self.get_main_token(token.head.head.head.head,
                   self.sent[token.head.head.head.head.i:token.head.head.head.head.i + 1])
        else:
            lefts = list(token.lefts)
            if lefts:
                if lefts[0].i > 2:
                    token_ = lefts[0]
                    if (self.sent[token_.i - 1].text in self.CONJ_IND
                            and self.sent[token_.i - 2].pos_ not in ['VERB', 'AUX']
                            and self.sent[token_.i - 3].pos_ not in ['VERB', 'AUX']):
                        return (self.get_main_token(self.sent[token_.i - 2], self.sent[token_.i - 2:token_.i - 1])
                                if self.sent[token_.i - 2].text != ','
                                else self.get_main_token(self.sent[token_.i - 3], self.sent[token_.i - 3:token_.i - 2]))
            lefts = list(kw_span[-1].lefts)
            if lefts:
                if lefts[0].i > 2:
                    token_ = lefts[0]
                    if (self.sent[token_.i - 1].text in self.CONJ_IND
                            and self.sent[token_.i - 2].pos_ not in ['VERB', 'AUX']
                            and self.sent[token_.i - 3].pos_ not in ['VERB', 'AUX']):
                        return (self.get_main_token(self.sent[token_.i - 2], self.sent[token_.i - 2:token_.i - 1])
                                if self.sent[token_.i - 2].text != ','
                                else self.get_main_token(self.sent[token_.i - 3], self.sent[token_.i - 3:token_.i - 2]))
            if token.i > 2:
                if (self.sent[token.i - 1].text in self.CONJ_IND
                        and self.sent[token.i - 2].pos_ not in ['VERB', 'AUX']
                        and self.sent[token.i - 3].pos_ not in ['VERB', 'AUX']):
                    return (self.get_main_token(self.sent[token.i - 2], self.sent[token.i - 2:token.i - 1])
                            if self.sent[token.i - 2].text != ','
                            else self.get_main_token(self.sent[token.i - 3], self.sent[token.i - 3:token.i - 2]))
            if kw_span[0].i > 2:
                token_ = kw_span[0]
                if (self.sent[token_.i - 1].text in self.CONJ_IND
                        and self.sent[token_.i - 2].pos_ not in ['VERB', 'AUX']
                        and self.sent[token_.i - 3].pos_ not in ['VERB', 'AUX']):
                    return (self.get_main_token(self.sent[token_.i - 2], self.sent[token_.i - 2:token_.i - 1])
                            if self.sent[token_.i - 2].text != ','
                            else self.get_main_token(self.sent[token_.i - 3], self.sent[token_.i - 3:token_.i - 2]))
            else:
                return token
            return token

    def get_main_verb_token(self, verb: spacy.tokens.token.Token):
        """

        Args:
            verb: the main (rightmost) token of the keyword if it's a verb

        Returns the token of the first element of the enumeration which the keyword is part of if the keyword is
        a verb.

        """
        last_verb = verb
        while verb.dep_ in self.CONJ_DEPS:
            if verb.head.pos_ == 'VERB':
                verb = verb.head
                self.all_verbs.append(verb)
            else:
                if self.get_synsets(verb.head) and verb.head.text[-4:] != 'ings':
                    verb = verb.head
                    self.all_verbs.append(verb)
                else:
                    break
        self.all_verbs += [last_verb]
        self.main_verb = verb
        return self

    @staticmethod
    def get_synsets(token: spacy.tokens.token.Token) -> List:
        """

        Returns the list of synsets from WordNet (if any) where the given token comes as a verb.

        """
        synsets = wn.synsets(token.lemma_.strip(string.punctuation),
                             pos=wn.VERB)  # берем все синсеты, в которых данное слово является глаголом
        return synsets
