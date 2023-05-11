import re
from itertools import chain, product
from typing import Tuple, Callable, List
import spacy

from ConjunctsHandler import ConjunctsHandler
from VerbTypeChecker import VerbTypeChecker

import nltk

nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn


verbs_stoplist = {'including', 'include', 'includes', 'consist', 'start up', 'paid', 'driven', 'proven', 'oriented',
                  'spans', 'span', 'limit', 'limited', 'thought', 'based', 'like', 'like to', 'love', 'love to',
                  'need', 'need to', 'want', 'want to', 'lead nurturing', 'lead generation'}
ROLES = {'leader', 'specialist', 'professional', 'strategist', 'manager', 'coordinator', 'intern', 'admin',
         'consultant', 'director', 'marketer', 'officer', 'apprentice', 'associate', 'assistant', 'expert'}
objects_exclude = {'lot', 'plenty', 'variety', 'deal', 'range'}
verbs_with_ccomps = {'help', 'allow', 'let', 'discover', 'enable'}
verb_methods = [attribute for attribute in dir(VerbTypeChecker) if callable(getattr(VerbTypeChecker, attribute))
                and attribute.startswith('__') is False]


def print_sentence_decomposition(sentence_doc, print_sentence=True, print_lefts_and_rights=True):
    if print_sentence:
        print(sentence_doc, "\n")

    if print_lefts_and_rights:
        for token in sentence_doc:
            print("{} {:25s} {:10s} {:10s} {:20s} {:20s} {}".format(token.i, token.text, token.pos_, token.dep_,
                                                                    token.head.text, "['" + "','".join(
                    [x.text for x in token.lefts]) + "']", "['" + "','".join([x.text for x in token.rights]) + "']"))

    else:
        for token in sentence_doc:
            print("{} {:25s} {:10s} {:10s} {:20s}".format(token.i, token.text, token.pos_, token.dep_, token.head.text))


def get_nearest_verb(doc: spacy.tokens.doc.Doc,
                     word: spacy.tokens.token.Token,
                     kw_span: spacy.tokens.span.Span,
                     go_left: bool = True) -> spacy.tokens.token.Token:
    """
    For sentence processed with spacy, word processed with spacy and root token of the word, finds the nearest verb to the left of the word

    Input
    -----
    doc : Doc
        sentence processed with spacy
    word: Token
        word processed with spacy
    kw_span: Span
        the whole keyword processed with spacy
    go_left: bool
        whether we search for the verb to the left of the given word or to the right

    Output
    ------
    token: Token
        token of the nearest verb to the left of the word

    """
    if go_left:
        m = -1
        n = 1
    else:
        m = 1
        n = -1
    for token in list(doc)[word.i - n::m]:
        if token.pos_ in ('VERB', 'AUX') and token not in kw_span:
            if token.dep_ == 'amod' and token.head.dep_ != 'ROOT':
                continue
            if token.lemma_ in verbs_stoplist:
                continue
            # if ',' in str(doc[token.i:word.i]):
            #   continue
            if (re.search(r'ed\b', token.text)
                    and ((token.head.text in ROLES
                         or (doc[token.i-1].text == ','
                             and doc[token.i-2].pos_ == 'ADJ')
                         or doc[token.i-1].pos_ == 'ADJ')
                         or (token.dep_ in ('conj','appos')
                         and not re.search(r'ed\b', token.head.text)))):
                continue
            if len(doc) >= token.i+2:
                if (token.text + ' ' + doc[token.i+1].text) in verbs_stoplist:
                    cj = ConjunctsHandler(doc)
                    cj.get_conjuncts(token)
                    if cj.all_verbs:
                        return cj.all_verbs[-2]
                    else:
                        continue
                else:
                    return token
            # if token in get_kw(doc, kw):
            #   continue
            else:
                return token
        elif token.pos_ == 'ADJ' and bool(re.search(r'ed\b|ing\b', token.text)) and token.head.dep_ in ('ROOT', 'nsubj') \
                and token not in kw_span and not (
                (doc[token.i - 1].text == ',' and doc[token.i - 2].pos_ == 'ADJ') or doc[token.i - 1].pos_ == 'ADJ'):
            return token


def get_prep(token: spacy.tokens.token.Token) -> str:
    if 'ADP' in list(map(lambda x: x.pos_, list(token.rights))):
        prep = list(filter(lambda x: x.pos_ == 'ADP', token.rights))[0]

        return prep if prep else ''


def get_dobj(token: spacy.tokens.token.Token) -> spacy.tokens.token.Token:
    if 'dobj' in list(map(lambda x: x.dep_, list(token.rights))):
        dobj = list(filter(lambda x: x.dep_ == 'dobj', token.rights))[0]
        return dobj


def get_compounds(token: spacy.tokens.token.Token) -> str:
    if 'compound' in list(map(lambda x: x.dep_, token.lefts)):
        comp = [tok.text for tok in token.lefts if tok.dep_ == 'compound']
        return ' '.join(comp) + ' ' + token.lemma_


def get_action_verb_objects(doc: spacy.tokens.doc.Doc,
                            verb: spacy.tokens.token.Token,
                            all_verbs: List[spacy.tokens.token.Token],
                            main_tok: spacy.tokens.token.Token,
                            kw_span: spacy.tokens.span.Span) -> List[str]:
    """
    For sentence processed with spacy and verb processed with spacy finds direct objects and prep.phrases of the verb

    Input
    -----
    doc : Doc
        sentence processed with spacy
    verb: Doc
        verb processed with spacy
    ...

    Output
    ------
    obj_text/pobj_text: list
        a list of strings (objects of the verb) with left children or [''] if there is no such objects of the verb

    """
    prep = ''
    if 'dobj' in list(map(lambda x: x.dep_, verb.rights)) or (all_verbs[-1].i < len(doc) - 1 and
        doc[all_verbs[-1].i + 1].pos_ == 'NOUN' and all_verbs[-1].lemma_ != 'be' and doc[all_verbs[-1].i + 1] != main_tok):
        if 'dobj' in list(map(lambda x: x.dep_, verb.rights)) and main_tok.i == kw_span[-1].i+1 and \
                [tok for tok in verb.rights if tok.dep_ == 'dobj'][0] == main_tok:
            return []
        if 'dobj' in list(map(lambda x: x.dep_, verb.rights)):
            objs = [tok for tok in verb.rights if tok.dep_ == 'dobj']
            obj = doc[objs[0].i:objs[-1].i+1]
        else:
            verb = all_verbs[-1]
            obj = doc[all_verbs[-1].i+1 + 1:all_verbs[-1].i+2]
        if obj.text not in kw_span.text:
            rights = list(chain([x for x in verb.rights if x.i > obj[-1].i], list(obj[-1].rights)))
            objs = [obj]
            obj_text = [obj.text]
            if obj[0].conjuncts or obj[-1].conjuncts:
                conjs = obj[0].conjuncts if obj[0].conjuncts else obj[-1].conjuncts
                conjs = [doc[conj.i:conj.i+1] for conj in conjs if
                         conj.i < main_tok.i and conj not in kw_span and conj != main_tok and conj not in main_tok.conjuncts]
                rights += list(chain.from_iterable([list(conj[-1].rights) for conj in conjs]))
                objs += conjs
                obj_text += [conj.text for conj in conjs]
            if not obj.text in objects_exclude:
                for i, obj in enumerate(objs):
                    lefts = [tok.text for tok in obj[0].subtree if
                             tok.i < obj[0].i and tok not in all_verbs and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts]
                    if lefts:
                        obj_text[i] = ' '.join(lefts) + ' ' + obj.text
            if 'ADP' in list(map(lambda x: x.pos_, rights)):
                prep = list(filter(lambda x: x.pos_ == 'ADP', rights))[0]
                for i, obj in enumerate(objs):
                    obj_text[i] += ' ' + prep.text
                if [tok for tok in prep.rights if
                    tok.dep_ == 'pobj' and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts]:
                    pobj = [tok for tok in prep.rights if
                            tok.dep_ == 'pobj' and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts][
                        0]
                    pobj_lefts = [tok.text for tok in pobj.subtree if (
                                tok.i <= pobj.i or tok.pos_ == 'ADP') and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts]
                    for i, obj in enumerate(objs):
                        obj_text[i] += ' ' + ' '.join(pobj_lefts)
            return obj_text
    if any(list(map(lambda word: word.pos_ == 'ADP', verb.rights))):
        prep = [word for word in verb.rights if word.pos_ == 'ADP'][0]
        if prep.i < main_tok.i:
            if [word for word in prep.rights if word.dep_ == 'pobj']:
                pobj = [word for word in prep.rights if word.dep_ == 'pobj'][0]
                if pobj not in kw_span and pobj != main_tok and pobj not in main_tok.conjuncts:
                    rights = list(chain([x for x in verb.rights if x.i > pobj.i], list(pobj.rights)))
                    pobjs = [pobj]
                    pobj_text = [pobj.text]
                    if pobj.conjuncts:
                        conjs = [conj for conj in pobj.conjuncts if
                                 conj.i < main_tok.i and conj not in kw_span and conj != main_tok and conj not in main_tok.conjuncts]
                        rights += list(chain.from_iterable([list(conj.rights) for conj in conjs]))
                        pobjs += conjs
                        pobj_text += [conj.text for conj in conjs]
                    if not pobj.text in objects_exclude:
                        for i, obj in enumerate(pobjs):
                            lefts = [tok.text for tok in pobj.subtree if
                                     tok.i <= pobj.i and tok not in all_verbs and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts]
                            pobj_text[i] = prep.text + ' ' + ' '.join(lefts)
                    if 'ADP' in list(map(lambda x: x.pos_, rights)):
                        prep2 = list(filter(lambda x: x.pos_ == 'ADP', rights))[0]
                        for i, pobj in enumerate(pobjs):
                            pobj_text[i] += ' ' + prep2.text
                        if [tok for tok in prep2.rights if
                            tok.dep_ == 'pobj' and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts]:
                            pobj2 = [tok for tok in prep2.rights if
                                     tok.dep_ == 'pobj' and tok not in kw_span and tok != main_tok and tok not in main_tok.conjuncts][
                                0]
                            pobj_lefts2 = [tok.text for tok in pobj2.subtree if (
                                        tok.i <= pobj2.i or tok.pos_ == 'ADP') and tok not in kw_span and tok not in main_tok.conjuncts]
                            for i, pobj in enumerate(pobjs):
                                pobj_text[i] += ' ' + ' '.join(pobj_lefts2)
                    return pobj_text


def get_other_verbs(doc: spacy.tokens.doc.Doc,
                    main_tok: spacy.tokens.token.Token,
                    kw_span: spacy.tokens.span.Span) -> List[Tuple]:
    '''

    For the given main token or verb with result/means/benefactive/indirect engagement/state flag, searches for another verb
    that should be non-action verb, so that we get 2 non-action verb tuples for the given keyword.
    The tuple comprises:
    - the index of the verb in the sentence Doc;
    - the text (string) of the verb token;
    - the text (string) of the verb object;
    - the text (string) of the preposition after the verb/before the keyword;
    - the flag that can be:
        - state
        - result
        - means
        - indirect engagement

    For each pair (verb, object) there is a separate tuple.

    '''
    verb2 = get_nearest_verb(doc, main_tok, kw_span)
    checker = VerbTypeChecker(doc)
    if verb2:
        flag2 = ''
        main_verb2, all_verbs2 = get_all_verbs(doc, verb2)
        p = get_prep(verb2)
        prep = p if p else get_prep_2(doc, p, kw_span)
        for method in verb_methods:
            flag = getattr(checker, method)(verb2)
            if flag:
                flag2 = flag
                if not all([getattr(checker, method)(v) == flag for v in all_verbs2]):
                    all_verbs2 = [verb2]
        if verb2.lemma_ == 'be':
            flag2 = 'state'
        verbs_text = get_verbs_text(doc, all_verbs2, main_tok)
        obj_text = get_action_verb_objects(doc, verb2, all_verbs2, main_tok, kw_span)
        if ((('dobj' in list(map(lambda x: x.dep_, verb2.rights)))
            or (all_verbs2[-1].i < len(doc) - 1
                and doc[all_verbs2[-1].i + 1].pos_ == 'NOUN'))
                and obj_text):
            if flag2: # and flag2 != flag:
                verbs = [(v[0], v[1], obj, prep, flag2) for (v, obj) in product(verbs_text, obj_text)]
            else:
                verbs = []
        else:
            if flag2: # and flag2 != flag:
                verbs = [(v[0], v[1], '', prep, flag2) for v in verbs_text]
            else:
                verbs = []
        return verbs


def get_verbs_text(doc: spacy.tokens.doc.Doc,
                   all_verbs: List[spacy.tokens.token.Token],
                   main_tok: spacy.tokens.token.Token) -> List[Tuple]:
    '''

    For a spacy verb token, returns the list of tuples, where each tuple consists of:
    - the index of the verb token
    - the full text of the verb which is formed as following:
        - if the verb is auxiliary, the semantic verb or adjective/noun is added with conjuncts
        - if the verb is semantic, the auxiliary verb is added if it comes before it

    '''
    verbs_text = []
    for v in all_verbs:
        v_text = v.lemma_
        if len(doc) > v.i:
            if doc[v.i - 2].lemma_ in ('be','do') and doc[v.i - 1].text == 'not':
                v_text = doc[v.i-2:v.i+1].text
        if v.pos_ == 'AUX' and doc[v.i+1].pos_ == 'VERB' and doc[v.i+1].tag_ == 'VBN':
            verbs_text.append((v.i, v.text + ' ' + doc[v.i + 1].text))
        elif v.tag_ == 'VBN' and 'AUX' in list(map(lambda x: x.pos_, v.lefts)):
            aux = [tok for tok in v.lefts if tok.pos_ == 'AUX'][0]
            verbs_text.append((v.i, aux.text + ' ' + v.text))
        elif v.tag_ == 'VBN' and v.conjuncts:
            aux = ''
            for conj in v.conjuncts:
                if 'AUX' in list(map(lambda x: x.pos_, conj.lefts)):
                    aux = [tok for tok in conj.lefts if tok.pos_ == 'AUX'][0]
            if aux:
                verbs_text.append((v.i, aux.text + ' ' + v.text))
            else:
                verbs_text.append((v.i, v_text))
        elif v.lemma_ == 'be' and (
                'ADJ' in list(map(lambda x: x.pos_, v.rights))):  # or 'NOUN' in list(map(lambda x: x.pos_, v.rights))):
            adj = [tok for tok in v.rights if tok.pos_ in ('ADJ', 'NOUN')][0]
            p = get_prep(adj)
            prep = ' ' + str(p) if p else ''
            verbs_text.append((v.i, v.text + ' ' + adj.text + prep))
        elif v.lemma_ == 'be' and 'one' in list(map(lambda x: x.text, v.rights)):
            one = list(filter(lambda x: x.text == 'one', v.rights))[0]
            if 'of' in list(map(lambda x: x.text, one.rights)):
                of = list(filter(lambda x: x.text == 'of', one.rights))[0]
                if 'NOUN' in list(map(lambda x: x.pos_, of.rights)):
                    noun = list(filter(lambda x: x.pos_ == 'NOUN', of.rights))[0]
                    if noun != main_tok:
                        compound = get_compounds(noun)
                        if compound:
                            verbs_text.append((v.i, v.text + ' ' + compound))
                        elif main_tok.i == noun.i - 1:
                            verbs_text.append((v.i, v.text + ' ' + main_tok.text + ' ' + noun.text))
                        else:
                            verbs_text.append((v.i, v.text + ' ' + noun.text))
                    else:
                        verbs_text.append((v.i, v_text))
                else:
                    verbs_text.append((v.i, v_text))
            else:
                verbs_text.append((v.i, v_text))
        elif v.lemma_ == 'be' and 'NOUN' in list(map(lambda x: x.pos_, v.rights)):
            noun = list(filter(lambda x: x.pos_ == 'NOUN', v.rights))[0]
            if noun != main_tok:
                compound = get_compounds(noun)
                if compound:
                    verbs_text.append((v.i, v.text + ' ' + compound))
                elif main_tok.i == noun.i - 1:
                    verbs_text.append((v.i, v.text + ' ' + main_tok.text + ' ' + noun.text))
                else:
                    verbs_text.append((v.i, v.text + ' ' + noun.text))
            else:
                verbs_text.append((v.i, v_text))
        else:
            verbs_text.append((v.i, v_text))
    return verbs_text


def check_verb_type(method: Callable,
                    flag: str,
                    doc: spacy.tokens.doc.Doc,
                    verb: spacy.tokens.token.Token,
                    main_verb: spacy.tokens.token.Token,
                    all_verbs: List[spacy.tokens.token.Token],
                    kw_span: spacy.tokens.span.Span) -> Tuple[str, List[Tuple], spacy.tokens.token.Token,
                                                            List[spacy.tokens.token.Token]]:
    verbs = []
    verb_type = method(verb)
    if verb_type:
        flag = verb_type
        if not all([method(v) == flag for v in all_verbs]):
            all_verbs = [verb]
            main_verb = verb
        other_verbs = get_other_verbs(doc, main_verb, kw_span)
        if other_verbs:
            verbs = other_verbs
    return flag, verbs, main_verb, all_verbs


def get_all_verbs(doc: spacy.tokens.doc.Doc,
                  verb: spacy.tokens.token.Token) -> Tuple[spacy.tokens.token.Token, List[spacy.tokens.token.Token]]:
    cj = ConjunctsHandler(doc)
    cj.get_conjuncts(verb)
    if verb.dep_ == 'conj' and not cj.all_verbs:
        cj.get_main_verb_token(verb)
    if cj.main_verb and cj.all_verbs:
        return cj.main_verb, cj.all_verbs
    return verb, [verb]


def get_subject_state(main_tok: spacy.tokens.token.Token,
                      verb: spacy.tokens.token.Token,
                      flag: str) -> str:
    if (main_tok.dep_ in ('nsubj', 'nsubjpass')
            or (main_tok.dep_ == 'ROOT'
                and main_tok.pos_ in ('NOUN', 'PROPN'))):
        return 'subject'
    elif verb.lemma_ == 'be':
        return 'state'
    return flag


def get_prep_2(doc: spacy.tokens.doc.Doc,
               prep: str,
               kw_span: spacy.tokens.span.Span) -> str:
    if not prep:
        if doc[kw_span[0].i-1].pos_ == 'ADP':
            prep = doc[kw_span[0].i-1].text
    return prep


def get_objects_text(doc: spacy.tokens.doc.Doc,
                     main_tok: spacy.tokens.token.Token,
                     kw: spacy.tokens.token.Token,
                     kw_span: spacy.tokens.span.Span,
                     all_verbs: List[spacy.tokens.token.Token]) -> dict:
    obj_text = {}
    for v in all_verbs:
        obj = get_action_verb_objects(doc, v, all_verbs, main_tok, kw_span)
        if obj:
            if kw.dep_ == 'pobj':
                obj = [o + ' ' + doc[kw_span[0].i-2:kw_span[0].i].text if (doc[kw_span[0].i-2].text not in o
                                                                           and kw_span[0].i > v.i) else o for o in obj]
        obj_text[v] = obj if obj else ['']
    return obj_text


def get_verbs_vbz_vbd(doc: spacy.tokens.doc.Doc,
                      verbs: List[Tuple],
                      verb: spacy.tokens.token.Token,
                      main_verb: spacy.tokens.token.Token) -> List:
    '''

    Returns empty list if the verb is 3d person singular or past tense, and some other conditions are met. It means that
    this verb is not a meaningful verb or does not refer to any agent

    '''
    if verb.tag_ == 'VBZ' and list(filter(lambda x: x.text == 'it' and x.dep_ == 'nsubj', verb.children)):
        return []
    if verb.tag_ == 'VBD':  # verb ends in '-ed'
        if main_verb.dep_ == 'amod':
            if doc[main_verb.i-1].pos_ == 'ADP' or (
                    main_verb.head.pos_ == 'VERB' and main_verb.dep_ not in ('ROOT', 'conj')):
                return []
    return verbs


def form_verb_object_tuples(verbs: List[Tuple],
                            verbs_text: List[Tuple],
                            obj_text: dict,
                            prep: str,
                            flag: str) -> List[Tuple]:
    if obj_text:
        obj_text_relevant = {verb:obj for verb, obj in obj_text.items() if obj != ['']}
        if len(obj_text_relevant) == 1:
            obj_text_relevant = list(obj_text_relevant.values())[0]
            verbs += [(v[0], v[1], obj, prep, flag) for (v, obj) in product(verbs_text, obj_text_relevant)]
        else:
            obj_text = list(obj_text.values())
            verbs += [(v[0], v[1], obj, prep, flag) for (v, obj_list) in zip(verbs_text, obj_text) for obj in obj_list]

    else:
        verbs += [(v[0], v[1], '', prep, flag) for v in verbs_text]
    return verbs


def get_action_verb_tuples(doc: spacy.tokens.doc.Doc,
                           main_tok: spacy.tokens.token.Token,
                           kw: spacy.tokens.token.Token,
                           verb: spacy.tokens.token.Token,
                           kw_span,
                           prep: str = '',
                           flag: str = 'action') -> List[Tuple]:
    """

    For sentence processed with spacy, verb, keyword, main token processed with spacy gets a list of tuples that
    comprise:
    - the index of the verb in the sentence Doc;
    - the text (string) of the verb token;
    - the text (string) of the verb object;
    - the text (string) of the preposition after the verb/before the keyword;
    - the flag that can be:
        - action
        - state
        - subject
        - result
        - means
        - indirect engagement

    For each pair (verb, object) there is a separate tuple.

    """
    checker = VerbTypeChecker(doc)
    verbs = []
    prep = get_prep_2(doc, prep, kw_span)
    main_verb, all_verbs = get_all_verbs(doc, verb)

    flag = get_subject_state(main_tok, verb, flag)
    if flag == 'action':
        for method in verb_methods:
            flag, verbs, main_verb, all_verbs = check_verb_type(getattr(checker, method), flag, doc, verb, main_verb,
                                                                all_verbs, kw_span)

    verbs_text = get_verbs_text(doc, all_verbs, main_tok)
    obj_text = get_objects_text(doc, main_tok, kw, kw_span, all_verbs)

    verbs = form_verb_object_tuples(verbs, verbs_text, obj_text, prep, flag)
    verbs = get_verbs_vbz_vbd(doc, verbs, verb, main_verb)
    return verbs


def get_other_result_tuple(doc: spacy.tokens.doc.Doc,
                           verb: spacy.tokens.token.Token,
                           main_verb: spacy.tokens.token.Token,
                           all_verbs: List[spacy.tokens.token.Token],
                           prep: str,
                           kw_span: spacy.tokens.span.Span) -> List[Tuple]:
    checker = VerbTypeChecker(doc)
    verb2 = get_nearest_verb(doc, main_verb, kw_span)
    if verb2:
        link_verb = main_verb.head.text if main_verb.head.pos_ != 'VERB' else ''
        link = ' '.join(
            [link_verb, 'to', main_verb.text, ' '.join([tok.text for tok in main_verb.rights if tok.pos_ != 'VERB'])])
        objects = get_verb_objects(doc, verb2)
        obj = ' ' + objects[0] if objects else ''
        verbs_text2 = []
        if verb2.tag_ == 'VBZ' and list(filter(lambda x: x.text == 'it' and x.dep_ == 'nsubj', verb2.children)):
            action_phrases = []
        else:
            if checker.isResultVerb(main_verb):
                if verb2.lemma_ == 'be' and 'ADJ' in list(map(lambda x: x.pos_, verb2.rights)):
                    adj = [tok.text for tok in verb2.rights if tok.pos_ == 'ADJ'][0]
                    verbs_text2.extend(
                        [(verb2.i, verb2.text + ' ' + adj + obj + ' to ' + main_verb.lemma_) for v in all_verbs])
                elif verb2.tag_ == 'VBN' and 'AUX' in list(map(lambda x: x.pos_, verb2.lefts)):
                    aux = [tok for tok in verb2.lefts if tok.pos_ == 'AUX'][0]
                    verbs_text2.extend(
                        [(verb2.i, aux.text + ' ' + verb2.text + obj + ' to ' + main_verb.lemma_) for v in all_verbs])
                else:
                    verbs_text2.extend([(verb2.i, verb2.text + obj + ' to ' + main_verb.lemma_) for v in all_verbs])
            else:
                if verb2.lemma_ == 'be' and 'ADJ' in list(map(lambda x: x.pos_, verb2.rights)):
                    adj = [tok.text for tok in verb2.rights if tok.pos_ == 'ADJ'][0]
                    verbs_text2.extend(
                        [(verb2.i, verb2.text + ' ' + adj + obj + ' ' + main_verb.head.text + ' ' + main_verb.lemma_)
                         for v in all_verbs])
                elif verb2.tag_ == 'VBN' and 'AUX' in list(map(lambda x: x.pos_, verb2.lefts)):
                    aux = [tok for tok in verb2.lefts if tok.pos_ == 'AUX'][0]
                    verbs_text2.extend([(verb2.i,
                                         aux.text + ' ' + verb2.text + obj + ' ' + main_verb.head.text + ' ' + main_verb.lemma_)
                                        for v in all_verbs])
                else:
                    verbs_text2.extend(
                        [(verb2.i, verb2.text + obj + ' ' + main_verb.head.text + ' ' + main_verb.lemma_) for v in
                         all_verbs])
            action_phrases = get_verb_tuples(doc, main_verb, all_verbs, verbs_text2, prep=prep, link=link)
        return action_phrases


def get_verb_tuples(doc: spacy.tokens.doc.Doc,
                    verb: spacy.tokens.token.Token,
                    all_verbs: List[spacy.tokens.token.Token],
                    verbs_text: List[Tuple],
                    prep: str = '',
                    link: str = '') -> List[Tuple]:
    verb_phrases = []
    objects = get_verb_objects(doc, verb)
    if objects:
        last_verb_object = objects[0]
        for i, v in enumerate(all_verbs):
            b = get_verb_objects(doc, v)
            if b:
                for phrase in b:
                    if phrase not in [(''), '']:
                        verb_phrases.append((verbs_text[i][0], verbs_text[i][1], phrase, prep, link))
            else:
                verb_phrases.append((verbs_text[i][0], verbs_text[i][1], last_verb_object, prep, link))
    else:
        verb_phrases.extend([(v[0], v[1], '', prep, link) for v in verbs_text])
    return verb_phrases


def get_verb_objects(doc: spacy.tokens.doc.Doc,
                     verb: spacy.tokens.token.Token) -> List[str]:
    """
    For sentence processed with spacy and verb processed with spacy finds direct objects and prep.phrases of the verb

    Input
    -----
    doc : Doc
        sentence processed with spacy
    verb: Doc
        verb processed with spacy

    Output
    ------
    answer: list
        a list of tokens (objects of the verb) with left children or [''] if there is no such objects of the verb

    """
    answer = []
    if any(list(map(lambda word: word.dep_ == 'dobj', verb.rights))):
        obj = [word for word in verb.rights if word.dep_ == 'dobj'][0]
        obj_lefts = [w.text for w in obj.subtree if w.i <= obj.i and w.pos_ != 'DET']
        if obj.pos_ != 'NOUN':
            if [w for w in obj.subtree if w.pos_ == 'NOUN']:
                obj = [w for w in obj.subtree if w.pos_ == 'NOUN'][0]
                obj_lefts = [w.text for w in doc[verb.i + 1:obj.i + 1] if w.pos_ != 'DET']
        # if obj != kw_root:
        answer.append(' '.join(obj_lefts))
        if obj.conjuncts:
            other_objects = list(obj.conjuncts)
            if other_objects:
                answer.extend([' '.join([w.text for w in word.subtree if w.i <= word.i and w.pos_ != 'DET']) for word in
                               other_objects])
                obj = other_objects[-1]
        # if any(list(map(lambda word: (word.pos_ == 'ADP') and (word.i == obj.i + 1), list(obj.rights)+list(verb.rights)))):
        #   answer[-1] += ' ' + [word.text for word in list(obj.rights)+list(verb.rights) if (word.pos_ == 'ADP') and (word.i == obj.i + 1)][0]

    elif any(list(map(lambda word: word.pos_ == 'ADP', verb.rights))):
        prep = [word for word in verb.rights if word.pos_ == 'ADP'][0]
        if [word for word in prep.rights if word.dep_ == 'pobj']:
            pobj = [word for word in prep.rights if word.dep_ == 'pobj'][0]
            pobj_lefts = [w.text for w in pobj.subtree if w.i <= pobj.i and w.pos_ != 'DET']
            if pobj.pos_ != 'NOUN':
                if [w for w in pobj.subtree if w.pos_ == 'NOUN']:
                    pobj = [w for w in pobj.subtree if w.pos_ == 'NOUN'][0]
                    pobj_lefts = [w.text for w in doc[verb.i + 2:pobj.i + 1] if w.pos_ != 'DET']
            # if pobj != kw_root:
            answer.append(prep.text + ' ' + ' '.join(pobj_lefts))
            if pobj.conjuncts:
                other_pobjects = list(pobj.conjuncts)
                if other_pobjects:
                    answer.extend(
                        [prep.text + ' ' + ' '.join([w.text for w in word.subtree if w.i <= word.i and w.pos_ != 'DET'])
                         for word in other_pobjects])
                    obj = other_pobjects[-1]
            # if any(list(map(lambda word: (word.pos_ == 'ADP') and (word.i == pobj.i + 1), list(pobj.rights)+list(verb.rights)))):
            #   answer[-1] += ' ' + [word.text for word in list(pobj.rights)+list(verb.rights) if (word.pos_ == 'ADP') and (word.i == pobj.i + 1)][0]
        else:
            answer.append(prep.text)
    elif verb.dep_ in ('amod', 'compound'):
        obj = verb.head
        # if obj != kw_root:
        answer.append(obj.text)
        if obj.conjuncts:
            other_objects = list(obj.conjuncts)
            if other_objects:
                answer.extend([word.text for word in other_objects])
    return answer


def get_ccomps(doc: spacy.tokens.doc.Doc,
               verb: spacy.tokens.token.Token) -> Tuple[List[spacy.tokens.token.Token], List[Tuple]]:
    all_verbs = []
    verbs_text = []
    for tok in verb.rights:
        if tok.dep_ in ('ccomp', 'conj'):
            all_verbs.append(tok)
            verbs_text.append((verb.i, verb.lemma_ + ' to ' + tok.text))
            cj = ConjunctsHandler(doc)
            cj.get_conjuncts(tok)
            if cj.all_verbs:
                all_verbs.extend(cj.all_verbs)
                verbs_text.extend([(verb.i, verb.lemma_ + ' to ' + conj.text) for conj in cj.all_verbs])
            if 'conj' in list(map(lambda x: x.dep_, tok.rights)):
                conjs = [x for x in tok.rights if x.dep_ == 'conj']
                all_verbs.extend(conjs)
                verbs_text.extend([(verb.i, verb.lemma_ + ' to ' + conj.text) for conj in conjs])
    if not all_verbs:
        all_verbs = [verb]
    if not verbs_text:
        verbs_text = [(verb.i, verb.lemma_)]
    return all_verbs, verbs_text


# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


def convert(word: str,
            from_pos: str,
            to_pos: str) ->  List[Tuple]:
    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and \
                    s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and \
                    l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result
