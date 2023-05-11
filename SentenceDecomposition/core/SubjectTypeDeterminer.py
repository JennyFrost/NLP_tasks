import spacy

PREDEFINED_SBJ = {'SOMEONE': {'community'},
                  'COMPANY': {'company', 'companies', 'inc', 'llc', 'services', 'platform', 'employees', 'agency',
                              'organization', 'organizations', 'organisation', 'organisations', 'firm', 'firms', 'us',
                              'solution', 'solutions', 'startup', 'group'},
                  'CompanyPronoun': {'we', 'it'},
                  'PERSON': {'i', 'my', 'he', 'she', 'whom', 'its', 'her', 'his', 'anybody', 'anyone', 'anything',
                             'each one', 'everybody', 'everyone', 'nobody', 'no one', 'one', 'somebody', 'someone',
                             'yourself'},
                  'TEAM': {'team', 'teams', 'Team'}
                  }


# --------------
def get_token_phrase(token_indx: int,
                     doc: spacy.tokens.doc.Doc) -> spacy.tokens.span.Span:
    token = doc[token_indx]
    subtree = list(token.subtree)
    start = subtree[0].i
    end = subtree[-1].i + 1
    return doc[start:end]


def has_atrr(doc: spacy.tokens.doc.Doc,
             verb_token: spacy.tokens.token.Token):
    """
    check if verb token has attr dependency and return it, return empty list otherwise.
    """
    t = []  # empty list
    tok_dep = [t.dep_ for t in verb_token.children]
    if "attr" in tok_dep:
        t = get_token_by_dependency(doc, "attr")
    return t


def get_token_by_dependency(doc: spacy.tokens.doc.Doc, dependency: str) -> list:
    tokens = [tok for tok in doc if tok.dep_ == dependency]
    return tokens


# -------------------------------------------------
def verb_procedure(doc: spacy.tokens.doc.Doc,
                   related_verb_indx: int,
                   predefined_subj: dict) -> str:
    subject_type = "Undefined"

    verb_token = doc[related_verb_indx]

    if verb_token.dep_ == "ROOT":
        t = has_atrr(doc, verb_token)
        if verb_token.pos_ == "AUX" and len(t) != 0:
            subject_field = predefined_sbj_process(predefined_subj, t[0].text, ['COMPANY', 'TEAM', 'SOMEONE'])
            if subject_field is not None:
                subject_type = subject_field

    return subject_type


def decompose_entity_type(ent_type: str) -> str:
    tmp = ent_type.split('___')
    if len(tmp) == 1:
        return ent_type
    else:
        return tmp[0]


def get_full_subject_name_list(sbj_indx: int,
                               NENP_dict: dict,
                               doc: spacy.tokens.doc.Doc, ) -> list:
    """
    This function takes subj sbj_indx, npne dict,
        and sent doc and returns the subject fullName or the single word subject.
    Arguments:
    ----------
        sbj_indx: the subject index in nlp doc.
        NENP_dict: a dictionary of coded noun phrases and entities. The key is the NP text for example (NP1,NP2...)
                    and the value is a tuple that contains the noun phrases string and another tuple
                    that contains the start and end positions of NP string in the sentence.
        doc : nlp spcay sentence doc
    Returns:
    ----------
        sbj_full_name: A list of tuples where each tuple contains (subject type, entity type)
    """
    sbj_full_name = []
    entity_type = None

    for flag, values in NENP_dict.items():
        for phrase in values:
            if phrase['root_index'] == sbj_indx:  # to prove
                if phrase['ent_type'] != '':
                    sbj_full_name.append((phrase['phrase'], phrase['ent_type']))
                else:
                    sbj_full_name.append((phrase['phrase'], None))

    if len(sbj_full_name) == 0:
        sbj_full_name.append((doc[sbj_indx].text, None))
    return sbj_full_name


# Function takes the plist, sbj-name and a list of fields name for the pred-list in order not to search all fields
def predefined_sbj_process(predefined_subj: dict,
                           subject_name: str,
                           fields: list):
    subject_field = None
    if fields is None:
        for k, v in predefined_subj.items():
            if subject_name in v or subject_name.lower() in v:
                subject_field = k
                break
            else:
                subject_field = None
    else:
        for lst in fields:
            if subject_name in predefined_subj[lst] or subject_name.lower() in predefined_subj[lst]:
                subject_field = lst
                break
            else:
                subject_field = None

    return subject_field


def process_sbj_type(sbj_indx: int,  # or None
                     predefined_subj: dict,
                     doc: spacy.tokens.doc.Doc,
                     NENP_dict: dict,
                     verb_indx: int) -> list:

    subject_type = "Undefined"

    if sbj_indx is None:
        return [subject_type]

    subject_details_list = []
    subject_name_list = get_full_subject_name_list(sbj_indx, NENP_dict, doc)

    # loop through the subject names list

    for subject_name, entity_type in subject_name_list:

        if entity_type is not None:
            subject_type = entity_type
        else:
            # NOT ENTITY
            # search subject (full, single) in the company list only
            subject_field = predefined_sbj_process(predefined_subj, subject_name, ['COMPANY', 'TEAM', 'SOMEONE'])
            if subject_field is not None:
                subject_type = subject_field

            else:
                # here if single and/or multi-word subject not found
                if len(subject_name.split()) > 1:  # subject is simple np

                    # we will search the single noun root here
                    subj_root = doc[sbj_indx]
                    subject_field = predefined_sbj_process(predefined_subj, subj_root.text,
                                                           ['COMPANY', 'TEAM', 'SOMEONE'])
                    pron_part = subject_name.split()[0].lower()

                    if subject_field is not None and pron_part not in ['my', 'her', 'his']:
                        # not our team not my team
                        subject_type = subject_field

                    elif pron_part in ['my', 'her', 'his']:
                        subject_type = 'PERSON'

                    elif pron_part in ['our']:
                        subject_type = 'COMPANY'  # or may be company

                    elif pron_part in ['their', 'its']:
                        subject_type = 'SOMEONE'  # or may be company

                    elif subject_field is None and pron_part not in {'our', 'their', 'its', 'my', 'her', 'his'}:
                        subject_type = verb_procedure(doc, verb_indx, predefined_subj)

                else:
                    # subject is a single word
                    subj_pos = doc[sbj_indx].pos_  # maybe noun or pronoun
                    if subj_pos == "PRON":
                        if subject_name.lower() == 'we':
                            subject_type = 'COMPANY'
                        else:
                            subject_field = predefined_sbj_process(predefined_subj, subject_name.lower(),
                                                                   ['CompanyPronoun', 'PERSON'])
                            if subject_field == 'PERSON':
                                subject_type = 'PERSON'
                            else:
                                # subject_field = None or not we. check the related noun
                                subject_type = verb_procedure(doc, verb_indx, predefined_subj)

                    else:
                        # single noun was checked in pred-list so here only check if it has attr in pred-list
                        subject_type = verb_procedure(doc, verb_indx, predefined_subj)

        subject_details_list.append(subject_type)

    return subject_details_list


class SubjectTypeDeterminer:
    def __init__(self):
        pass

    @staticmethod
    def get_subject_type(subject_indx: int,  # or None
                         doc: spacy.tokens.doc.Doc,
                         NENP_dict: dict,
                         verb_indx: int) -> str:
        sbj_type = process_sbj_type(subject_indx, PREDEFINED_SBJ, doc, NENP_dict, verb_indx)
        return sbj_type[0]  # decompose_entity_type(sbj_type[0])
