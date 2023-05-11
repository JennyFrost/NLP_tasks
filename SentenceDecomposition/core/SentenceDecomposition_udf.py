import traceback
from typing import Tuple, Callable, List, Dict, Optional

import numpy as np
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.token import Token as SpacyToken

from SubjectTypeDeterminer import SubjectTypeDeterminer
from expertiseIn import ExpertiseChecker
from getActionsForMeans import getActionsForMeans
from getActionsForResult import getActionsForResult
from getActionsforKeyword import getActionsforKeyword
from oneforce_logger import OneForceLogger
from processNoVerbs import processNoVerbs
from oneforce_swagger_docs import SentenceDecompositionDoc

DictStr = Dict[str, str]
TupleVb = Tuple[int | str, str, str, str, str]
DictSL = Dict[str, str | List[str | TupleVb | dict]]

SUBJECT_ERROR_FLAG = 'error-subject-no-verb'
NO_VERB_INDX = 'error-no-verb-index'

ExpCheck = ExpertiseChecker()
sbj_type_det = SubjectTypeDeterminer()
logger = OneForceLogger('SD-udf')

DATA_FORMAT_COLS = ['foundKeyword', 'improvedKeyword', 'whereFound', 'verb', 'verbPrep', 'additionalObject', 'object',
                    'resultLink', 'resultVerb', 'resultVerbPrep', 'resultAdditionalObject', 'resultObject', 'meansLink',
                    'meansVerb', 'meansVerbPrep', 'meansAdditionalObject', 'meansObject', 'indirectLink',
                    'indirectVerb', 'indirectVerbPrep', 'indirectAdditionalObject', 'indirectObject', 'extractedLink',
                    'extractedVerb', 'extractedObject', 'benefactive', 'benefactiveLink', 'role', 'subjectToken',
                    'subjectType', 'isPassive', 'agentInfo', 'improvedKeywordAddInfo']


# =======================================================
# HELP FUNCTIONS
# =======================================================
def get_short_flag(flag: str) -> str:
    # tmp = flag.split('+')[1]
    return 'indirect' if flag == 'indirect engagement' else flag


def add_flag_object(dict_: DictStr,
                    kw_dict: DictSL,
                    flag: str) -> DictStr:
    if flag != 'benefactive':
        dict_[f'{flag}Object'] = kw_dict['improvedKeyword']
    else:
        dict_['benefactive'] = kw_dict['improvedKeyword']
    return dict_


def decomp_verb_by_part(verb: TupleVb,
                        part: str) -> str:
    if part == 'indx':
        return str(verb[0])
    elif part == 'verb':
        return str(verb[1])
    elif part == 'addobj':
        return str(verb[2])
    elif part == 'prep' or part == 'link':
        return str(verb[3])
    else:
        logger.error(f"decomp_verb_by_part - wrong part {part} for verb: {str(verb)}")
        return str(-1)


def fix_verb_link(v1: str | SpacyToken) -> str:
    return str(v1).replace('action', '').replace('means', '').replace('result', '').replace('indirect engagement', '')


def flatten_list(_2d_list: list) -> list:
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if isinstance(element, list):
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def fill_dict(dict_: DictStr,
              keys_: List[str],
              fill_str: str) -> DictStr:
    tmp_keys = set(dict_.keys())
    for k in keys_:
        if k not in tmp_keys:
            dict_[k] = fill_str
    return dict_


# =======================================================
# STEP 1 - EXPERTISE DETERMINING
# =======================================================
def check_expertise_in(sentence_doc: SpacyDoc,
                       improved_kw_tok: SpacyToken,
                       found_kw_tok: str) -> Tuple:
    expertise, expertise_kw = ExpCheck.checkExpertise(sentence_doc, improved_kw_tok, found_kw_tok)
    return expertise, expertise_kw


# =======================================================
# STEP 2 - VERBS DETERMINING
# =======================================================
def get_actions_for_verb(func_get_action: Callable,
                         verb: TupleVb,
                         doc: SpacyDoc,
                         akw_indices: list) -> List[TupleVb]:
    if str(verb[1]) == '':
        cur_action = func_get_action(doc, akw_indices)
    else:
        cur_action = func_get_action(doc, doc[verb[0]])
    return cur_action


def run_get_actions(func_get_action: Callable,
                    flag: str,
                    doc: SpacyDoc,
                    akw_indices: list,
                    cur_verbs: List[TupleVb],
                    kw_dict: DictSL) -> DictSL:
    # real_action = []
    # for v in cur_verbs:
    #     cur_action = get_actions_for_verb(func_get_action, v, doc, akw_indices)
    #     real_action.append(cur_action)
    real_action = [get_actions_for_verb(func_get_action, v, doc, akw_indices) for v in cur_verbs]
    if len(real_action[-1]) > 0:
        kw_dict['action'] = real_action[-1]
        kw_dict[flag] = cur_verbs
        kw_dict['FLAG'] = f'action+{flag}'
    else:
        kw_dict[flag] = cur_verbs
        kw_dict['FLAG'] = flag
    return kw_dict


def run_get_actions_2(func_get_action: Callable,
                      cur_verb_flag: str,
                      cur_flags: List[str],
                      verb: TupleVb,
                      doc: SpacyDoc,
                      akw_indices: list,
                      # cur_verbs: list,
                      kw_dict: DictSL) -> Tuple[DictSL, List[str]]:
    cur_action = get_actions_for_verb(func_get_action, verb, doc, akw_indices)

    if len(cur_action) > 0:
        kw_dict['action'] = cur_action
        kw_dict[cur_verb_flag].append(verb)
        cur_flags.append('action+' + cur_verb_flag)
    else:
        # kw_dict['action'] = [v]
        # cur_flags.append('action')
        kw_dict[cur_verb_flag] = [verb]
        cur_flags.append(cur_verb_flag)

    return kw_dict, cur_flags


def get_verbs_for_kws(sentence_doc: SpacyDoc,
                      skw_akw_list: List[DictSL]) -> List[DictSL]:
    """
    Output params:
    ----------
        - kws_list: list - list of dicts (kw_dict), containing info about each kw found in the sentence.
                    kw_dict can be by following structure (key -value):
                        'improveKeyword' - AKW str
                        'foundKeyword' - SKW str
                        'experiseIn' - result of expertise checker
                        'FLAG' - kw type; possible values:
                                    -- action/ means/ result/
                                    -- action+type_1 (in case kw has type type_1 and action verb(s) was found)
                                    -- action+type1_type2
                                    -- experise
                        'REAL_FLAG' - 
                        'ALL_VERBS' - actions verbs from getActionsforKeyword
                        'special' - results from processNoVerbs/ EnumProcessing if exist
                        'action' - action verbs  if exist
                        'means' - means verbs  if exist
                        'result' - result verbs  if exist
                        'action' - action verbs  if exist
    """
    kws_list = []
    if len(skw_akw_list) > 0:
        try:
            for skw_akw in skw_akw_list:
                kw_dict = {'improvedKeyword': skw_akw['akw_text'],
                           'foundKeyword': skw_akw['skw_text'],
                           'expertise': skw_akw['expertise'],
                           'akw_indices': skw_akw['akw_indices'],
                           'akw_pos': skw_akw['akw_pos'],
                           'akw_head_text': skw_akw['akw_head_text'],
                           }

                if skw_akw['expertise']:
                    kw_dict['FLAG'] = 'expertise'
                    kws_list.append(kw_dict)
                else:
                    cur_verbs = getActionsforKeyword(sentence_doc, skw_akw['akw_indices'])
                    kw_dict['ALL_VERBS'] = cur_verbs
                    if len(cur_verbs) == 0:
                        # (flag,list of verbs string, prep str) or ('', [], '')
                        tmp = processNoVerbs(sentence_doc, skw_akw['akw_indices'])
                        kw_dict['special'] = tmp
                        kw_dict['FLAG'] = 'no-verbs'
                        kws_list.append(kw_dict)
                    else:
                        flags = set(list(map(lambda x: x[-1], cur_verbs)))
                        flags_list = list(flags)
                        # 1 FLAG
                        if len(flags) == 1:
                            if flags == {'subject'}:
                                kw_dict[flags_list[0]] = cur_verbs
                                kw_dict['FLAG'] = 'subject'
                            elif flags_list[0] in {'action', 'state'}:
                                kw_dict[flags_list[0]] = cur_verbs
                                kw_dict['FLAG'] = flags_list[0]
                            elif flags_list[0] == 'means':
                                kw_dict = run_get_actions(getActionsForMeans, 'means', sentence_doc,
                                                          skw_akw['akw_indices'], cur_verbs, kw_dict)
                            elif flags_list[0] in {'result', 'indirect engagement', 'benefactive'}:
                                kw_dict = run_get_actions(getActionsForResult, flags_list[0], sentence_doc,
                                                          skw_akw['akw_indices'], cur_verbs, kw_dict)
                            else:
                                logger.error(f'get_verbs_for_kws - unexpected flag; 1 flag: {flags_list[0]}')
                                return []
                        # 2 FLAGS
                        elif len(flags) == 2:
                            kw_dict[str(flags_list[0])] = []
                            kw_dict[str(flags_list[1])] = []
                            cur_flags = []
                            real_flags = []
                            # [(verbIndx, verbToken, prepToken, object,link)] or []
                            for j, verb in enumerate(cur_verbs):
                                cur_verb_flag = verb[-1]
                                real_flags.append(cur_verb_flag)
                                if cur_verb_flag in {'action', 'state'}:
                                    cur_flags.append(cur_verb_flag)
                                    kw_dict[cur_verb_flag].append(verb)
                                elif cur_verb_flag in {'result', 'benefactive', 'indirect engagement'}:
                                    if j == 0:
                                        kw_dict, cur_flags = run_get_actions_2(getActionsForResult, cur_verb_flag,
                                                                               cur_flags, verb, sentence_doc,
                                                                               skw_akw['akw_indices'],  # cur_verbs,
                                                                               kw_dict)

                                    else:
                                        cur_flags.append(cur_verb_flag)
                                        kw_dict[cur_verb_flag].append(verb)

                                elif cur_verb_flag in {'means'}:
                                    if j == 0:
                                        kw_dict, cur_flags = run_get_actions_2(getActionsForMeans, cur_verb_flag,
                                                                               cur_flags, verb, sentence_doc,
                                                                               skw_akw['akw_indices'],  # cur_verbs,
                                                                               kw_dict)

                                    else:
                                        cur_flags.append(cur_verb_flag)
                                        kw_dict[cur_verb_flag].append(verb)
                                else:
                                    logger.error(f'''get_verbs_for_kws - unexpected flag;
                                                    2 flags: {"|".join(flags_list)}''')
                                    return []

                            kw_dict['FLAG'] = '_'.join(cur_flags)
                            kw_dict['REAL_FLAG'] = '_'.join(real_flags)
                        else:
                            logger.error(f'get_verbs_for_kws - unexpected flag; not 2 flags: {"|".join(flags_list)}')
                            return []
                        kws_list.append(kw_dict)
        except Exception as e:
            logger.error(f"get_verbs_for_kws - unexpected error")
            logger.error(str(e))
            logger.error(traceback.format_exc())
    return kws_list


# =======================================================
# STEP 3 - SUBJECT DETERMINING
# =======================================================
# def get_kw_indx_word(skw_akw_list: list) -> Tuple[list, list]:
#     temp1 = list(map(lambda x: (x['foundKeyword'], x['improvedKeyword']), skw_akw_list))
#     improved_indx = list(
#         filter(lambda x: x != -100, map(lambda x: x[1].i if not isinstance(x[1], str) else -100, temp1)))
#     found_kw = list(map(lambda x: str(x[0]), temp1))
#     return improved_indx, found_kw


# def make_verbs_set(verbs_indxs: list) -> list:
#     tmp = list(filter(lambda x: list(str(x)) != ['[', ']'], verbs_indxs))
#     if len(tmp) > 0:
#         verbs_ = {item for item in verbs_indxs}
#         verbs_ = list(verbs_)
#         verbs_.sort()
#         return verbs_
#     else:
#         return []


def get_verbs_from_kw_dict(kw_dict: DictSL) -> List[TupleVb]:
    flags = kw_dict['FLAG'].split('_')
    verbs = []
    if len(flags) == 1:
        if kw_dict['FLAG'] not in {'no-verbs', 'expertise'}:
            if '+' in kw_dict['FLAG']:
                verbs = kw_dict[kw_dict['FLAG'].split('+')[0]]
            else:
                verbs = kw_dict[kw_dict['FLAG']]
    elif len(set(flags)) in [2, 3]:
        if '+' in flags[0]:
            verbs = kw_dict[flags[0].split('+')[0]]
        else:
            verbs = kw_dict[flags[0]]
    else:
        logger.error(f'get verbs_from_kw_dict - unexpected flags: {"|".join(flags)}')
        return []
    if len(verbs) > 0:
        return list(map(lambda y: y[0], verbs))
    else:
        return []


def get_all_indx_verbs(skw_akw_list: List[DictSL]) -> List[DictSL]:
    tmp = list(map(lambda x: get_verbs_from_kw_dict(x), skw_akw_list))
    tmp = list(set(flatten_list(tmp)))
    tmp = list(filter(lambda y: str(y) != '', tmp))
    return tmp


def add_empty_sbj(d: DictSL) -> DictSL:
    d['subjectTokens'] = []
    d['subjectTypes'] = []
    return d


def process_sbj_type(sbj_type: str, profile_id: str) -> str:
    tmp = sbj_type.split('=')
    if len(tmp) == 1:
        return sbj_type
    else:
        if tmp[1] == profile_id:
            return tmp[0]
        else:
            return tmp[0] + "_MAYBE"


def get_sbj_type(sbj_ph: dict | str, sentence_doc: SpacyDoc, NENP_dict: dict,
                 verb: TupleVb | Dict[str, str | TupleVb | dict],
                 profile_id: str) -> str:
    if isinstance(verb, dict):
        if isinstance(verb['passed'][0], int):
            if isinstance(verb['real'], str):
                return 'Undefined'
            return process_sbj_type(sbj_type_det.get_subject_type(sbj_ph['sbj_indx'], sentence_doc,
                                                                  NENP_dict, verb['real']['phrase_head_in']),
                                    profile_id)
    return SUBJECT_ERROR_FLAG


def get_sbj(sentence_doc: SpacyDoc, sbj_phrase: dict) -> str:
    if sbj_phrase['sbj_indx'] is not None:
        return sentence_doc[sbj_phrase['phrase_start']: sbj_phrase['phrase_end']].text
    return 'None'


def get_vb_info(sentence_doc: SpacyDoc,
                verb: TupleVb | Dict[str, str | TupleVb | dict]) -> Dict[str, str | bool | None]:
    vb = {'phrase': None,
          'is_passive': False,
          'agent': None}
    if isinstance(verb, tuple):
        return vb
    tmp = verb['real']
    if isinstance(tmp, str):
        return vb
    vb['phrase'] = sentence_doc[tmp['phrase_start']: tmp['phrase_end']].text
    vb['is_passive'] = tmp['passive_info']['is_passive']
    vb['agent'] = sentence_doc[tmp['passive_info']['agent_info']['phrase_start']: tmp['passive_info']['agent_info'][
        'phrase_end']].text if tmp['passive_info']['agent_info']['is_found'] else None
    return vb


def add_sbj_type_update_sbj_tok(d: DictSL, sentence_doc: SpacyDoc,
                                NENP_dict: dict,
                                verbs: List[TupleVb | Dict[str, str | TupleVb | dict]],  # dict from preproc info
                                profile_id: str) -> DictSL:
    d['subjectTypes'] = [get_sbj_type(sbj_ph, sentence_doc, NENP_dict, verbs[i], profile_id) for i, sbj_ph
                         in enumerate(d['subjectTokens'])]
    d['subjectTokens'] = [get_sbj(sentence_doc, sbj_ph) if isinstance(sbj_ph, dict) else str(sbj_ph) for sbj_ph
                          in d['subjectTokens']]  # dict from preproc info
    d['realVerbs'] = [get_vb_info(sentence_doc, vb) for vb in d['realVerbs']]
    return d


def verb_range(verb_info: dict) -> range:
    return range(verb_info['phrase_start'], verb_info['phrase_end'])


def error_verb_not_found_in_preproc(vb_indx: str | int, preproc_verbs: List[dict]):
    """
        The function codes the case when determined verb in the sentence found
        wasn't found at preprocessing steps (verbs and subjects determiner).
        For example:
            verb with index N was determined in the sentence. Preprocessing
            returned the whole list of verb phrases in it with start and end indexes: i and j, k and l.
            It means that the sentence contains 2 verb phrases: [i:j] and [k:l]. But neither of
            them contains index N. Therefore the error raise.
        In such case, the code of the error will be constructed by following rule:
        1) code phrase named by global value SUBJECT_ERROR_FLAG
        2) index of determined verb N
        3) list of all verbs phrases from preprocessing joined by | in following format:
            'verb phrase start index'-'verb phrase end index'.
        4) join 1-3 step by '_'
        The result for example:
            "@SUBJECT_ERROR_FLAG@_N_i-j|k-l"
    """
    tmp = list(map(lambda x: str(x['phrase_start']) + '-' + str(x['phrase_end']), preproc_verbs))
    error = f"{SUBJECT_ERROR_FLAG}_{vb_indx}_{'|'.join(tmp)}"
    return error


def get_subjects_for_kws_verbs(doc: SpacyDoc,
                               kws_list: List[DictSL],
                               preprocessing_info: dict,
                               profile_id: str) -> List[DictSL]:
    if len(kws_list) == 0:
        return kws_list
    else:
        verbs_indx = get_all_indx_verbs(kws_list)
        if len(verbs_indx) == 0:
            return [add_empty_sbj(kw_dict) for kw_dict in kws_list]
        else:
            # improved_indx, found_kw = get_kw_indx_word(skw_akw_list)
            preproc_vb_sbj = preprocessing_info['verbs_subjects']
            all_found_verbs = np.array(list(map(verb_range, preproc_vb_sbj['verbs'])))
            verbs_list = []
            for kw_dict in kws_list:
                subject_info = []
                verb_info = []
                flags = kw_dict['FLAG'].split('_')
                if len(flags) == 1:
                    if flags[0] in {'result', 'indirect engagement', 'means', 'action', 'state'}:
                        kw_verbs = kw_dict[flags[0]]
                    elif flags[0] == 'subject':
                        kw_verbs = kw_dict['subject']
                    elif '+' in flags[0]:
                        kw_verbs = kw_dict['action']
                    elif flags[0] in {'no-verbs', 'expertise', 'benefactive'}:
                        kw_verbs = []
                    else:  # unexpected flag
                        logger.error(f'get subjects_for_kws_verbs - unexpected flag: {"|".join(flags)}')
                        kw_verbs = []
                elif len(set(flags)) in [2, 3]:
                    if '+' in flags[0]:
                        kw_verbs = kw_dict[flags[0].split('+')[0]]
                    else:
                        kw_verbs = kw_dict[flags[0]]
                else:  # unexpected flag
                    logger.error(f'get subjects_for_kws_verbs - unexpected flags: {"|".join(flags)}')
                    kw_verbs = []
                for verb in kw_verbs:
                    if isinstance(verb[0], int):
                        # if verb[0] in improved_indx or str(vrb[1]) in found_kw:
                        #     sbj_indxs.append('verb-is-keyword')
                        indx = np.where([verb[0] in x for x in all_found_verbs])[0]
                        if len(indx):
                            tmp = {'passed': verb,
                                   'real': preproc_vb_sbj['verbs'][indx[0]]}
                            verb_info.append(tmp)
                            subject_info.append(preproc_vb_sbj['subjects'][indx[0]])
                        else:
                            if flags[0] == 'subject' and verb[1] == '':
                                tmp = {'passed': verb,
                                       'real': 'no-verbs'}
                                verb_info.append(tmp)
                                subject_info.append(kw_dict['improvedKeyword'])
                            else:
                                verb_info.append(verb)
                                err = error_verb_not_found_in_preproc(verb[0], preproc_vb_sbj['verbs'])
                                subject_info.append(err)
                    elif verb[0] == '':
                        verb_info.append(verb)
                        err = f"{NO_VERB_INDX}_{verb[0]}_{verb[1]}"
                        subject_info.append(err)

                kw_dict['subjectTokens'] = subject_info
                kw_dict['realVerbs'] = verb_info
                verbs_list.append(verb_info)
            kws_list = [add_sbj_type_update_sbj_tok(kw_dict, doc, preprocessing_info['ne_np'],
                                                    verbs_list[i], profile_id)
                        for i, kw_dict in enumerate(kws_list)]
    return kws_list


# =======================================================
# STEP 4 - OUTPUT DATA FORMAT
# =======================================================
def decompose_kw_dict_to_data_format_action_flag(kw_dict: DictSL, flag: str,
                                                 answer: List[DictStr],
                                                 improved_keyword_add_info: dict,
                                                 keys_: List[str]) -> List[DictStr]:
    out_flag = get_short_flag(flag)
    for i, v1 in enumerate(kw_dict['action']):
        if len(kw_dict[flag]) != 0:
            for j, v2 in enumerate(kw_dict[flag]):
                dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                         'improvedKeyword': kw_dict['improvedKeyword'],
                         'whereFound': flag, 'verb': decomp_verb_by_part(v1, 'verb'),
                         'verbPrep': decomp_verb_by_part(v1, 'prep'),
                         'additionalObject': decomp_verb_by_part(v1, 'addobj'),
                         f'{out_flag}Link': fix_verb_link(v1[-1]),
                         f'{out_flag}Verb': decomp_verb_by_part(v2, 'verb'),
                         f'{out_flag}VerbPrep': decomp_verb_by_part(v2, 'prep'),
                         f'{out_flag}AdditionalObject': decomp_verb_by_part(v2, 'addobj'),
                         'subjectToken': kw_dict['subjectTokens'][i],
                         'isPassive': str(kw_dict['realVerbs'][i]['is_passive']),
                         'agentInfo': str(kw_dict['realVerbs'][i]['agent']),
                         'subjectType': kw_dict['subjectTypes'][i],
                         'improvedKeywordAddInfo': improved_keyword_add_info}

                dict_ = add_flag_object(dict_, kw_dict, out_flag)
                dict_ = fill_dict(dict_, keys_, '')
                answer.append(dict_)
        else:
            dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                     'improvedKeyword': kw_dict['improvedKeyword'],
                     'whereFound': flag, f'{out_flag}Link': fix_verb_link(v1[-1]),
                     f'{out_flag}Verb': decomp_verb_by_part(v1, 'verb'),
                     f'{out_flag}VerbPrep': decomp_verb_by_part(v1, 'prep'),
                     f'{out_flag}AdditionalObject': decomp_verb_by_part(v1, 'addobj'),
                     f'{out_flag}Object': kw_dict['improvedKeyword'],
                     'improvedKeywordAddInfo': improved_keyword_add_info}
            dict_ = fill_dict(dict_, keys_, '')
            answer.append(dict_)
    return answer


def get_data_format_cols(kw_dict: DictSL) -> List[DictStr]:
    """
    return list of keyword decomposed to sentence decomposition's data format:
    ['foundKeyword', 'improvedKeyword', 'whereFound', 'verb', 'verbPrep', 'additionalObject', 'object',
     'resultLink', 'resultVerb', 'resultVerbPrep', 'resultAdditionalObject', 'resultObject', 'meansLink',
     'meansVerb', 'meansVerbPrep', 'meansAdditionalObject', 'meansObject', 'indirectLink', 'indirectVerb',
     'indirectVerbPrep', 'indirectAdditionalObject', 'indirectObject', 'extractedLink', 'extractedVerb',
     'extractedObject', 'benefactive', 'benefactiveLink', 'role', 'subjectToken', 'subjectType']
    """
    answer = []
    improved_keyword_add_info = {'akw_pos': kw_dict['akw_pos'], 'akw_head_text': kw_dict['akw_head_text']}
    if kw_dict['FLAG'] == 'expertise':
        dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                 'improvedKeyword': kw_dict['improvedKeyword'],
                 'whereFound': 'expertise in',
                 'improvedKeywordAddInfo': improved_keyword_add_info}
        dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
        answer.append(dict_)
    elif kw_dict['FLAG'] == 'no-verbs':
        if kw_dict['special'][0] != '':
            if kw_dict['special'][0] in {'junk', 'subject', 'role'}:
                dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                         'improvedKeyword': kw_dict['improvedKeyword'],
                         'whereFound': kw_dict['special'][0],
                         'improvedKeywordAddInfo': improved_keyword_add_info}
                if kw_dict['special'][0] == 'role':
                    dict_['role'] = kw_dict['special'][2]
                if kw_dict['special'][0] == 'subject':
                    dict_['subjectToken'] = str(kw_dict['improvedKeyword'])
                    dict_['subjectType'] = 'undefined'
                dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                answer.append(dict_)

            elif kw_dict['special'][0] == 'extracted object':
                if len(kw_dict['special'][1]) > 0:
                    for v in kw_dict['special'][1]:
                        dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                                 'improvedKeyword': kw_dict['improvedKeyword'],
                                 'whereFound': kw_dict['special'][0],
                                 'extractedVerb': str(v).strip(),
                                 'extractedObject': kw_dict['foundKeyword'],
                                 'extractedLink': str(kw_dict['special'][2]).strip(),
                                 'improvedKeywordAddInfo': improved_keyword_add_info}
                        dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                        answer.append(dict_)
                else:
                    dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                             'improvedKeyword': kw_dict['improvedKeyword'],
                             'whereFound': kw_dict['special'][0],
                             'extractedVerb': '',
                             'extractedObject': kw_dict['foundKeyword'],
                             'improvedKeywordAddInfo': improved_keyword_add_info}
                    dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                    answer.append(dict_)

            else:  # unexpected fla
                logger.error(f"get subjects_for_kws_verbs - unexpected flag: {kw_dict['FLAG']}")
                return []
        else:
            dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                     'improvedKeyword': kw_dict['improvedKeyword'],
                     'whereFound': 'no-verbs',
                     'improvedKeywordAddInfo': improved_keyword_add_info}
            dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
            answer.append(dict_)
    else:
        # (verb id, verb, prep, last_verb_object, link)
        flags = kw_dict['FLAG'].split('_')
        flags_set = set(flags)
        if len(flags_set) == 1:
            if flags[0] == 'subject':
                for i, verb in enumerate(kw_dict['subject']):
                    dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                             'improvedKeyword': kw_dict['improvedKeyword'],
                             'whereFound': flags[0],
                             'verb': decomp_verb_by_part(verb, 'verb'),
                             'verbPrep': decomp_verb_by_part(verb, 'prep'),
                             'additionalObject': decomp_verb_by_part(verb, 'addobj'),
                             'subjectToken': kw_dict['subjectTokens'][i],  # improvedKeyword
                             'subjectType': kw_dict['subjectTypes'][i],
                             'improvedKeywordAddInfo': improved_keyword_add_info}  # keyword-in-subject'}
                    dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                    answer.append(dict_)
            elif flags[0] in {'action', 'state'}:
                for i, v in enumerate(kw_dict[flags[0]]):
                    if str(v[1]) == '':
                        dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                                 'improvedKeyword': kw_dict['improvedKeyword'],
                                 'whereFound': flags[0],
                                 'object': kw_dict['improvedKeyword'],
                                 'improvedKeywordAddInfo': improved_keyword_add_info}
                        dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                        answer.append(dict_)
                        continue
                    dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                             'improvedKeyword': kw_dict['improvedKeyword'],
                             'whereFound': flags[0],
                             'verb': decomp_verb_by_part(v, 'verb'),
                             'verbPrep': decomp_verb_by_part(v, 'prep'),
                             'additionalObject': decomp_verb_by_part(v, 'addobj'),
                             'object': kw_dict['improvedKeyword'],
                             'subjectToken': kw_dict['subjectTokens'][i],
                             'isPassive': str(kw_dict['realVerbs'][i]['is_passive']),
                             'agentInfo': str(kw_dict['realVerbs'][i]['agent']),
                             'subjectType': kw_dict['subjectTypes'][i],
                             'improvedKeywordAddInfo': improved_keyword_add_info}
                    dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                    answer.append(dict_)
            elif '+' in flags[0]:
                answer = decompose_kw_dict_to_data_format_action_flag(kw_dict, flags[0].split('+')[1],
                                                                      answer, improved_keyword_add_info,
                                                                      DATA_FORMAT_COLS)
            elif flags[0] == 'benefactive':
                dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                         'improvedKeyword': kw_dict['improvedKeyword'],
                         'whereFound': 'benefactive',
                         'benefactiveLink': '?',
                         'benefactive': kw_dict['improvedKeyword'],
                         'improvedKeywordAddInfo': improved_keyword_add_info}
                dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                answer.append(dict_)
            elif flags[0] in {'result', 'means', 'indirect engagement'}:
                flag_0 = get_short_flag(flags[0])
                for i, v2 in enumerate(kw_dict[flags[0]]):
                    dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                             'improvedKeyword': kw_dict['improvedKeyword'],
                             'whereFound': flags[0],
                             f'{flag_0}Link': fix_verb_link(v2[-1]),
                             f'{flag_0}Verb': decomp_verb_by_part(v2, 'verb'),
                             f'{flag_0}AdditionalObject': decomp_verb_by_part(v2, 'addobj'),
                             f'{flag_0}VerbPrep': decomp_verb_by_part(v2, 'prep'),
                             'subjectToken': kw_dict['subjectTokens'][i],
                             'isPassive': str(kw_dict['realVerbs'][i]['is_passive']),
                             'agentInfo': str(kw_dict['realVerbs'][i]['agent']),
                             'subjectType': kw_dict['subjectTypes'][i],
                             'improvedKeywordAddInfo': improved_keyword_add_info}
                    dict_ = add_flag_object(dict_, kw_dict, flag_0)
                    dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                    answer.append(dict_)
            else:  # unexpected flag
                logger.error(f"get subjects_for_kws_verbs - unexpected flag: {kw_dict['FLAG']}")
                return []

        elif len(flags_set) in [2, 3]:
            if 'action' not in flags_set and '+' not in flags[0] and 'state' not in flags_set:
                flag_0 = get_short_flag(flags[0])
                flag_1 = get_short_flag(flags[1])
                for i, v2 in enumerate(kw_dict[flags[0]]):
                    for j, v3 in enumerate(kw_dict[flags[1]]):
                        dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                                 'improvedKeyword': kw_dict['improvedKeyword'],
                                 'whereFound': flags[1],
                                 f'{flag_0}Link': fix_verb_link(v2[-2]),
                                 f'{flag_0}Verb': decomp_verb_by_part(v2, 'verb'),
                                 f'{flag_0}AdditionalObject': decomp_verb_by_part(v2, 'addobj'),
                                 f'{flag_0}VerbPrep': decomp_verb_by_part(v2, 'prep'),
                                 f'{flag_1}Verb': decomp_verb_by_part(v3, 'verb'),
                                 f'{flag_1}VerbPrep': decomp_verb_by_part(v3, 'prep'),
                                 f'{flag_1}AdditionalObject': decomp_verb_by_part(v3, 'addobj'),
                                 'subjectToken': kw_dict['subjectTokens'][i],
                                 'isPassive': str(kw_dict['realVerbs'][i]['is_passive']),
                                 'agentInfo': str(kw_dict['realVerbs'][i]['agent']),
                                 'subjectType': kw_dict['subjectTypes'][i],
                                 'improvedKeywordAddInfo': improved_keyword_add_info}
                        dict_ = add_flag_object(dict_, kw_dict, flag_1)
                        dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                        answer.append(dict_)
            else:
                if len(flags_set) == 2 and '+' not in flags[0]:
                    flag_1 = list(set(flags_set) - {'action', 'state'})[0]
                    flag_11 = get_short_flag(flag_1)
                    for i, v1 in enumerate(kw_dict[flags[0]]):
                        for j, v2 in enumerate(kw_dict[flag_1]):
                            dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                                     'improvedKeyword': kw_dict['improvedKeyword'],
                                     'whereFound': flag_1,
                                     'verb': decomp_verb_by_part(v1, 'verb'),
                                     'verbPrep': decomp_verb_by_part(v1, 'prep'),
                                     'additionalObject': decomp_verb_by_part(v1, 'addobj'),
                                     f'{flag_11}Link': fix_verb_link(v1[-1]),
                                     f'{flag_11}Verb': decomp_verb_by_part(v2, 'verb'),
                                     f'{flag_11}VerbPrep': decomp_verb_by_part(v2, 'prep'),
                                     f'{flag_11}AdditionalObject': decomp_verb_by_part(v2, 'addobj'),
                                     'subjectToken': kw_dict['subjectTokens'][i],
                                     'isPassive': str(kw_dict['realVerbs'][i]['is_passive']),
                                     'agentInfo': str(kw_dict['realVerbs'][i]['agent']),
                                     'subjectType': kw_dict['subjectTypes'][i],
                                     'improvedKeywordAddInfo': improved_keyword_add_info}
                            dict_ = add_flag_object(dict_, kw_dict, flag_11)
                            dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                            answer.append(dict_)
                elif len(flags_set) == 3 or '+' in flags[0]:
                    flag_2 = flags[-1]
                    flag_22 = get_short_flag(flag_2)
                    if '+' in flags[0]:
                        flag_1 = flags[0].split('+')[1]
                        flag_11 = get_short_flag(flag_1)
                    else:
                        flag_1 = list(flags_set - {'action', flag_2})[0]
                        flag_11 = get_short_flag(flag_1)
                    for i, v1 in enumerate(kw_dict['action']):
                        for j, v2 in enumerate(kw_dict[flag_1]):
                            for k, v3 in enumerate(kw_dict[flag_2]):
                                dict_ = {'foundKeyword': kw_dict['foundKeyword'],
                                         'improvedKeyword': kw_dict['improvedKeyword'],
                                         'whereFound': flag_2,
                                         'verb': decomp_verb_by_part(v1, 'verb'),
                                         'verbPrep': decomp_verb_by_part(v1, 'prep'),
                                         'additionalObject': decomp_verb_by_part(v1, 'addobj'),
                                         f'{flag_11}Link': fix_verb_link(v1[-1]),
                                         f'{flag_11}Verb': decomp_verb_by_part(v2, 'verb'),
                                         f'{flag_11}VerbPrep': decomp_verb_by_part(v2, 'prep'),
                                         f'{flag_11}AdditionalObject': decomp_verb_by_part(v2, 'addobj'),
                                         f'{flag_22}Link': decomp_verb_by_part(v2, 'link'),
                                         f'{flag_22}Verb': decomp_verb_by_part(v3, 'verb'),
                                         f'{flag_22}VerbPrep': decomp_verb_by_part(v3, 'prep'),
                                         f'{flag_22}AdditionalObject': decomp_verb_by_part(v3, 'addobj'),
                                         'subjectToken': kw_dict['subjectTokens'][i],
                                         'isPassive': str(kw_dict['realVerbs'][i]['is_passive']),
                                         'agentInfo': str(kw_dict['realVerbs'][i]['agent']),
                                         'subjectType': kw_dict['subjectTypes'][i],
                                         'improvedKeywordAddInfo': improved_keyword_add_info}
                                dict_ = add_flag_object(dict_, kw_dict, flag_22)
                                dict_ = fill_dict(dict_, DATA_FORMAT_COLS, '')
                                answer.append(dict_)
                else:  # unexpected flag
                    logger.error(f"get subjects_for_kws_verbs - unexpected flags: {kw_dict['FLAG']}")
                    return []
        else:
            logger.error(f"get subjects_for_kws_verbs- - unexpected flags len: {kw_dict['FLAG']}")
            return []
    return answer


# =======================================================
# MAIN FUNCTION + ADDITIONAL
# =======================================================
def add_static_columns(d: Dict[str, str | int], profile: str, person_name: str, company_name: str, sentence: str,
                       section: str, order: int, ref_id: str, ref_type: str, sentence_id: str) -> SentenceDecompositionDoc:
    d['profile'] = profile
    d['personName'] = person_name
    d['companyName'] = company_name
    d['sentence'] = sentence
    d['section'] = section
    d['order'] = order
    d['refId'] = ref_id
    d['refType'] = ref_type
    d['sentenceId'] = sentence_id
    return SentenceDecompositionDoc(**d)


def add_expertise(sentence_doc: SpacyDoc, skw_akw: dict) -> dict:
    akw_span = sentence_doc[skw_akw['akw_indices'][0]:skw_akw['akw_indices'][2] + 1]
    skw_text = skw_akw['skw_text']
    skw_akw['expertise'] = ExpCheck.checkExpertise(sentence_doc, akw_span, skw_text)
    return skw_akw


def analyze_sentence(sentence_doc: SpacyDoc,
                     skw_akw_list: List[dict],
                     sentence: str,
                     profile: str,
                     profile_type: str,
                     profile_id: str,
                     sentence_id: str,
                     person_name: str,
                     company_name: str,
                     section: str,
                     order: int,
                     preprocessing_info: dict) -> List[SentenceDecompositionDoc]:
    # 1 - expertise
    skw_akw_list = [add_expertise(sentence_doc, skw_akw) for skw_akw in skw_akw_list]
    # 2 - verbs
    kws_dict_list = get_verbs_for_kws(sentence_doc, skw_akw_list)
    # 3 - subjects
    kws_dict_list = get_subjects_for_kws_verbs(sentence_doc, kws_dict_list, preprocessing_info, profile_id)
    # 4 - output data format
    out_list = [add_static_columns(d, profile, person_name, company_name, sentence, section, order,
                                   profile_id, profile_type, sentence_id)
                for dict_ in kws_dict_list if dict_ for d in get_data_format_cols(dict_)]

    logger.debug(f' 4 - output data format: {sentence}')
    return out_list


def analyze_sentence_dict(sentence_dict: Dict[str, SpacyDoc | str | int | list | dict]) -> List[SentenceDecompositionDoc]:
    return analyze_sentence(sentence_dict['sentenceDoc'],
                            sentence_dict['skwAkw'],
                            sentence_dict['text'],
                            sentence_dict['profileUrl'],
                            sentence_dict['refType'],
                            sentence_dict['refId'],
                            sentence_dict['sentenceId'],
                            sentence_dict['personName'],
                            sentence_dict['companyName'],
                            sentence_dict['section'],
                            sentence_dict['order'],
                            sentence_dict['preprocessingInfo'])
