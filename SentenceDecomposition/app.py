import traceback
from collections import defaultdict

from flask import request

from core.SentenceDecomposition_udf import analyze_sentence_dict
from oneforce_common import base_microservice as bm, validateResponseAndReturn
from oneforce_elasticsearch import (sentence_es_actions, company_profile_es_actions, person_profile_es_actions,
                                    statistic_es_actions)
from oneforce_logger import OneForceLogger
from oneforce_rest_api_client import sentence_analyzer_client
from oneforce_spacy_utils import spacy_utils as su
from oneforce_statistics import statistic
from oneforce_swagger_docs import sentence_decomposition_response_schema, SentenceDecompositionDocSchema, BaseList

appData = bm.create_by_name("sentenceDecomposition", "Sentence Decomposition",
                            "Service that determines the type of the keyword "
                            "and extracts verbs, objects and other information for the keyword")

app = appData["app"]
port = appData["port"]
rest_api_prefix = appData["rest_api_prefix"]
rest_api_prefix_v2 = appData["rest_api_prefix_v2"]
logger: OneForceLogger = appData["logger"]

key = 'list'


def return_es_actions(ref_type: str):
    if ref_type == 'company':
        return company_profile_es_actions
    elif ref_type == 'person':
        return person_profile_es_actions


def convert_akw_dict(akw: dict, skw_text: str) -> dict:
    return {'akw_text': akw['akw_text'],
            'akw_indices': akw['akw_indices'],
            'skw_text': skw_text,
            'akw_pos': akw['akw_pos'],
            'akw_head_text': akw['akw_head_text']}


def convert_skw_akw_list(skw_akw_list: list):
    out_dicts = [convert_akw_dict(akw, dict_['skw_text']) for dict_ in skw_akw_list if dict_ for akw in
                 dict_['akw_list']]
    return out_dicts


def get_dict_for_sd(source: dict, _id: str, skw_akw: list, add_info: bool = False) -> dict:
    dict_ = {'skwAkw': convert_skw_akw_list(skw_akw),
             'sentenceDoc': su.decode_bs64(source['sentenceDoc']),  # TODO: add condition for existence
             'section': source['section'],
             'refType': source['refType'],
             'refId': source['refId'],
             'sentenceId': _id,
             "text": source['text'],
             'preprocessingInfo': source['preprocessingInfo'],
             'order': source['order']}
    if add_info:
        dict_['profileUrl'] = source['profileUrl']
        dict_['companyName'] = source['companyName']
        dict_['personName'] = source['personName']
    return dict_


def convert_sent_dict(es_doc: dict, skw_akw: dict, profile_sentence_map: dict) -> dict:
    source = es_doc["_source"]
    dict_ = get_dict_for_sd(source, es_doc['_id'], skw_akw['skwAkw'])
    profile_sentence_map[source['refId']].add(es_doc['_id'])
    return dict_


def update_sent_dict(sent_dict: dict, ref_type: str, profile: dict) -> dict:
    if ref_type == 'person':
        sent_dict['companyName'] = '|||'.join([profile['_source']['company'], profile['_source']['company2']])
        sent_dict['personName'] = profile['_source']['fullName']
        sent_dict['profileUrl'] = profile['_source']['linkedinProfile']
    elif ref_type == 'company':
        sent_dict['companyName'] = profile['_source']['CompanyName']
        sent_dict['personName'] = ''
        if 'linkedInCompanyUrl' in profile['_source']:
            sent_dict['profileUrl'] = profile['_source']['linkedInCompanyUrl']
        else:
            sent_dict['profileUrl'] = "none"
    return sent_dict


def map_profiles_update_sentences(profile_dict: dict, ref_type: str, profile_sentence_map: dict,
                                  sentences: dict) -> bool:
    """
    The function maps profiles to the sentences by refId and update each sentence dict by profile's additional info.
    """
    cnt = 0
    for sent_id in profile_sentence_map[profile_dict['_id']]:
        sentences[sent_id] = update_sent_dict(sentences[sent_id], ref_type, profile_dict)
        cnt += 1
    return cnt == len(profile_sentence_map[profile_dict['_id']])  # FIX


def decompose(ref_type: str, parsed_akw_doc: dict):
    es_actions = return_es_actions(ref_type)
    if not isinstance(parsed_akw_doc, dict):
        return bm.error("Wrong request. parsed_akw_doc are not a dict", 400)
    if key not in parsed_akw_doc:
        return bm.error("Wrong request. parsed_akw_doc should contain key 'list' ", 400)
    parsed_akw_doc_list = parsed_akw_doc[key]
    if not isinstance(parsed_akw_doc_list, list):
        return bm.error("Wrong request. 'list' key must contain a list of data for parsed_akw_doc", 400)
    parsed_akw_doc_map = {elem['refId']: elem for elem in parsed_akw_doc_list}
    profile_sentence_map = defaultdict(set)
    docs_generator = sentence_es_actions.get_by_ids(list(parsed_akw_doc_map.keys()), pagination_by=100)
    if isinstance(docs_generator, dict):
        return bm.error(f"Sentences not found", 404)

    sentences = {doc['_id']: convert_sent_dict(doc, parsed_akw_doc_map[doc['_id']], profile_sentence_map) for page in
                 docs_generator if page for doc in page}
    profiles = es_actions.get_by_ids(list(profile_sentence_map.keys()), pagination_by=100)
    if isinstance(docs_generator, dict):
        return bm.error(f"Profiles for sentences not found", 404)

    try:
        tmp = {prof_doc['_id']: map_profiles_update_sentences(prof_doc, ref_type, profile_sentence_map,
                                                              sentences) for page in profiles if page for
               prof_doc in page}
    except Exception as e:
        logger.error("The ERROR is HERE!!")
        logger.error(traceback.format_exc())
    out_list = [sent_sd for sent_id, sent_dict in sentences.items() for sent_sd in
                analyze_sentence_dict(sent_dict)]
    return BaseList(out_list, SentenceDecompositionDocSchema).json()


@app.route(rest_api_prefix + "/<ref_type>/decomp-json", methods=['POST'])
@statistic.oneforce_stat
def decomp_json(ref_type: str):
    return validateResponseAndReturn(sentence_decomposition_response_schema, decompose(ref_type, request.json))


@app.route(rest_api_prefix + "/<ref_type>", methods=['POST'])
@statistic.oneforce_stat
def run_sd(ref_type: str):
    args = request.args
    search_left = request.args.get('search-left', default='True') == 'True'
    resp = sentence_analyzer_client.get_by_user_keyword(ref_type, args.get("q"), args.get("sections"), search_left)
    res = decompose(ref_type, resp)
    return res  # validateResponseAndReturn(sentence_decomposition_response_schema, res)


@app.route(rest_api_prefix_v2 + "/<ref_type>", methods=['POST'])
@statistic.oneforce_stat
def run_sd_v2(ref_type: str):
    args = request.args
    resp = sentence_analyzer_client.search_by_user_keywords_v2(ref_type, request.json,
                                                               args.get("search-left", default='True') == 'True',
                                                               args.get('preproc', type=str, default='all'))
    out_list = [sent_sd for sent_dict in resp['list'] for sent_sd in
                analyze_sentence_dict(get_dict_for_sd(sent_dict, '', sent_dict['skwAkw'], True))]
    return BaseList(out_list, SentenceDecompositionDocSchema).json()


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=port)


def create_app():
    return app
