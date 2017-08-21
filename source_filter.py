# coding=utf-8

import json
import pickle
import re
import tokenize
import ast
# import parser
import cStringIO
import heapq
import subprocess
import token
import copy

lines_limit = 0.97
tokens_limit = 0.97
tokens_freq_limit = 0.999
rename_correction_rate = 0.98

input_value = '1 2\n'
output_value = '3\n'

temp_file = '_temp.py'

input_file = '1000-6-2017-07-13-12:55:21.json'


def remove_redundent_lines(data):
    for d in data:
        d['source'] = re.sub('(\\n)+', '\n', d['source'])  # TODO 토큰 기준으로 하면 더 정확할듯
        d['source'] = re.sub('(\n)$', '', d['source']) # remove last linebreak
        # d['source'] = re.sub('[]', '', d['source'])
        d['line_length'] = d['source'].count('\n') + 1


def get_length_stat(data):
    length_stat = {}
    for d in data:
        if d['line_length'] in length_stat:
            length_stat[d['line_length']] += 1
        else:
            length_stat[d['line_length']] = 1
    return length_stat


def get_token_stat(data):
    token_stat = {}
    for d in data:
        if len(d['token']) in token_stat:
            token_stat[len(d['token'])] += 1
        else:
            token_stat[len(d['token'])] = 1
    return token_stat


def get_token_freq_stat(data):
    total_token = 0
    token_freq_stat = {}
    for di, d in enumerate(data):
        d['index'] = di
        for token in d['token']:
            total_token += 1
            key = (token[0], token[1])
            if key in token_freq_stat:
                token_freq_stat[key]['freq'] += 1
                token_freq_stat[key]['data'].append(di)
            else:
                token_freq_stat[key] = {
                    'freq': 0,
                    'data': []
                }
                token_freq_stat[key]['freq'] = 1
                token_freq_stat[key]['data'].append(di)

    removed_token_freq_stat = copy.deepcopy(token_freq_stat)
    total_removed = 0
    removed_data = []
    h = []

    for key, value in token_freq_stat.iteritems():
        heapq.heappush(h, (value['freq'], (key, value)))

    while total_removed < int(round((1 - tokens_freq_limit) * total_token)):
        stat = heapq.heappop(h)
        total_removed += stat[0]
        removed_data.extend(stat[1][1]['data'])
        del removed_token_freq_stat[stat[1][0]]

    return token_freq_stat, total_removed, removed_data, removed_token_freq_stat


def filter_lines_too_long(data):
    sorted_data = sorted(data, key=lambda d: d['line_length'])
    return sorted_data[:int(round(len(data) * lines_limit))]


def filter_tokens_too_much(data):
    sorted_data = sorted(data, key=lambda d: len(d['token']))
    return sorted_data[:int(round(len(data) * tokens_limit))]


def set_token(data):
    filterd_data = []
    for idx, d in enumerate(data):
        # d['tree'] = ast.parse(d['source'].encode('ascii'), filename='parsed.py', mode='exec')
        # d['st'] = parser.suite(d['source'])
        # d['st_list'] = parser.st2tuple(d['st'])
        src_readline = cStringIO.StringIO(d['source']).readline
        try:
            d['token'] = list(tokenize.generate_tokens(src_readline))
            filterd_data.append(d)
        except Exception as e:
            e
            # fail to tokeninze
            # delete_list.append(idx)
    return filterd_data


def filter_danger(data, dangerous_pattern='(import\s+os)|(from\s+os)|(shutil)'):
    p = re.compile(dangerous_pattern)
    return filter(lambda d: not p.match(d['source']), data)


# 가공된 데이터 정확성 테스트
def test_source(data, source):
    with open(temp_file, 'w') as f:
        output_list = []
        for d in data:
            if not 'test_result' in d or not d['test_result']:
                # if d['test_result']:
                f.seek(0)
                f.write(d[source])
                f.truncate()

                p = subprocess.Popen(['python', temp_file], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                out = p.communicate(input=input_value)[0]
                result = out == output_value
                d['test_result'] = result
                output_list.append(result)
                # else:
                #     output_list.append(d['test_result'])
            else:
                output_list.append(d['test_result'])

    return output_list


def bool_list2percent(bool_list):
    cnt = 0
    for bool in bool_list:
        if bool:
            cnt += 1

    return cnt / (len(bool_list) * 1.0)


def rename_vairables(data, token_freq_stat):
    h = []

    for key, value in token_freq_stat.iteritems():
        heapq.heappush(h, (-value['freq'], (key, value)))

    i = 0
    while h:
        stat = heapq.heappop(h)
        key = stat[1][0]
        value = stat[1][1]
        if key[0] == token.NAME:
            value['replace_name'] = 'x' + str(i)
            i += 1

    for d in data:
        d['renamed_token'] = []
        for t in d['token']:
            if t[0] == token.NAME:
                stat = token_freq_stat[(t[0], t[1])]
                d['renamed_token'].append((t[0], stat['replace_name'], t[2], t[3], t[4]))
            else:
                d['renamed_token'].append(t)

        d['renamed_source'] = tokenize.untokenize(d['renamed_token'])


    # if bool_list2percent(test_source(data, 'source')) != 1.0:
    #     raise Exception('invalid accurate source.')

    token_freq_stat_list = filter(lambda x: x[0][0] == token.NAME, token_freq_stat.items())
    token_freq_stat_list.sort(key=lambda x: x[1]['freq'])

    forbidden_list = []
    correction_rate = 0.0
    while correction_rate < 1.0:
        most_freq_token = token_freq_stat_list.pop()
        for di in most_freq_token[1]['data']:
            dl = filter(lambda x: x['index'] == di, data)
            if len(dl) == 1:
                d = dl[0]
                for ti, t in enumerate(d['renamed_token']):
                    if t[0] == token.NAME and t[1] == most_freq_token[1]['replace_name']:
                        d['renamed_token'][ti] = (t[0], most_freq_token[0][1], t[2], t[3], t[4])

                d['renamed_source'] = tokenize.untokenize(d['renamed_token'])

        correction_rate = bool_list2percent(test_source(data, 'renamed_source'))
        forbidden_list.append(most_freq_token)
        print correction_rate * 100, most_freq_token

    for d in data:
        d['renamed_source_temp'] = d['renamed_source']
        d['renamed_token_temp'] = copy.deepcopy(d['renamed_token'])
        d['test_result'] = False

    # with open(input_file + '.temp', 'w') as data_file:
    #     data_file.write(json.dumps([data, forbidden_list, correction_rate]))
    #
    # with open(input_file + '.temp', 'r') as data_file:
    #     data, forbidden_list, correction_rate = json.load(data_file)

    forbidden_deleted = []
    for forbidden in forbidden_list:
        for di in forbidden[1]['data']:
            dl = filter(lambda x: x['index'] == di, data)
            if len(dl) == 1:
                d = dl[0]
                for ti, t in enumerate(d['renamed_token_temp']):
                    if t[0] == token.NAME and t[1] == forbidden[0][1]:
                        d['renamed_token_temp'][ti] = (t[0], forbidden[1]['replace_name'], t[2], t[3], t[4])

                d['renamed_source_temp'] = tokenize.untokenize(d['renamed_token_temp'])

        correction_rate_temp = bool_list2percent(test_source(data, 'renamed_source_temp'))
        print correction_rate_temp, forbidden
        if correction_rate_temp < correction_rate:
            forbidden_deleted.append(forbidden)
            for d in data:
                d['renamed_source_temp'] = d['renamed_source']
                d['renamed_token_temp'] = copy.deepcopy(d['renamed_token'])
                d['test_result'] = False
        else:
            for d in data:
                d['renamed_source'] = d['renamed_source_temp']
                d['renamed_token'] = copy.deepcopy(d['renamed_token_temp'])
                d['test_result'] = False

    #         forbidden_list 에서 forbidden_deleted 빼기
    return forbidden_deleted


if __name__ == '__main__':
    with open(input_file) as data_file:
        raw_data = json.load(data_file)

    data = filter_danger(raw_data)

    data = filter(lambda d: d['accurate'], data)  # get only accurate
    remove_redundent_lines(data)  # too much newlines

    length_stat = get_length_stat(data)
    line_limit_data = filter_lines_too_long(data)
    line_limit_stat = get_length_stat(line_limit_data)

    set_token(line_limit_data)
    token_stat = get_token_stat(line_limit_data)
    token_limit_data = filter_tokens_too_much(line_limit_data)
    token_limit_stat = get_token_stat(token_limit_data)

    token_freq_stat, total_removed, removed_data, removed_token_freq_stat = get_token_freq_stat(token_limit_data)
    freq_limit_data = []
    for i, d in enumerate(token_limit_data):
        if not i in removed_data:
            freq_limit_data.append(d)

    re_data = freq_limit_data


    forbidden_list = rename_vairables(re_data, removed_token_freq_stat)

    for d in re_data:
        del d['renamed_source_temp']
        del d['renamed_token_temp']
        del d['test_result']

    with open(input_file + '.pkl', 'w') as data_file:
        pickle.dump([re_data, removed_token_freq_stat, forbidden_list], data_file)