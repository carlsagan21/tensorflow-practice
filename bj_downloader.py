# coding=utf-8
import ast
from _ast import AST
import msgpack as pickle
import datetime

import requests
from bs4 import BeautifulSoup

root = 'https://www.acmicpc.net'
language_dict = {
    'c': '0',
    'c++': '1',
    'pascal': '2',
    'java': '3',
    'python': '6',
    'php': '7',
    'perl': '8',
    'go': '12',
    'fortran': '13',
    'scheme': '14',
    'lua': '16',
    'nodejs': '17',
    'ada': '19',
    'awk': '21',
    'ocaml': '22',
    'brainfuck': '23',
    'whitespace': '24',
    'tcl': '26',
    'assembly32': '27',
    'python3': '28',
    'd': '29',
    'pypy': '32',
    'clojure': '33',
    'rhino': '34',
    'cobol': '35',
    'f#': '37',
    'spidermonkey': '38',
    'pike': '41',
    'sed': '43',
    'rust': '44',
    'boo': '46',
    'intercal': '47',
    'bc': '48',
    'c++11': '49',
    'nemerle': '53',
    'cobra': '54',
    'nimrod': '55',
    'cclang': '59',
    'c++clang': '60',
    'c#4.0': '62',
    'vb.net4.0': '63',
    'c++11clang': '66',
    'c++14clang': '67',
    'ruby2.2': '68',
    'kotlin': '69',
    'algol68': '70',
    'befunge': '71',
    'r': '72',
    'pypy3': '73',
    'swift': '74',
    'c11': '75',
    'c11clang': '77',
    'freebasic': '78',
    'golfscript': '79',
    'gosu': '80',
    'haxe': '81',
    'lolcode': '82',
    '아희': '83',
}

cookies = dict(bojautologin='24ebc463a52ca1da648b4caa9211deae784bedb3', OnlineJudge='f74lhe3586uc33h8gh4icl4t57')

data_dir = './code-data/'

limit = 10


def get_data_from_web(language, problem_id=1000):
    language_ids = [language_dict[l] for l in language]

    for language_id in language_ids:
        response = requests.get(
            root + '/status/?problem_id=' + str(problem_id) + '&language_id=' + language_id + '&from_problem=1',
            cookies=cookies
        )
        # html_file = open('result.html', 'w')
        # html_file.write(r.content)
        # html_file.close()

        soup = BeautifulSoup(response.content, 'html.parser')

        # tr_list = soup.find(id='status-table').find('tbody').find_all('tr')
        # next_page = soup.find(id='next_page')

        solutions = []

        l = 0
        i = 0
        try:
            with open(data_dir + str(problem_id) + '-' + language_id + '.' + pickle.__name__, 'r') as save_file:
                saved_data = pickle.load(save_file)
                print 'read data success ' + str(len(saved_data))
        except Exception as e:
            print e
            saved_data = []

        while True:
            tr_list = soup.find(id='status-table').find('tbody').find_all('tr')
            next_page = soup.find(id='next_page')
            if not next_page:
                break

            j = 0
            for tr in tr_list:
                # try:
                j += 1

                temp_solution = {}
                td_list = tr.find_all('td')
                temp_solution['id'] = int(td_list[0].text)
                if any(source['id'] == temp_solution['id'] for source in saved_data):
                # if len(saved_data) != 0 and int(saved_data[-1]['id']) <= int(temp_solution['id']) <= int(saved_data[0]['id']):
                    print 'saved continue ' + str(j)
                    continue

                temp_solution['user'] = td_list[1].find('span').text
                temp_solution['accurate'] = True if td_list[3].find(class_='result-ac') else False
                temp_solution['source_link'] = '/source/' + str(temp_solution['id'])
                a_tag = td_list[6].find('a')
                if not a_tag:
                    # print 'unaccessable source ' + str(j)
                    continue

                # if temp_solution['source_link'] == '':
                #     continue
                sol_r = requests.get('https://www.acmicpc.net' + temp_solution['source_link'], cookies=cookies)
                if sol_r.status_code != 200:
                    print 'non 200 continue ' + str(j)
                    continue
                sol_soup = BeautifulSoup(sol_r.content, 'html.parser')
                source = sol_soup.find(id='source').contents[0]
                # tree = ast.parse(source, filename='parsed.py', mode='exec')
                # code = compile(tree, filename='parsed.py', mode='exec')
                temp_solution['source'] = source
                # temp_solution['tree'] = tree
                # temp_solution['code'] = code

                print 'solution ' + str(j)

                if len(saved_data) != 0 and temp_solution['id'] > int(saved_data[0]['id']):
                    solutions.insert(0, temp_solution)
                else:
                    solutions.append(temp_solution)

                # output.write(json.dumps(solutions))
                # output.truncate()
                # except:
                #     print 'exception in %d' % j
                # l += 1
                # if limit <= l:
                #     break

            i += 1
            print 'page ' + str(i)

            if not next_page:
                break

            if not len(solutions) == 0:
                with open(data_dir + str(problem_id) + '-' + language_id + '.' + pickle.__name__, 'w') as save_file:
                    updated_data = saved_data + solutions
                    updated_data.sort(reverse=True, key=id)
                    pickle.dump(updated_data, save_file)
                    saved_data = updated_data
                    solutions = []

            response = requests.get(root + next_page.attrs['href'], cookies=cookies)
            soup = BeautifulSoup(response.content, 'html.parser')


        # pickle.dump(solutions, save_file)


# def _format(node):
#     if isinstance(node, AST):
#         fields = [('_PyType', _format(node.__class__.__name__))]
#         fields += [(a, _format(b)) for a, b in ast.iter_fields(node)]
#
#         return '{ %s }' % ', '.join(('"%s": %s' % field for field in fields))
#
#     if isinstance(node, list):
#         return '[ %s ]' % ', '.join([_format(x) for x in node])
#
#     return pickle.dumps(node)


if __name__ == '__main__':
    get_data_from_web(['c'])
