import ast
from _ast import AST
import msgpack as pickle
import datetime

import requests
from bs4 import BeautifulSoup

root = 'https://www.acmicpc.net'
lauguage_dict = {
    'python': '6',
    'python3': '28'
}
# python 6, python3 28
cookies = dict(bojautologin='24ebc463a52ca1da648b4caa9211deae784bedb3', OnlineJudge='jst46dkvpmvn9lp7lc5alce100')

data_dir = './code-data/'

limit = 10


def get_data_from_web(problem_id=1000, language='python', cookies=cookies):
    language_id = lauguage_dict[language]
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

    with open(data_dir + str(problem_id) + '-' + language_id + '-' + str(
            datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')) + '.' + pickle.__name__, 'w') as output:
        l = 0
        i = 0
        while True:
            tr_list = soup.find(id='status-table').find('tbody').find_all('tr')
            next_page = soup.find(id='next_page')

            j = 0
            for tr in tr_list:
                temp_solution = {}
                td_list = tr.find_all('td')
                temp_solution['id'] = td_list[0].text
                temp_solution['user'] = td_list[1].find('span').text
                temp_solution['accurate'] = True if td_list[3].find(class_='result-ac') else False
                temp_solution['source_link'] = '/source/' + temp_solution['id']

                # if temp_solution['source_link'] == '':
                #     continue
                sol_r = requests.get('https://www.acmicpc.net' + temp_solution['source_link'], cookies=cookies)
                if sol_r.status_code != 200:
                    continue
                sol_soup = BeautifulSoup(sol_r.content, 'html.parser')
                source = sol_soup.find(id='source').contents[0]
                # tree = ast.parse(source, filename='parsed.py', mode='exec')
                # code = compile(tree, filename='parsed.py', mode='exec')
                temp_solution['source'] = source
                # temp_solution['tree'] = tree
                # temp_solution['code'] = code
                j += 1
                print 'solution: ' + str(j)

                solutions.append(temp_solution)

                # output.write(json.dumps(solutions))
                # output.truncate()

                # l += 1
                if limit <= l:
                    break

            i += 1
            print 'page: ' + str(i)

            if not next_page:
                break
            if limit <= l:
                break

            response = requests.get(root + next_page.attrs['href'], cookies=cookies)
            soup = BeautifulSoup(response.content, 'html.parser')

            # success. make it as a file
            # for solution in solutions:
            #     if solution['source_link'] == '':
            #         continue
            #     sol_r = requests.get('https://www.acmicpc.net' + solution['source_link'], cookies=cookies)
            #     soup = BeautifulSoup(sol_r.content, 'html.parser')
            #     source = soup.find(id='source').contents[0]
            #     tree = ast.parse(source, filename='parsed.py', mode='exec')
            #     code = compile(tree, filename='parsed.py', mode='exec')
            #     solution['source'] = source
            #     solution['tree'] = tree
            #     solution['code'] = code
            #     print 'solution: ' + str(i)
            # exec(code)

        pickle.dump(solutions, output)
        # output.write(pickle.dumps(solutions))
        # output.truncate()


def _format(node):
    if isinstance(node, AST):
        fields = [('_PyType', _format(node.__class__.__name__))]
        fields += [(a, _format(b)) for a, b in ast.iter_fields(node)]

        return '{ %s }' % ', '.join(('"%s": %s' % field for field in fields))

    if isinstance(node, list):
        return '[ %s ]' % ', '.join([_format(x) for x in node])

    return pickle.dumps(node)


# tree_json = _format(solutions[0]['tree'])
# output = open('output-' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.json', 'w')
# output.write(tree_json)
if __name__ == '__main__':
    get_data_from_web()
