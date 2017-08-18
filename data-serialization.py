import json
# import simplejson               # pip install simplejson
import pickle
import cPickle
import marshal
# import yaml                     # pip install pyyaml
# import ujson                    # pip install ujson
# from cStringIO import StringIO
import bson                     # pip install bson
import msgpack

import random
import tensorflow as tf
import time

pickles = [cPickle, pickle, json, marshal, msgpack]
serializers = [bson]
for srlz in serializers:
    def dump(obj, fp):
        srlzed = srlz.dumps(obj)
        fp.write(srlzed)
    srlz.dump = dump
# savers = pickles + serializers
savers = pickles

def compare(size, ran_range):
    # result = []
    print("Write Saver Test.")
    message = []
    for _ in range(size):
        message.append(random.randrange(ran_range))
    for saver in savers:
        with open("./srl_data/" + saver.dump.__module__ + '.' + saver.dump.__name__, mode="w") as f:
            start_time = time.time()
            saver.dump(message, f)
            taken_time = time.time() - start_time
            # result.append(taken_time)
            print(saver.dump.__module__ + '.' + saver.dump.__name__ + ":  %s seconds" % taken_time)

    print("Read Saver Test.")
    for saver in savers:
        with open("./srl_data/" + saver.dump.__module__ + '.' + saver.dump.__name__, mode="r") as f:
            start_time = time.time()
            _ = saver.load(f)
            taken_time = time.time() - start_time
            # result.append(taken_time)
            print(saver.dump.__module__ + '.' + saver.dump.__name__ + ":  %s seconds" % taken_time)


def main():
    # Functions                                 # Average usr + sys time
    # =========                                 ========================
    # serialization_func = no_op                # 0.19 seconds (to measure basic overhead)
    # serialization_func = ujson.dumps          # 0.26 seconds
    # serialization_func = home_brew  # fastest was 0.37 seconds
    # serialization_func = marshal.dump         # 0.58 seconds            <-- Needs a file
    serialization_func = json.dumps  # 0.78 seconds
    # serialization_func = cPickle.dump         # 1.07 seconds            <-- Needs a file
    # serialization_func = simplejson.dumps     # 1.11 seconds
    # serialization_func = pickle.dump          # 4.22 seconds            <-- Needs a file
    # serialization_func = bson.dumps           # 5.25 seconds
    # serialization_func = yaml.dump            # 62.80 SECONDS!

    print "using {fn} function from {mod}".format(fn=serialization_func.__name__, mod=serialization_func.__module__)

    size = 1000
    random_range = 1000

    with open("./srl_data/" + serialization_func.__module__ + '.' + serialization_func.__name__ + ".size" + str(
            size) + ".rr" + str(random_range), mode="w") as f:
        message = []
        for _ in range(size):
            message.append(random.randrange(random_range))

            # serialization_func(message, f)        # <-- Some serialization_funcs need a file
            # serialization_func(message)  # <-- Some don't

        start_time = time.time()
        serialization_func(message, f)
        print("--- %s seconds ---" % (time.time() - start_time))


def no_op(message):
    # do no work, used as a control
    return message


def home_brew(message):
    s = ""

    # First attempt
    # 0.55 seconds
    # for k, v in message.items():
    #     s += "{}={}|".format(k, v)

    # Old-school string interpolation
    # 0.41 seconds
    # for k, v in message.items():
    #     s += "%s=%s|" % (k, v)

    # Skip call to items()
    # 0.37 seconds
    for k in message:
        s += "%s=%s|" % (k, message[k])

    # Iterate over array of keys
    # 0.39 seconds
    # keys = message.keys()
    # for k in keys:
    #     s += "%s=%s|" % (k, message[k])

    # Try using join instead of +=
    # 0.45 seconds
    # s = []
    # for k in message:
    #     s.append("%s=%s" % (k, message[k]))
    # s = "|".join(s)

    # Using cString
    # 0.63 seconds
    # s = StringIO()
    # for k in message:
    #     s.write("%s=%s|" % (k, message[k]))

    # Try removing explicit loops, via Guido: https://www.python.org/doc/essays/list2str/
    # 0.41 seconds
    # "|".join(["%s=%s" % (k, message[k]) for k in message.keys()])

    # Can we use reduce???
    # def r(x, y):
    #     return "%s=%s|%s=%s" % (x, message[x], y, message[y])
    # s = reduce(r, message.keys())
    # ... No, we can't on dicts

    # Try using + instead of +=
    # 0.39 seconds
    # keys = message.keys()
    # for k in keys:
    #     s = s +  "%s=%s|" % (k, message[k])

    # Suggested by /u/Lucretiel on reddit
    # 0.64 seconds
    # concat = '|'.join
    # k_v = '{}={}'.format
    # concat(k_v(k, message[k]) for k in message)

    # ...and without caching the attributes, for comparison to previous attempt
    # 0.51 seconds
    # "|".join(["{}={}".format(x, message[x]) for x in message])

    # Using generator expressions
    # 0.66 seconds
    # s = str.join('|', ('{}={}'.format(k, message[k]) for k in message))

    return s


if __name__ == '__main__':
    compare(100000, 100000)
