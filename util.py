from __future__ import print_function


def code(file, *args):
    if file:
        for a in args:
            print(a, file=file)

def constant_int(file, *args):
    for i in range(0,len(args),2):
        code(file, "static int const {0} = {1};".format(args[i],args[i+1]))


def listToCArrayString(l):
    def stringify(l):
        return '{' + str(l).replace('[', '').replace(']', '') + '}'
    return stringify(toList(l))


def toList(l):
    if type(l) == list:
        return l
    else:
        return l.tolist()

def tupleToDimension(tp):
    r = ''
    for t in tp:
        r += '{0}*'.format(t)
    return  '[' + r[:-1] + ']'

def tupleToDimensionBracket(tp):
    r = ''
    for t in tp:
        r += '[{0}]'.format(t)
    return r

def flatList(l, fun=float):
    s =  str(l).replace('[', '').replace(']', '')
    lst = s.split(',')
    return [fun(x) for x in lst]


def float_array(file=None,list=[], name='arr', shape=(), decimal=2, static=False, declare_only=False):



    if not declare_only:
        s = "float {0}{1} = {2};".format(name,tupleToDimensionBracket(shape),
                                 listToCArrayString([float("%.{0}f".format(decimal)%l)
                                                 for l in flatList(toList(list), float)]))
    else:
        s = "float {0}{1};".format(name, tupleToDimensionBracket(shape))

    if static:
        s = 'static ' + s
    if file == None:
        return s
    else:
        code(file, s)
        return s

def float_array_in_struct(list=[], shape=(), decimal=3):
    return "(float {0}){1}".format(tupleToDimension(shape),
                                   listToCArrayString([float("%.{0}f".format(decimal)%l)
                                                        for l in flatList(toList(list), float)]))


def extern(s, file=None):
    s =  "extern " + s
    if file:
        code(file, s)
    else:
        return s
