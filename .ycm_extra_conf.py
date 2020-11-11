import os.path as p

DIR_OF_THIS_SCRIPT = p.abspath(p.dirname(__file__))

def PythonSysPath(**kwargs):
    sys_path = kwargs['sys_path']

    dependencies = [
        p.join(DIR_OF_THIS_SCRIPT, 'services'),
        p.join(DIR_OF_THIS_SCRIPT, 'cv', 'app'),
        p.join(DIR_OF_THIS_SCRIPT, 'cv', 'torchcv'),
        p.join(DIR_OF_THIS_SCRIPT, 'nlp', 'app'),
        p.join(DIR_OF_THIS_SCRIPT, 'nlp', 'allennlp'),
        p.join(DIR_OF_THIS_SCRIPT, 'ml', 'app'),
        p.join(DIR_OF_THIS_SCRIPT, 'rl', 'app'),
        p.join(DIR_OF_THIS_SCRIPT, 'rl', 'rlpyt'),
    ]

    sys_path[0:0] = dependencies

    return sys_path
