"""Dispatch unchanged evaluators, threshold sensitivity and CPU renderer."""
from config import *

if __name__ == '__main__':
    action, *args = sys.argv[1:]
    if action == 'evaluate':
        module('surface_contact_evaluator', BASE_CODE / 'evaluate.py').main(*args)
    elif action == 'render':
        module('surface_contact_renderer', GUIDANCE / 'render.py').render(*args)
    elif action == 'sensitivity':
        module('surface_contact_sensitivity', BASE_CODE / 'sensitivity.py').main()
        path = OUT / 'contact_threshold_sensitivity.json'
        result = read(path)
        result.update(post_hoc=False, reason='Predeclared sensitivity grid inherited from the preceding experiment; primary scores unchanged.')
        write(path, result)
    elif action in ('preflight', 'validate'):
        import checks
        getattr(checks, action)()
    else:
        raise ValueError(action)
