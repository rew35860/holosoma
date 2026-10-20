"""Reuse frozen metrics and CPU rendering; keep signed residual out of metrics."""
from config import *

if __name__ == '__main__':
    action, *args = sys.argv[1:]
    if action == 'evaluate':
        module('signed_contact_evaluator', BASE_CODE / 'evaluate.py').main(*args)
    elif action == 'render':
        module('signed_contact_renderer', GUIDANCE / 'render.py').render(*args)
    elif action == 'sensitivity':
        module('signed_contact_sensitivity', BASE_CODE / 'sensitivity.py').main()
        path = OUT / 'contact_threshold_sensitivity.json'; result = read(path)
        result.update(post_hoc=False, reason='Predeclared inherited sensitivity grid; no primary metric changed.')
        write(path, result)
    elif action == 'links':
        import link_audit
        link_audit.main(*args)
    elif action in ('preflight', 'validate'):
        import checks
        getattr(checks, action)()
    else:
        raise ValueError(action)
