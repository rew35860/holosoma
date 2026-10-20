"""Bounded persistent CPU run with independent signed geometry preflight."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from config import *


def command(script, args, log):
    start = time.perf_counter()
    with Path(log).open('w') as stream:
        subprocess.run([PY, str(CODE / script), *args], cwd=WORK,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def one(seq):
    dest = destination(seq, 'signed'); dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'completed.json').exists():
        return read(dest / 'completed.json')
    times = {}
    if not (dest / 'final_v2.npz').exists():
        times['solve_process_s'] = command('solver.py', [seq, 'signed'], dest / 'solver.log')
    times['evaluation_s'] = command('tasks.py', ['evaluate', seq, 'signed'], dest / 'evaluation.log')
    times['link_audit_s'] = command('tasks.py', ['links', seq], dest / 'link_audit.log')
    result = dict(sequence=seq, variant='signed', state='complete', timings=times)
    write(dest / 'completed.json', result)
    return result


def main():
    start = time.perf_counter()
    write(OUT / 'status.json', dict(state='numerical_checks', completed=2, total=4))
    command('tasks.py', ['preflight'], OUT / 'preflight.log')
    command('report.py', [], OUT / 'report.log')
    done, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(one, seq): seq for seq in COHORT}
        while pending:
            finished, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in finished:
                seq = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as exc:
                    errors.append(dict(sequence=seq, error=repr(exc)))
            write(OUT / 'status.json', dict(state='running', completed=2+len(done), total=4,
                  new_solves_completed=len(done), new_solves_total=2, errors=errors,
                  elapsed_seconds=time.perf_counter()-start, pending=list(pending.values())))
            if finished:
                command('report.py', [], OUT / 'report.log')
    if errors:
        raise RuntimeError(errors)
    command('tasks.py', ['sensitivity'], OUT / 'sensitivity.log')
    write(OUT / 'status.json', dict(state='rendering', completed=4, total=4,
          new_solves_completed=2, new_solves_total=2, elapsed_seconds=time.perf_counter()-start))
    with ThreadPoolExecutor(max_workers=2) as pool:
        renders = [pool.submit(command, 'tasks.py', ['render', seq], OUT / f'render_{seq}.log') for seq in COHORT]
        render_times = [f.result() for f in renders]
    command('tasks.py', ['validate'], OUT / 'validation.log')
    write(OUT / 'status.json', dict(state='complete', completed=4, total=4,
          new_solves_completed=2, new_solves_total=2, errors=[],
          elapsed_seconds=time.perf_counter()-start, render_seconds=render_times, validation_passed=True))
    command('report.py', [], OUT / 'report.log')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status = read(OUT / 'status.json'); status.update(state='failed', error=repr(exc))
        write(OUT / 'status.json', status)
        raise
