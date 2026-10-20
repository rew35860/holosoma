"""Record this experiment's inputs and implementation before starting it."""
from config import *
from solver import SETTINGS


def main():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('An experiment already exists; do not overwrite its protocol.')
    previous=read(FIXED/'protocol.json')
    paths=list(previous['input_sha256'])
    paths.extend(str(FIXED/'variants/contact_ik/clips'/SEQ/name) for name in
                 ('final_v2.npz','metrics.json','per_frame_metrics.npz','run.json'))
    sources=list(CODE.glob('*.py'))
    sources.extend(Path(p) for p in previous['source_sha256'] if p.endswith(('pipeline_v2.py','evaluate_one.py','ref_candidates_metrics/evaluate.py')))
    sources.extend((GUIDANCE_CODE/'render.py',CODE.parent/'pipeline_contact_ik/solver.py'))
    write(OUT/'protocol.json',dict(sequence=SEQ,predeclared_before_run=True,
        purpose='Allow finger adjustment in the previous contact-first IK organizer pilot.',
        previous_experiment=str(FIXED),branch='feature/wuji-pipeline-v2',
        variants=LABELS,active='14 arm/wrist and 40 independent robot finger coordinates.',
        frozen='Saved V2 root, waist, legs and object coordinates exactly unchanged per frame.',
        initialization='Same saved V2 initial trajectory; transport previous active-joint correction each frame.',
        controlled_difference='Add 40 finger columns to contact and collision Jacobians. The existing temporal correction preference, damping, joint/rate bounds and combined trust region now cover all 54 active joints. The arm joint-center and palm preferences are unchanged. No finger pose/bone/mesh prior is added.',
        targets='Same source <30 mm fingertip contact mask and nearest original triangle-mesh surface anchors; unchanged throughout optimization. No IK-derived interaction mesh.',
        hierarchy=previous['hierarchy'],nonlinear_acceptance=previous['nonlinear_acceptance'],
        geometry=previous['geometry'],collision_backend=previous['changed_collision_backend'],
        settings=SETTINGS,evaluation='Frozen evaluator unchanged. Reevaluate supplemental Coal metrics on all three saved trajectories with the same 54-column diagnostic Jacobian. Use all registered collision pairs, including zero-gradient rows, for the comparable feasibility count. Preserve original solver feasibility counts separately.',
        rigid_fit_caution='Rigid-fit residual is a diagnostic of the current finger shape; for the adjustable-finger method it is NOT a lower bound on achievable contact error.',
        resources=dict(cpu_only=True,cpu_quota_percent=300,nice=19,memory_max_gib=5,evaluation_workers=3),
        input_sha256={str(p):sha(p) for p in paths},source_sha256={str(p):sha(p) for p in sources}))
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in CODE.glob('*.py'):(snapshot/path.name).write_bytes(path.read_bytes())
    print(OUT)


if __name__=='__main__':main()
