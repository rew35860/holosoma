"""Load the isolated OmniRetarget source, bypassing unrelated editable installs."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parents[2]/'src/holosoma_retargeting'
sys.meta_path=[f for f in sys.meta_path if '_EditableFinder' not in getattr(f,'__name__',type(f).__name__)]
sys.path.insert(0,str(root))
import holosoma_retargeting.config_types.data_type as probe
assert root in Path(probe.__file__).resolve().parents,probe.__file__
print('Isolated OmniRetarget:',probe.__file__,flush=True)
entry=root/'holosoma_retargeting/examples/robot_retarget.py'
sys.argv[0]=str(entry);runpy.run_path(str(entry),run_name='__main__')
