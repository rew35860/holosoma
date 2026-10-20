"""CPU human + four robot variants, synchronized full motion and close-up PNGs."""
from config import *

if __name__=='__main__':
    renderer=module('contact_geometry_renderer',GUIDANCE/'render.py')
    renderer.render(sys.argv[1])
