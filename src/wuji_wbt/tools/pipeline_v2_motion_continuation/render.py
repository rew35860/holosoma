"""Reuse the same human mesh/cameras, with five synchronized panels."""
from config import *
if __name__=='__main__':
    module('continuation_five_panel_render',GEOMETRY_CODE/'render.py').render(sys.argv[1])
