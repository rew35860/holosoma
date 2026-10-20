"""Reuse the CPU-only human/robot renderer with this pilot's isolated outputs."""
from config import *

if __name__=='__main__':
    renderer=module('contact_ik_renderer',GUIDANCE_CODE/'render.py')
    renderer.render(SEQ)
