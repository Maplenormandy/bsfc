import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

print sys.path

from helpers.bsfc_helper import HirexData
print "OK!"
