import os, sys

os.system('nvcc cuda/{0}.cu -o bin/{0} -Xcompiler "/wd 4819"'.format(sys.argv[1]))