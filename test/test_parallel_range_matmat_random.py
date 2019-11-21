import sys

if len(sys.argv) != 5:
    print "\nusage: <#processors> <local size> <start density> <end density> <density step size>"
    print "\nexample: python ./test_parallel_range_matmat_random.py 2 3 0.1 0.6 0.03\n"

# for x in range(3, int(sys.argv[1])+1):
#     print(x)

import subprocess
# args = ("bin/bar", "-c", "somefile.xml", "-d", "text.txt", "-r", "aString", "-f", "anotherString")
#Or just:
#args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
# args = "./Saena 3 3 3"

start_range = int(float(sys.argv[3]) * 1000);
end_range   = int(float(sys.argv[4]) * 1000);
step_size   = int(float(sys.argv[5]) * 1000);

for x in range(start_range, end_range, step_size):
    # print str(float(x)/100)
    args = ("mpirun", "-np", sys.argv[1], "./Saena_matmat", sys.argv[2], str(float(x)/1000))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print output
    print "================================================================="
    # print "================================================================="
    # print "================================================================="
