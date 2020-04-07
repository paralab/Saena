import sys

if len(sys.argv) != 3:
    print("usage: <start #processors> <end #processors> <matrix_file_address>")

# for x in range(3, int(sys.argv[1])+1):
#     print(x)

import subprocess
# args = ("bin/bar", "-c", "somefile.xml", "-d", "text.txt", "-r", "aString", "-f", "anotherString")
#Or just:
#args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
# args = "./Saena 3 3 3"

for x in range(int(sys.argv[1]), int(sys.argv[2]), 1):
    args = "mpirun", "-np", str(x), "./Saena_matmat", sys.argv[3]
    # print(x);
    # print(args);
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print(output)
    # print("\n\n==============================================================")
    print("==============================================================")
    # print("==============================================================\n\n")
