import sys

if len(sys.argv) != 4:
    print("usage: <Laplacian size start> <Laplacian size end> <#nprocs>")

# for x in range(3, int(sys.argv[1])+1):
#     print(x)

import subprocess
# args = ("bin/bar", "-c", "somefile.xml", "-d", "text.txt", "-r", "aString", "-f", "anotherString")
#Or just:
#args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
# args = "./Saena 3 3 3"

for x in range(int(sys.argv[1]), int(sys.argv[2]) + 1, 1):
    for y in range(int(sys.argv[1]), int(sys.argv[2]) + 1, 1):
        for z in range(int(sys.argv[1]), int(sys.argv[2]) + 1, 1):
            args = "mpirun", "-np", str(sys.argv[3]), "./experiments/profile", str(x), str(y), str(z)
            # print(x)
            # print(y)
            # print(z)
            print(args)
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            output = popen.stdout.read()
            print(output)
            # print("\n\n==============================================================")
            print("==============================================================")
            # print("==============================================================\n\n")
