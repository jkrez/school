import sys
import getopt
import random

Size = 10
#Sequence = [random.randint(0,9) for i in range(Size)]
Sequence = [9,0,9,1,9,2,9,3,8,4]

Memo = [1] * Size

#
# Iterate over the sequence storing the longest sequence so far.
#

for i in range(Size):
    for j in range(i+1, Size):
        if (Sequence[j] >= Sequence[i] and
            Memo[j] <= Memo[i]):

            Memo[j] = Memo[i] + 1;

print "Sequence[]: ", Sequence
print "Memo[]: ", Memo
print "Max: ", max(Memo)


