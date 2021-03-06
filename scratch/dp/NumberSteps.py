import sys
import getopt

def GetMinimumStepsTopRecusive(n, MemoList):

    #
    # Base case.
    #

    if n == 1:
        return 0

    #
    # Check if already solved.
    #

    if MemoList[n] != -1:
        return MemoList[n]

    #
    # Solve solution & save it.
    #

    #
    # This is the -1 step.
    #

    IntermediateSteps = 1 + GetMinimumStepsTopRecusive(n-1, MemoList)

    #
    # This is the /2 step 
    #

    if n % 2 == 0:
        IntermediateSteps = min(IntermediateSteps,
                                1 + GetMinimumStepsTopRecusive(n/2, MemoList))

    if n % 3 == 0:
        IntermediateSteps = min(IntermediateSteps,
                                1 + GetMinimumStepsTopRecusive(n/3, MemoList))
    
    MemoList[n] = IntermediateSteps
    return IntermediateSteps

def GetMinimumStepsBottom(n, MemoList):
    MemoList[1] = 0

    for i in range(2, n+1):
        MemoList[i] = 1 + MemoList[i-1]
        if i % 2 == 0:
            MemoList[i] = min(MemoList[i], 1 + MemoList[i/2])

        if i % 3 == 0:
            MemoList[i] = min(MemoList[i], 1 + MemoList[i/3])

    return MemoList[n]


def MinimumSteps(n):
    MemoList = [-1] * (n+1)

    #
    # Top down recursive approach.
    #

    Steps = GetMinimumStepsBottom(n, MemoList);
    return Steps

#
# Main function of the program which parses command line and calls helper
# functions.
#

def main():

    if len(sys.argv) != 2:
        print "Bad input!"
        sys.exit(0)

    Number = int(sys.argv[1])
    MemoListBottom = [-1] * (Number + 1)
    MemoListTop = [-1] * (Number + 1)
    #Steps = MinimumSteps(Number)
    print "Minimum Steps Bottom: ", GetMinimumStepsBottom(Number, MemoListBottom)
    #print "Minimum Steps Top: ", GetMinimumStepsTopRecusive(Number, MemoListTop)
    return



#
# Run main program.
#

main()
