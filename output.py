def output(*args, outputFileW):
    for s in args:
        outputFileW.write("\n" + str(s))
        print(s)
        outputFileW.flush()