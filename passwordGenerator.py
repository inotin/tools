import string, sys, random

def getPassword(length=20):
    symbols = string.ascii_letters+'123456789012345678901234567890'
    ps =[]
    for i in range(int(length/5)):
        ps.append(''.join(random.choices(symbols, k=5)))
    if length%5!=0:
        ps.append(''.join(random.choices(symbols, k=length-5*int(length/5))))
    return '-'.join(ps)

if __name__ == "__main__":
    if len(sys.argv)==1:
        print(getPassword())
    else:
        print(getPassword(int(sys.argv[1])))
