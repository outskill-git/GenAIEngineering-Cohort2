import sys

def add(a,b):
  print(a+b)

if __name__ == "__main__":
    num1 = float(sys.argv[1])
    num2 = float(sys.argv[2])
    print("num1 is ", num1)
    print("num2 is ", num2)

    add(num1, num2)
    # add(num1, num2)
