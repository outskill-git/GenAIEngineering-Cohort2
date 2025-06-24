# calculator.py
# A simple calculator with add and subtract functions
import sys

def add(a, b):
    """Add two numbers and return the result."""
    return a + b

def subtract(a, b):
    """Subtract b from a and return the result."""
    return a - b

# Main program
if __name__ == "__main__":
    choice = sys.argv[1]
    num1 = float(sys.argv[2])
    num2 = float(sys.argv[3])

    if choice.upper() == 'ADD':
        result = add(num1, num2)
        print(f"{num1} + {num2} = {result}")
    elif choice.upper() == 'SUB':
        result = subtract(num1, num2)
        print(f"{num1} - {num2} = {result}")
    else:
        print("Invalid input")
