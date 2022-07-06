import numpy as np,pickle
from prettytable import PrettyTable


# Activation Function
def activationFunction(linearUnit):
    if linearUnit <= 0:
        return 0
    else:
        return 1

# AND Gate
def andGate(Input, weights, bias):
    return activationFunction(np.dot(weights, Input)+bias)

# Print in table format.
def printTable(a,b,output):
    myTbale=PrettyTable(["a","b","AND"])
    myTbale.add_row([a,b,output])
    print(myTbale)

if __name__ == '__main__':
    #Load the model
    with open('AND.pkl','rb') as f:
            parameters=pickle.load(f)
    
    while True:
        try: 
            a,b=input("Enter a and b (N to Exit) = ").split()
            if(int(a)<0 or int(a)>1 or int(b)<0 or int(b)>1 ):
                print("Invalid Inputs")
                continue
            printTable(a,b, andGate(np.array([int(a), int(b)]),parameters[0],parameters[1]))
        except:
            break