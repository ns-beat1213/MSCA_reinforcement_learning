import numpy as np

def main():

    # Create a 2D array of size 2x3
    a = np.array([[1,2,3],[4,5,6]])
    print(a)

    # Create a 3D array of size 2x3x4
    b = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                     [[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
    print(b)

main()