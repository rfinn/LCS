"""numpy.append() vs list.append()"""
import numpy as np
from time import time

def numpy_append():
    arr = np.empty((1, 0), int)
    for i in range(100000):
        arr = np.append(arr, np.array(i))
    return arr

def numpy_fill():
    N = 100000
    arr = np.zeros(100000, int)
    for i in range(N):
        arr[i] = i
    return arr

def list_append():
    list_1 = []
    for i in range(100000):
        list_1.append(i)
    return list_1

def creating_list_implicitly():
    list_1 = [1 for i in range(100000)]
    return list_1

def creating_list_from_ones():
    list_1 = np.ones(100000,'i')
    return list_1

def main ():
    #Start timing numpy array
    start1 = time()
    new_np_arr = numpy_append()
    #End timing
    end1 = time()
    #Time taken
    print(f"Computation time of appending the numpy array : {end1 - start1}")    #Start timing numpy array

    start2 = time()
    new_list = list_append()
    #End timing
    end2 = time()
    #Time taken
    print(f"Computation time of the list: {end2 - start2}")
    #Testing
    assert list(new_np_arr) == new_list, "Arrays tested are not the same"

    start3 = time()
    new_np_arr2 = numpy_fill()
    #End timing
    end3 = time()
    #Time taken
    print(f"Computation time of filling numpy array : {end3 - start3}")    #Start timing numpy array

    start3 = time()
    new_np_arr2 = creating_list_implicitly()
    #End timing
    end3 = time()
    #Time taken
    print(f"Computation time of creating list implicitly : {end3 - start3}")    #Start timing numpy array

    start3 = time()
    new_np_arr2 = creating_list_from_ones()
    #End timing
    end3 = time()
    #Time taken
    print(f"Computation time of creating list from np.ones : {end3 - start3}")    #Start timing numpy array
    


if __name__ == "__main__":
    main()

