def col2zero(arr, column):

    for col in range(len(arr)):
        arr[col][column] = 0

    return arr

def row2zero(arr, row_num):

    row_length = len(arr[row_num])
    for idx in range(row_length):
        arr[row_num][idx] = 0

    return arr