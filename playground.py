# my_arr = []
# my_arr = [1, 2, 3, 4]

# other_arr = [0] * len(my_arr)
# # other_arr = [0, 0, 0, ... for len(my_arr) many times]


# for i in range(len(my_arr)):
#     # i is an integer representing the index in the loop
#     print("My array", my_arr[i])
#     print("Other array", other_arr[i])
    
# # for val in my_arr:
# #     print(val)
# print("\n\n\n\n\n")

# for index, val, in enumerate(my_arr):
#     print(val)
#     print(other_arr[i])

# def q4(a, b):
#     # Returns a list of the difference between two lists element-wise
#     diff = []
#     for i in range(len(a)):
#         diff.append(b[i] - a[i])

# a = [i for i in range(0, 100)]
# b = [i for i in range(1, 101)]

# print(q4(a,b))

# # Given two lists sized n and m, what process can I use to find the output of a function 
# # on every combination of these variables, and how many outputs would I have?
# # f(x, y) = integer
# # list1 = [x_1, x_2, ..., x_n]
# # list2 = [y_1, y_2, ..., y_m]

# def myfunc():    
#     return (a, b, c, d)

# a = myfunc() # tuple
# first_a, second_a, third_a, fourth_a = myfunc()

# # a == (10, 11, 12, 13)
# # a[0] == 10
# # first_a = a[0]
# # a[1] == 11
# # tuple() - immutable 

# # my_tup = (1, 5, 6, 10)

# # my_tup[0] = 1771

# # dont return a new list, but mutate the current one

# def function_mutation(lst):
#     # orignial: lst[0] cannot be equal to new lst[0]
#     lst[0] = 1
#     return None

# lst = [5555]
# function_mutation(lst)


import pandas as pd

new_dict = {"Day": [1],
"Month": [1]}

test_df = pd.DataFrame(new_dict)


new_df = pd.DataFrame(new_dict)

test_df = test_df.append(new_df)

print(test_df.loc[0, 'Day'])