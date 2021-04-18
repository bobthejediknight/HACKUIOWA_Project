"""
Title: Preprocess.py
Author: Antonio Marino
Purpose: HackUIowa Submission

Summary:
This file preprocesses weightlifting data from the
Open Powerlifting dataset. Specifically, it extracts
the best Squat, Bench, and Deadlift of each individual
at each lifting competition.

1. https://docs.python.org/3/library/csv.html
This source is the python documentation on reading and writing
to a CSV file.

"""
import csv


def preprocess(filePath):
    X = []
    Y = []

    """
    Here, we open the csv file that contains the
    lifting data and iterate through each row of
    the file. Note: the counter i will allow use
    to only access the first 4000 data entries for
    the sake of training the model quickly
    """
    with open(filePath) as csvfile:
      liftingDataReader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
      i = 0
      for row in liftingDataReader:
          rowlist = []
          for item in row:
              list = item.split(",")
              rowlist.append(list)
          i+=1
          if (i > 1):
              if (rowlist[2][0] == '#1'):
                  rowlist[2].pop(0)
                  rowlist[1] = rowlist[1] + rowlist.pop(2)

              if (len(rowlist[1]) == 1):
                  rowlist.pop(1)
                  rowlist.pop(1)

              if (rowlist[2][0] == 'Under'):
                  rowlist.pop(2)
                  rowlist[2].pop(0)
                  rowlist[1] = rowlist[1] + rowlist.pop(2)

              if (rowlist[1][7] == 'Masters'):
                  rowlist[2].pop(0)
                  rowlist[1] = rowlist[1] + rowlist.pop(2)


              #x variables
              #Squat
              x1 = rowlist[1][10]
              if (x1 != ''):
                  x1 = abs(float(x1))
              #Bench
              x2 = rowlist[1][15]
              if (x2 != ''):
                  x2 = abs(float(x2))
              #Deadlift
              x3 = rowlist[1][20]
              if (x3 != ''):
                  x3 = abs(float(x3))

              #y variable
              #WeightClassKg
              y = rowlist[1][9]
              if (y != ''):
                  y = y.split('+')
                  y = int(abs(float(y[0])))

              """
              Here, we only add datapoints to X and Y that
              have defined Squat, Bench, Deadlift, and
              WeightClassKg variables.
              """
              if ((((y != '') and (x3 != '')) and (x2 != '')) and (x1 != '')):
                  X.append([x1, x2, x3])
                  Y.append(y)


          if i == 4000:
            break


    x1min, x1max = X[0][0], X[0][0]
    x2min, x2max = X[0][1], X[0][1]
    x3min, x3max = X[0][2], X[0][2]

    for row in range(len(X)):
        if X[row][0] < x1min:
            x1min = X[row][0]
        if X[row][0] < x2min:
            x2min = X[row][1]
        if X[row][0] < x3min:
            x3min = X[row][2]

        if X[row][0] > x1max:
            x1max = X[row][0]
        if X[row][0] > x2max:
            x2max = X[row][1]
        if X[row][0] > x3max:
            x3max = X[row][2]

    """
    This is our feature normalization step where we
    set all of the X features to something between
    0 and 1. Note: we do not perform this step for
    the Y features because we treat those as categories.
    """
    for row in range(len(X)):
        X[row] = [((x1max - X[row][0])/(x1max - x1min)),((x2max - X[row][1])/(x2max - x2min)),((x3max - X[row][2])/(x3max - x3min))]

    return X, Y
