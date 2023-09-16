from ModelSelectFinal import ModelSelection

# Test for regression: (dependant value is 'sales')
# reg_test = ModelSelection('insurance.csv')

# Test for dummies and regression (dependant value is 'charges')
# OBS : This takes a very long time to run.
# dummy_test = ModelSelection('Advertising.csv')

# Test for classification and missing data: (dependant value is 'target')
# class_test = ModelSelection('heart.csv')

'''
REFLECTION:

Both the classification and regression processes work, and the correct
conclusion is made. I added some visualization to get a better overview 
of the data and I think it adds to the comprehension well. Offering to 
drop rows with missing values also makes using this algorithm easier. Though you
loose some data and will therefore get a slightly worse accuracy I think
it is worth in exchange for easier use. Of course this isn't ideal for 
data that has a substantial amount of missing values, but I would count
on the user to know better than performing this test on such weak data
anyway.

One issue is that the regression selection takes a long time to run, presumably
because of the GridSearches. 5 of them are done in regression as opposed to 3 in
classification. It does work properly, but I haven't figured out a way
to speed it up. A solution to this would be to not perform GridSeachCV on every model
in exchange with not finding and reporting the best parameters. I think that this
solution is the better alternative due to the considerate improvement in speed,
and you can still offer to use GridSerachVC for the final model to improve it.

I did not figure out a way to identify werther a dataframe is intended
for regression or classification. My solution was to look for object-
type dependant values to identify as classification, but this does not
work when the df has already been digitalized, so this solution still has room 
for improvement. The issue caused by this is that when you choose the wrong
model type, the program will crash.

The last issue is the fact that when doing regression, if the wrong dependant
value is chosen, there will be an error and the program will crash. It does work
for classification, however. Additionally I didn't figure out how to let the user
change to another model to save if unhappy about the suggestion.

In conclusion I am happy with this algorithm, however there area ares that can
be improved upon. Working on this project I was able to use my acquired knowledge 
of supervised learning and I feel like my understanding has been improved upon.
Especially when it comes to GridSearchCV and scoring.

This was a well planned and adjusted exam. 
'''
