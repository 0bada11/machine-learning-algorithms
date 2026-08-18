import pandas as pd
from matplotlib.pyplot import subplot
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# ---------------- reading the data -----------------------

teams = pd.read_csv("teams.csv")
teams = teams[["team","country","year","athletes", "age", "prev_medals","medals"]]
# finding Correlation between Target and all other features
print(teams[["year","athletes", "age","prev_medals","medals"]].corr()["medals"])
# we see that we got very good correlation 0.92 for prev_medals and 0.84 for athletes


# --------------- data cleaning ---------------

#-------  find the missing values
# this mean find any rows that have null value
print(teams[teams.isnull().any(axis=1)])

# now we should fill or drop this records that have null value
teams = teams.dropna()

#----------- splitting & visualisation ------------

#------ 1) now split the data
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# ----- 2) see #of rows in train and test
print("training rows ",train.shape[0])
print("testing rows ",test.shape[0])

#------ 3) visualisation
plt.style.use("dark_background")

plt.figure(figsize = (17,5))
plt.subplot(1, 3, 1)
plt.scatter(train["athletes"], train["medals"], color="red",s=50, alpha=0.5, label="Train")
plt.scatter(test["athletes"] , test["medals"], color="pink",s=50, alpha=0.5, label="Test")
plt.xlabel("Athletes")
plt.ylabel("Medals")
plt.title("Medals vs Athletes")

plt.subplot(1, 3, 2)
plt.scatter(train["prev_medals"], train["medals"], color="blue",s=50, alpha=0.5, label="Train")
plt.scatter(test["prev_medals"] , test["medals"], color="pink",s=50, alpha=0.5, label="Test")
plt.xlabel("prev_medals")
plt.ylabel("Medals")
plt.title("Medals vs prev_medals")

plt.subplot(1, 3, 3, projection='3d')
plt.scatter(train["prev_medals"], train["athletes"],train["medals"],color="cyan", alpha=0.5, label="Train")
plt.scatter(test["prev_medals"] ,test["athletes"],test["medals"],color="pink", alpha=0.5, label="Test")
plt.xlabel("prev_medals")
plt.ylabel("Medals")
plt.title("Medals vs prev_medals vs Athletes")

plt.show()

# --------- training the model --------

reg = LinearRegression()
predictors = ["athletes","prev_medals"] # features
targets = ["medals"] # target
reg.fit(train[predictors],train[targets]) # this fit function take (inputs features , target)

# ---- finding the optimal slope(w) and bais(b)
print("slope",reg.coef_)
print("intercept",reg.intercept_)

# ---- find the predictions
# in predictions, we pass only the test features to see if we can predict right predictions on the test target
predictions = reg.predict(test[predictors])
print(predictions)
# we clearly see that the predictions contain negative values and non-rounded values
# make new col to the data frame called predictions
test["predictions"] = predictions
test.loc[test["predictions"]< 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()
print(test)

# ---- measure the error

error_abs = mean_absolute_error(test["medals"], test["predictions"])
error_squared = mean_squared_error(test["medals"], test["predictions"])
score = reg.score(train[predictors],train[targets])

print("error abs :",error_abs)
print("error squared ",error_squared)
print("score ",score)

