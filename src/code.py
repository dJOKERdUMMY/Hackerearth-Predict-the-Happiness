import pandas as pd
#reading the training file
train = pd.read_csv("train.csv")

#exploratory data analysis for knowing the type of data
#uncomment these lines of code to see the actual output for data analysis
#train.head()
#train.Browser_Used.value_counts()
#found error in data collection as most browsers with same name has been used again with different ID . we need to merge it.
#print("data-type :",type(train.Browser_Used.value_counts()))
#browsers = train.Browser_Used.value_counts().index
#print("data-type of browsers :",type(browsers))
#print("Name of Browsers :")
#for i in browsers:
#    print(i)

#cleaning the data because there is multiple use of same browser with different names
train["Browser_Used"] = train["Browser_Used"].str.replace('Mozilla Firefox', 'Firefox')
train["Browser_Used"] = train["Browser_Used"].str.replace('Mozilla','Firefox')
train["Browser_Used"] = train["Browser_Used"].str.replace('Internet Explorer', 'InternetExplorer')
train["Browser_Used"] = train["Browser_Used"].str.replace('IE', 'InternetExplorer')
train["Browser_Used"] = train["Browser_Used"].str.replace('Google Chrome', 'Chrome')
