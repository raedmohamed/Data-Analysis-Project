import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##################################### Load CSV dataset files ##################################### 
sales = pd.read_csv("data.csv")
weather = pd.read_csv("weather.csv")
fuel = pd.read_csv("fuel pricing.csv")


##################################### Functions ##################################### 

#function to check if there are any columns contain negative values
def CheckValue(df):
    for col in df.columns:
        if any(str(df[col].dtype).startswith(x) for x in ["float","int","unit"]):
            if (df[col] < 0).any():
                print(col, "!!contains negative values!!")
            else:
                print(col, "does not contain negative values")

def convertNegativeToPositive(data,col):
    data[col] = data[col].apply(lambda x:x*-1 if x<0 else x)
    

##################################### Sales DataSet ##################################### 

print("#"*20+" Sales DataSet "+"#"*20+"\n")
print("Comment: display the top ten for each dataset")
print(sales.head(10))

print("\n"+"Comment: display all columns and their data types")
print(sales.dtypes) 

print("\n"+"Comment: display basic statistics for numeric columns (Count, mean, std, min, max)")
print(sales.describe().loc[["count", "mean", "std", "min", "max"]])

print("\n"+"Comment: Check if there are any missing values in columns")
print(sales.isna().any(axis=0)) #No NaN

print("\n"+"Comment: Check if there are any Columns contain Negative values")
CheckValue(sales) #Weekly_Sales column contains negative values

#start handling negative values to convert negative to positive value
convertNegativeToPositive(sales,"Weekly_Sales")
print("\n"+"Comment: Check negativity again after convert Negative To Positive ")
CheckValue(sales)


##################################### Weather DataSet ##################################### 

print("\n"+"#"*20+" Weather DataSet "+"#"*20+"\n")
print("Comment: displays the top ten for each dataset")
print(weather.head(10))

print("\n"+"Comment: displays all columns and their data types")
print(weather.dtypes)

print("\n"+"Comment: display basic statistics for numeric columns (Count, mean, std, min, max)")
print(weather.describe().loc[["count", "mean", "std", "min", "max"]])

print("\n"+"Comment: Check if there are any missing values in columns")
print(weather.isna().any(axis=0)) #No NaN

print("\n"+"Comment: Check if there are any Columns contain Negative values")
CheckValue(weather) #Temperature column contains negative values




##################################### Fuel DataSet ##################################### 

print("\n"+"#"*20+" Fuel DataSet "+"#"*20+"\n")
print("Comment: displaysthe top ten for each dataset")
print(fuel.head(10))

print("\n"+"Comment: displays all columns and their data types")
print(fuel.dtypes)

print("\n"+"Comment: display basic statistics for numeric columns (Count, mean, std, min, max)")
print(fuel.describe().loc[["count", "mean", "std", "min", "max"]])

print("\n"+"Comment: Check if there are any missing values in columns")
print(fuel.isna().any(axis=0)) #No NaN

print("\n"+"Comment: Check if there are any Columns contain Negative values")
CheckValue(fuel) #No negative values

##################################### Merge DataSets ##################################### 
print("\n"+"#"*20+" Merge DataSets "+"#"*20+"\n")
salesMergeWeather = pd.merge(sales,weather,left_on=["Store","Date"],right_on=["Store","Date"])
salesMergeWeatherMergeFuel = pd.merge(salesMergeWeather,fuel,left_on=["Store","Date"],right_on=["Store","Date"])
print(salesMergeWeatherMergeFuel.head(10))
#################################### Chart Of Weekly Sales #################################
weeks = np.arange(1, len(sales["Date"])+1)
plt.style.use("seaborn")

#First graph(plot) for weekly sales 
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
plt.plot(weeks,sales["Weekly_Sales"],color="blue",label="Sales")
plt.title("Weekly_Sales")
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.ylim(sales["Weekly_Sales"].min(),sales["Weekly_Sales"].max())
plt.legend(loc="best")

#second graph(scatter) for weekly sales 
plt.subplot(1,2,2)
plt.scatter(weeks,sales["Weekly_Sales"],marker="|",s=1,color="blue",label="Sales")
plt.title("Weekly_Sales")
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.ylim(sales["Weekly_Sales"].min(),sales["Weekly_Sales"].max())
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
####################################Chart Of Brand Sells####################################
StoreGroupByWeekly_Sales = sales["Weekly_Sales"].groupby(sales["Store"]).sum()
print(StoreGroupByWeekly_Sales)
brand = list(set(sales["Store"]))

plt.figure(figsize=(12, 6))
plt.bar(brand, StoreGroupByWeekly_Sales,label="Stores")
plt.title('Brand Sales')
plt.xlabel('Brand')
plt.ylabel('Sales')
plt.xticks(np.arange(1, len(brand)+1, 1),rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="upper right")
plt.show()
##################### Determine The Top ten Selling Stores######################

topTenbrands = StoreGroupByWeekly_Sales.nlargest(10)

#########################Histogram Top 10 Stores Sales##########################

topTenBrandsDataFrame = pd.DataFrame({"Stores":topTenbrands.index,"Sales":topTenbrands})
topTenBrandsDataFrame["Stores"] = topTenBrandsDataFrame["Stores"].astype(str)


plt.subplot(1,2,1)
plt.style.use("seaborn")
plt.hist(topTenBrandsDataFrame["Stores"],bins=42,align="mid")
plt.xlabel("Brands")
plt.ylabel("Sales")


plt.subplot(1,2,2)
plt.style.use("seaborn")
print(topTenBrandsDataFrame["Stores"])
plt.bar(topTenBrandsDataFrame["Stores"],topTenBrandsDataFrame["Sales"],width=.5,color="blue")
plt.plot(topTenBrandsDataFrame["Stores"],topTenBrandsDataFrame["Sales"],"-ro")
plt.ticklabel_format(style='plain', axis='y')
plt.xlabel("Brands")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()
###########################Chart Average Weekly Sales Holidays and Non-Holidays##########################

StoreGroupByWeekly_SalesAndHoliday = sales["Weekly_Sales"].groupby([sales["Store"],sales["Holiday"]]).mean()
print(StoreGroupByWeekly_SalesAndHoliday)

topTenBrandsOnHolidayAndNot = StoreGroupByWeekly_SalesAndHoliday.nlargest(10)
print(topTenBrandsOnHolidayAndNot)

topTenBrandsOnHolidayAndNotDataFrame = topTenBrandsOnHolidayAndNot.reset_index()
topTenBrandsOnHolidayAndNotDataFrame.columns = ["Store", "Holiday", "Sales"]

plt.bar(topTenBrandsOnHolidayAndNotDataFrame["Store"].astype(str) + ' - ' + topTenBrandsOnHolidayAndNotDataFrame["Holiday"].astype(str),
        topTenBrandsOnHolidayAndNotDataFrame["Sales"], color=['blue', 'orange'])
plt.xlabel("Store - Holiday")
plt.ylabel("Average Weekly Sales")
plt.title("Average Weekly Sales for Top Ten Stores (Holidays vs. Non-Holidays)")
plt.xticks(rotation=45)
plt.show()
##############################################
