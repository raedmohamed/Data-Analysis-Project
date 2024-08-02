import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import screeninfo
from matplotlib import patches

screen = screeninfo.get_monitors()[0]
screen_width = screen.width
screen_height = screen.height

fig_width = screen_width / 100 -3.5
fig_height = screen_height / 100 - 3.5
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

####################################Chart Of Weekly Sales#################################
weekly_SalesGroupbyDate = salesMergeWeatherMergeFuel["Weekly_Sales"].groupby(salesMergeWeatherMergeFuel["Date"]).sum()
weekly_SalesGroupbyDateDataFrame = weekly_SalesGroupbyDate.reset_index()
weekly_SalesGroupbyDateDataFrame.columns = ["Date","Weekly_Sales"]
# print("\n"+"#"*20+" weekly_Sales Groupedby Date DataSet "+"#"*20+"\n")
# print(weekly_SalesGroupbyDateDataFrame)

weeks = np.arange(1, len(weekly_SalesGroupbyDateDataFrame["Date"])+1)
plt.figure(figsize=(fig_width, fig_height)).suptitle("Chart to illustrate if weekly sales are increasing or decreasing over time")

plt.subplot(1,2,1)
plt.plot(weeks,weekly_SalesGroupbyDateDataFrame["Weekly_Sales"],color="blue",label="Sales")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("weekly sales over time")
plt.title("Weekly_Sales (Plot)")
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.xticks(np.arange(0,len(weeks)+1,10))
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")

#second graph(scatter) for weekly sales 
plt.subplot(1,2,2)
plt.scatter(weeks,weekly_SalesGroupbyDateDataFrame["Weekly_Sales"],marker="+",s=50,color="blue",label="Sales")
plt.title("Weekly_Sales (Scatter)")
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.xticks(np.arange(0,len(weeks)+1,10))
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")

plt.tight_layout()
plt.show()

####################################Chart Of Brand Sells####################################
weekly_SalesGroupbyBrand = salesMergeWeatherMergeFuel["Weekly_Sales"].groupby(salesMergeWeatherMergeFuel["Category"]).sum()
weekly_SalesGroupbyBrandDataFrame = weekly_SalesGroupbyBrand.reset_index()
weekly_SalesGroupbyBrandDataFrame.columns = ["Brand","Weekly_Sales"]
# print("\n"+"#"*20+" weekly_Sales Groupedby Brand(Category) DataSet "+"#"*20+"\n")
# print(weekly_SalesGroupbyBrandDataFrame)

plt.figure(figsize=(fig_width, fig_height)).suptitle("Chart to show how much each brand sells")
plt.bar(weekly_SalesGroupbyBrandDataFrame["Brand"].astype("str"), weekly_SalesGroupbyBrandDataFrame["Weekly_Sales"],width=0.8,label="Brands")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("Brand Sells")
plt.title('Brand Sales (Bar)')
plt.xlabel('Brand')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")
plt.show()

##################### Determine The Top ten Selling Stores ######################
weekly_SalesGroupbyStore = salesMergeWeatherMergeFuel["Weekly_Sales"].groupby(salesMergeWeatherMergeFuel["Store"]).sum()
weekly_SalesGroupbyStoreDataFrame = weekly_SalesGroupbyStore.reset_index()
weekly_SalesGroupbyStoreDataFrame.columns = ["Store","Weekly_Sales"]
# topTenSellingStore = weekly_SalesGroupbyStoreDataFrame.sort_values(by='Weekly_Sales', ascending=False)
# print(topTenSellingStore.head(10))

# print("\n"+"#"*20+" weekly_Sales Groupedby Store DataSet "+"#"*20+"\n")
# print(topTenSellingStore.index.astype("str"))

topTenSellingStore = weekly_SalesGroupbyStore.nlargest(10)
topTenSellingStoreDataFrame = topTenSellingStore.reset_index()
topTenSellingStoreDataFrame.columns = ["Store","Weekly_Sales"]
# print(topTenSellingStoreDataFrame)

plt.figure(figsize=(fig_width, fig_height)).suptitle("Top 10 stores sales")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("Top ten Selling Stores")

plt.subplot(1,2,1)
plt.style.use("seaborn")
plt.title("Histogram")
plt.hist(topTenSellingStore.index.astype("str"),bins=19)
plt.xlabel("Stores")
plt.ylabel("Sales")


plt.subplot(1,2,2)
plt.style.use("seaborn")
plt.title("Bar")
plt.bar(topTenSellingStoreDataFrame["Store"].astype("str"),topTenSellingStoreDataFrame["Weekly_Sales"],width=.5,color="blue")
plt.plot(topTenSellingStoreDataFrame["Store"].astype("str"),topTenSellingStoreDataFrame["Weekly_Sales"],"-ro")
plt.ticklabel_format(style='plain', axis='y')
plt.xlabel("Stores")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()

###########################Chart Average Weekly Sales Holidays and Non-Holidays##########################
### Chart Average Weekly Sales On -Only- Holidays ###
salesMergeWeatherMergeFuelOnlyHoliday = salesMergeWeatherMergeFuel.drop(salesMergeWeatherMergeFuel[salesMergeWeatherMergeFuel["Holiday"]==False].index)
Weekly_SalesGroupByStoreAndOnlyHoliday = salesMergeWeatherMergeFuelOnlyHoliday["Weekly_Sales"].groupby([salesMergeWeatherMergeFuelOnlyHoliday["Store"],
                                                                                                        salesMergeWeatherMergeFuelOnlyHoliday["Holiday"]]).mean()
#print(salesMergeWeatherMergeFuelOnlyHoliday)
topTenWeekly_SalesGroupByStoreAndOnlyHoliday = Weekly_SalesGroupByStoreAndOnlyHoliday.nlargest(10)
topTenWeekly_SalesGroupByStoreAndOnlyHolidayDataFrame = topTenWeekly_SalesGroupByStoreAndOnlyHoliday.reset_index()
topTenWeekly_SalesGroupByStoreAndOnlyHolidayDataFrame.columns =["Store","Holiday","Weekly_Sales"]
print(topTenWeekly_SalesGroupByStoreAndOnlyHolidayDataFrame)

plt.figure(figsize=(fig_width, fig_height)).suptitle("Average Weekly Sales Holidays and Non-Holidays")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("Average Weekly Sales Holidays and Non-Holidays")

plt.subplot(2,2,1)
plt.bar(topTenWeekly_SalesGroupByStoreAndOnlyHolidayDataFrame["Store"].astype(str) + ' - ' + topTenWeekly_SalesGroupByStoreAndOnlyHolidayDataFrame["Holiday"].astype(str),
        topTenWeekly_SalesGroupByStoreAndOnlyHolidayDataFrame["Weekly_Sales"], color='blue')
plt.xlabel("Store - Holiday(True)")
plt.ylabel("Average Weekly Sales")
plt.title("Average Weekly Sales for Top Ten Stores Holidays(Only)")
plt.xticks(rotation=45)



### Chart Average Weekly Sales On -NOT- Holidays ###
salesMergeWeatherMergeFuelNotHoliday = salesMergeWeatherMergeFuel.drop(salesMergeWeatherMergeFuel[salesMergeWeatherMergeFuel["Holiday"]==True].index)
Weekly_SalesGroupByStoreAndNotHoliday = salesMergeWeatherMergeFuelNotHoliday["Weekly_Sales"].groupby([salesMergeWeatherMergeFuelNotHoliday["Store"],
                                                                                                        salesMergeWeatherMergeFuelNotHoliday["Holiday"]]).mean()
#print(salesMergeWeatherMergeFuelNotHoliday)
topTenWeekly_SalesGroupByStoreAndNotHoliday = Weekly_SalesGroupByStoreAndNotHoliday.nlargest(10)
topTenWeekly_SalesGroupByStoreAndNotHolidayDataFrame = topTenWeekly_SalesGroupByStoreAndNotHoliday.reset_index()
topTenWeekly_SalesGroupByStoreAndNotHolidayDataFrame.columns =["Store","Holiday","Weekly_Sales"]
# print(topTenWeekly_SalesGroupByStoreAndNotHolidayDataFrame)

plt.subplot(2,2,2)
plt.bar(topTenWeekly_SalesGroupByStoreAndNotHolidayDataFrame["Store"].astype(str) + ' - ' + topTenWeekly_SalesGroupByStoreAndNotHolidayDataFrame["Holiday"].astype(str),
        topTenWeekly_SalesGroupByStoreAndNotHolidayDataFrame["Weekly_Sales"], color='orange')
plt.xlabel("Store - Holiday(False)")
plt.ylabel("Average Weekly Sales")
plt.title("Average Weekly Sales for Top Ten Stores Non-Holidays(Only)")
plt.xticks(rotation=45)


### Chart Average Weekly Sales Holidays and Non-Holidays ###
Weekly_SalesGroupByStoreAndHoliday = salesMergeWeatherMergeFuel["Weekly_Sales"].groupby([salesMergeWeatherMergeFuel["Store"],salesMergeWeatherMergeFuel["Holiday"]]).mean()
#print(Weekly_SalesGroupByStoreAndHoliday)

topTenWeekly_SalesGroupByStoreAndHoliday = Weekly_SalesGroupByStoreAndHoliday.nlargest(10)
topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame = topTenWeekly_SalesGroupByStoreAndHoliday.reset_index()
topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame.columns =["Store","Holiday","Weekly_Sales"]
#print(topTenWeekly_SalesAndHolidayGroupByStoreDataFrame)

colors = ['blue' if x == True else 'orange' for x in topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame["Holiday"]]
legend_colors = ["blue","orange"]
unique_x_values = list(set(topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame["Holiday"]))
labels = ['Holiday', 'Non-Holiday']
legend_handles = [patches.Patch(color=legend_colors[i], label=labels[i]) for i in range(len(unique_x_values))]
plt.subplot(2,2,3)
plt.bar(topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame["Store"].astype(str) + ' - ' + topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame["Holiday"].astype(str),
        topTentopTenWeekly_SalesGroupByStoreAndHolidayDataFrame["Weekly_Sales"], color=colors)

plt.legend(handles=legend_handles,loc="best")
plt.xlabel("Store - Holiday(True or False)")
plt.ylabel("Average Weekly Sales")
plt.title("Average Weekly Sales for Top Ten Stores Holidays vs Non-Holidays(Mix)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

##################Chart Average Weekly Sales For Each Brand Department For The Top 10 Selling Stores################
Weekly_SalesGroupByStoreAndCategory = salesMergeWeatherMergeFuel["Weekly_Sales"].groupby([salesMergeWeatherMergeFuel["Store"],salesMergeWeatherMergeFuel["Category"]]).mean()

topTenStores = pd.DataFrame(topTenSellingStoreDataFrame["Store"])
StoreAndCategoryAndWeekly_Sales = pd.DataFrame(salesMergeWeatherMergeFuel[["Store","Category","Weekly_Sales"]])
topTenStoresMergeStoreAndCategoryAndWeekly_Sales = pd.merge(topTenStores,StoreAndCategoryAndWeekly_Sales,left_on=["Store"],right_on=["Store"])
CategorySalesOfTopTenStores = topTenStoresMergeStoreAndCategoryAndWeekly_Sales["Weekly_Sales"].groupby([topTenStoresMergeStoreAndCategoryAndWeekly_Sales["Store"],topTenStoresMergeStoreAndCategoryAndWeekly_Sales["Category"]]).sum()
CategorySalesOfTopTenStoresDataFrame = CategorySalesOfTopTenStores.reset_index()
CategorySalesOfTopTenStoresDataFrame.columns = ["Store","Category","Weekly_Sales"]
Weekly_SalesGroupByCategoryOfTopTenSales = CategorySalesOfTopTenStoresDataFrame["Weekly_Sales"].groupby(CategorySalesOfTopTenStoresDataFrame["Category"]).mean()
Weekly_SalesGroupByCategoryOfTopTenSalesDataFrame = Weekly_SalesGroupByCategoryOfTopTenSales.reset_index()
Weekly_SalesGroupByCategoryOfTopTenSalesDataFrame.columns = ["Category","Weekly_Sales"]
# print(topTenStores)
# print(StoreAndCategory)
# print(asd)
print(Weekly_SalesGroupByStoreAndCategory.loc[[20,4,14,13,2,10,27,6,1,39]])
print(Weekly_SalesGroupByCategoryOfTopTenSalesDataFrame)

plt.figure(figsize=(fig_width, fig_height)).suptitle("Bar chart of average weekly sales for each brand department for the top 10 selling stores")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("Average weekly sales for each brand of top ten stores")
plt.bar(Weekly_SalesGroupByCategoryOfTopTenSalesDataFrame["Category"].astype("str"),Weekly_SalesGroupByCategoryOfTopTenSalesDataFrame["Weekly_Sales"])
plt.xlabel("Brands_Of_Top_Ten_Stores")
plt.ylabel("Weekly_Sales")
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.show()
############################ Relationship Between Weekly Sales And Weather Temperature ###############################
Weekly_SalesGroupByTemp = salesMergeWeatherMergeFuel["Weekly_Sales"].groupby(salesMergeWeatherMergeFuel["Temperature"]).sum()
Weekly_SalesGroupByTempDataFrame = Weekly_SalesGroupByTemp.reset_index()
Weekly_SalesGroupByTempDataFrame.columns = ["Temperature","Weekly_Sales"]
print(Weekly_SalesGroupByTempDataFrame)

plt.figure(figsize=(fig_width, fig_height)).suptitle("line chart to show the relationship between weekly sales and weather Temperature.")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("relationship between weekly sales and weather Temperature")
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Weekly_SalesGroupByTempDataFrame["Weekly_Sales"])
plt.xlabel("Temperature in °F")
plt.ylabel("Weekly_Sales")
plt.ticklabel_format(style='plain', axis='y')
plt.show()

####################################
# Fuel_PriceGroupByWeekly_SalesAndTemperature = salesMergeWeatherMergeFuel["Fuel_Price"].groupby([salesMergeWeatherMergeFuel["Temperature"],salesMergeWeatherMergeFuel["Weekly_Sales"]]).sum()
# print(Fuel_PriceGroupByWeekly_SalesAndTemperature)
# Fuel_PriceGroupByWeekly_SalesAndTemperatureDataFrame = Fuel_PriceGroupByWeekly_SalesAndTemperature.reset_index()
# Fuel_PriceGroupByWeekly_SalesAndTemperatureDataFrame.columns = ["Temperature","Weekly_Sales","Fuel_Price"]
# print(Fuel_PriceGroupByWeekly_SalesAndTemperatureDataFrame)
Fuel_PriceGroupByTemp = salesMergeWeatherMergeFuel["Fuel_Price"].groupby(salesMergeWeatherMergeFuel["Temperature"]).sum()
Fuel_PriceGroupByTempDataFrame = Fuel_PriceGroupByTemp.reset_index()
Fuel_PriceGroupByTempDataFrame.columns = ["Temperature","Fuel_Price"]

Weekly_SalesGroupByTempDataFrame['Weekly_Sales_norm'] = (Weekly_SalesGroupByTempDataFrame['Weekly_Sales'] - Weekly_SalesGroupByTempDataFrame['Weekly_Sales'].min()) / (Weekly_SalesGroupByTempDataFrame['Weekly_Sales'].max() - Weekly_SalesGroupByTempDataFrame['Weekly_Sales'].min())
Fuel_PriceGroupByTempDataFrame['Fuel_Price_norm'] = (Fuel_PriceGroupByTempDataFrame['Fuel_Price'] - Fuel_PriceGroupByTempDataFrame['Fuel_Price'].min()) / (Fuel_PriceGroupByTempDataFrame['Fuel_Price'].max() - Fuel_PriceGroupByTempDataFrame['Fuel_Price'].min())

print(Fuel_PriceGroupByTempDataFrame)

plt.figure(figsize=(fig_width, fig_height)).suptitle("line chart of relationship between weekly sales and weather weekly sales.")
plt.get_current_fig_manager().window.geometry('+0+0')
plt.get_current_fig_manager().set_window_title("relationship between weekly sales and weather weekly sales")
plt.subplot(3,3,1)
plt.plot(Fuel_PriceGroupByTempDataFrame["Fuel_Price"],Weekly_SalesGroupByTempDataFrame["Weekly_Sales"])
plt.xlabel("Fuel_Price According Temperature")
plt.ylabel("Weekly_Sales According Temperature")
plt.title("(weekly sales and Fuel_Price) according Temperature\n(Before Standardize Scale)")
plt.ticklabel_format(style='plain', axis='y')

#subplot(2,3,2) is leaved empty
plt.subplot(3,3,3)
plt.plot(Fuel_PriceGroupByTempDataFrame["Fuel_Price_norm"],Weekly_SalesGroupByTempDataFrame["Weekly_Sales_norm"])
plt.xlabel("Fuel_Price According Temperature")
plt.ylabel("Weekly_Sales According Temperature")
plt.title("(weekly_Sales and Fuel_Price) according Temperature\n(After Standardize Scale)")
plt.ticklabel_format(style='plain', axis='y')


plt.subplot(3,3,5)
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Fuel_PriceGroupByTempDataFrame["Fuel_Price"],color="blue",label="Fuel_Price")
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Weekly_SalesGroupByTempDataFrame["Weekly_Sales"],color="orange",label="Weekly_Sales")
plt.xlabel("Temperature")
plt.ylabel("Non Normalized(Fuel And Weekly_Sales")
plt.title("(weekly_Sales and Fuel_Price) according Temperature\n(Before Standardize Scale)")
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")


plt.subplot(3,3,7)
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Fuel_PriceGroupByTempDataFrame["Fuel_Price_norm"],color="blue",label="Fuel_Price")
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Weekly_SalesGroupByTempDataFrame["Weekly_Sales_norm"],color="orange",label="Weekly_Sales")
plt.xlabel("Temperature")
plt.ylabel("Normalized_Fuel And Normalized_Weekly_Sales")
plt.title("(weekly_Sales and Fuel_Price) according Temperature\n(After Standardize Scale)")
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")


plt.subplot(3,3,8)
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Weekly_SalesGroupByTempDataFrame["Weekly_Sales_norm"],color="orange",label="Weekly_Sales")
plt.xlabel("Temperature")
plt.ylabel("Weekly_Sales_norm")
plt.title("Temperature and Weekly_Sales\n(After Standardize Scale)")
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")

plt.subplot(3,3,9)
plt.plot(Weekly_SalesGroupByTempDataFrame["Temperature"],Fuel_PriceGroupByTempDataFrame["Fuel_Price_norm"],color="blue",label="Fuel_Price")
plt.xlabel("Temperature")
plt.ylabel("Fuel_Price_norm")
plt.title("Temperature and Fuel_Price\n(After Standardize Scale)")
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

salesMergeWeatherMergeFuel['Date'] = pd.to_datetime(salesMergeWeatherMergeFuel['Date'])
salesMergeWeatherMergeFuel['Year'] = salesMergeWeatherMergeFuel['Date'].dt.year
salesMergeWeatherMergeFuel['Month'] = salesMergeWeatherMergeFuel['Date'].dt.month
salesMergeWeatherMergeFuel['Day'] = salesMergeWeatherMergeFuel['Date'].dt.day

linear_regression_model_path = "linear_regression_model.joblib"
random_forest_model_path = "random_forest_model.joblib"

x = salesMergeWeatherMergeFuel.drop(['Date', 'Weekly_Sales'], axis=1)
y = salesMergeWeatherMergeFuel['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
if os.path.isfile(linear_regression_model_path):
    model1 = joblib.load(linear_regression_model_path)
    print("Linear Regression model loaded successfully!")
else:
    
    model1 = LinearRegression()
    model1.fit(X_train, y_train)
    joblib.dump(model1, linear_regression_model_path)
    print("Linear Regression model saved successfully!")
if os.path.isfile(random_forest_model_path):
    model2 = joblib.load(random_forest_model_path)
    print("Random Forest Regression model loaded successfully!")
else:
    model2 = RandomForestRegressor()
    model2.fit(X_train, y_train)
    joblib.dump(model2, random_forest_model_path)
    print("Random Forest Regression model saved successfully!")



predictions1 = model1.predict(X_test)
accuracy1 = r2_score(y_test, predictions1) * 100

predictions2 = model2.predict(X_test)
accuracy2 = r2_score(y_test, predictions2) * 100

print("Model 1 - Linear Regression:")
print("R-squared:", accuracy1)
print()

print("Model 2 - Random Forest Regression:")
print("R-squared:", accuracy2)
