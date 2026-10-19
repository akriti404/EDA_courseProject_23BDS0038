import pandas as pd
import numpy as np

dataframe1 = pd.DataFrame({
    "ProductID": range(1, 11),
    "NameofProd": ["Tshirt","Jeans","Sunglasses","Pepsi","TV","Laptop","Mobile","Headphones","Jacket","Watch"],
    "Price": [100,200,50,30,25000,17000,12000,7000,350,4000]
})
dataframe2 = pd.DataFrame({
    "ProductID": [1,3,5,8,10,11,13,14,16,19],
    "Company_Location": ["Bangalore","Delhi","Chennai","Kolkata","Pune","Delhi","Chennai","Patna","Hyderabad","Bangalore"]
})
dataframe3 = pd.DataFrame({
    "ProductID": range(11,21),
    "NameofProd": ["Tshirt","Jeans","Sunglasses","Pepsi","TV","Laptop","Mobile","Headphones","Jacket","Watch"],
    "Price": [100,200,50,30,25000,17000,12000,7000,350,4000]
})
print("DATAFRAME 1\n", dataframe1)
print("DATAFRAME 2\n", dataframe2)
print("DATAFRAME 3\n", dataframe3)
print("\nAPPEND ROWS (concat):\n", pd.concat([dataframe1, dataframe3], ignore_index=True))
print("\nCONCATENATE COLUMNS (illustrative, aligns by row index):\n",
      pd.concat([dataframe1, dataframe2[["Company_Location"]]], axis=1))
print("\nMERGE BY ProductID (inner merge):\n", dataframe1.merge(dataframe2, on="ProductID"))
print("\nINNER JOIN\n", dataframe1.merge(dataframe2, on="ProductID", how="inner"))
print("\nLEFT JOIN\n", dataframe1.merge(dataframe2, on="ProductID", how="left"))
print("\nRIGHT JOIN\n", dataframe1.merge(dataframe2, on="ProductID", how="right"))
print("\nFULL OUTER JOIN\n", dataframe1.merge(dataframe2, on="ProductID", how="outer"))

values = pd.Series([38,89,76,11,23,60,67,14,88,77], name="Akriti")
vmin, vmax = values.min(), values.max()
print("\nORIGINAL VALUES\n", values.to_list())
print("NORMALIZED [-3,3]\n", (((values-vmin)/(vmax-vmin))*6-3).to_string())
print("NORMALIZED [-1,1]\n", (((values-vmin)/(vmax-vmin))*2-1).to_string())
print("MIN-MAX [0,1]\n", ((values-vmin)/(vmax-vmin)).to_string())
print("Z-SCORE\n", ((values-values.mean())/values.std(ddof=1)).to_string())
# Decimal scaling convention: divide by 10**j so values fall below 1.
j = len(str(int(abs(values).max())))
print("DECIMAL SCALING\n", (values/(10**j)).to_string())
