import csv
import os
import pandas as pd


def combinedatacsv(directory = 'TH-data'):

    # list all file name in directory
    file_names = os.listdir(directory)

    file_names.sort()

    # record all dataframes
    dataframes = []

    # loop to each file
    for filename in file_names:
        
        file_path = os.path.join(directory,filename)

        # convert csv file to dataframe
        df = pd.read_csv(file_path)

        split_parts = filename.split("-")
        first_part = split_parts[0]

        # get the year of the data
        digits = "".join(filter(str.isdigit, first_part))

        # get quarter of the data
        second_part = split_parts[1].split(".")[0].replace("q","")

        # create the year column
        df["Year"] = digits

        # create Quarter column
        df["Quarter"] = second_part

        #append the dataframe 
        print("Complete Year: ",digits,"Quarter: ",second_part,"File: ",filename,"Shape: ",df.shape)
        dataframes.append(df)

    # combine dataframe together
    print('combine all dataframes')
    combined_dataframe = pd.concat(dataframes,ignore_index =True)

    return combined_dataframe

# clean the data
def datapreprocessing(df,columns):

    # Convert all columns to float
    for column in columns:
        if df[column].dtype == object:  # Check if column has string values

            #replace the , " to '' and if the string is side set to 0
            df[column] = df[column].apply(lambda x: float(str(x).replace(',', '').replace('"','')) if x != 'side' else 0)

    print('Complete Clean Data')

    # create Gross Profit column
    df['Gross Profit'] = df['Revenue'] - df['Expenses']

    # Gross Profit Margin
    df["ROE"] = df["Net Profit"] / df["Equity"] * 100
    df["ROE"] = df["ROE"].round(2)

    # Net Profit Margin
    df["ROA"] = df["Net Profit"] / df["Total Asset"] * 100
    df["ROA"] = df["ROA"].round(2)

    print('Complete create all feature')

    return df


columns=['Total Asset','Total Liabilities','Paid-up Cap','Equity','Revenue',
         'Expenses','EBITDA','EBIT','Net Profit','EPS','Operating Cash Flow',
         'Investing Cash Flow','Financing Cash Flow','Net Cash Flow','ROE',
         'ROA','Gross Profit Margin','Net Profit Margin','EBIT Margin','D/E',
         'Int. Coverage','Current Ratio','Quick Ratio','Fixed Asset Turnover',
         'Total Asset Turnover','Inventory Turnover','Average Sale Period',
         'Accounts Receivable Turnover','Average Collection Period',
         'Account Payable Turnover','Average Payment Period','Cash Cycle'
         ]

combined_dataframe = combinedatacsv(directory = 'TH-data')

df_use = datapreprocessing(combined_dataframe,columns=columns)

# load to csv file
df_use.to_csv("alldata.csv",index = False)
