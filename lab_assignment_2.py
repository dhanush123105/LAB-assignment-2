import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


#  A1 
def a1_purchase_analysis(file_name):
    df = pd.read_excel(file_name)

    X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    Y = df['Payment (Rs)'].values

    rank = np.linalg.matrix_rank(X)
    X_pinv = np.linalg.pinv(X)
    cost = X_pinv.dot(Y)

    return X, Y, rank, cost


#  A2 
def a2_rich_poor_classification(Y):
    result = []
    count = 1
    for i in Y:
        if i > 200:
            result.append("c" + str(count) + " is rich")
        else:
            result.append("c" + str(count) + " is poor")
        count += 1
    return result


#  A3 
def find_mean(data):
    s = 0
    for i in data:
        s += i
    return s / len(data)


def find_variance(data):
    m = find_mean(data)
    s = 0
    for i in data:
        s += (i - m) ** 2
    return s / len(data)


def a3_irctc_analysis(file_name):
    df = pd.read_excel(file_name, sheet_name="IRCTC Stock Price")

    price = df.iloc[:, 3].dropna().values
    chg = df.iloc[:, 8].dropna().values

    mean_np = np.mean(price)
    var_np = np.var(price)

    mean_time = []
    var_time = []

    for i in range(10):
        start = time.time()
        find_mean(price)
        mean_time.append(time.time() - start)

        start = time.time()
        find_variance(price)
        var_time.append(time.time() - start)

    wed_price = df[df["Day"] == "Wed"].iloc[:, 3].dropna().values
    wed_mean = find_mean(wed_price)

    apr_price = df[df["Month"] == "Apr"].iloc[:, 3].dropna().values
    apr_mean = find_mean(apr_price)

    loss = list(filter(lambda x: x < 0, chg))
    loss_prob = len(loss) / len(chg)

    wed_profit = df[(df["Day"] == "Wed") & (df.iloc[:, 8] > 0)]
    profit_wed = len(wed_profit) / len(df)

    cond_prob = len(wed_profit) / len(df[df["Day"] == "Wed"])

    return mean_np, var_np, sum(mean_time)/10, sum(var_time)/10, \
           wed_mean, apr_mean, loss_prob, profit_wed, cond_prob, df


#  A4 
def a4_thyroid_exploration(file_name):
    df = pd.read_excel(file_name, sheet_name="thyroid0387_UCI")

    info = {}
    for col in df.columns:
        info[col] = {}
        info[col]["datatype"] = str(df[col].dtype)
        info[col]["missing"] = int(df[col].isna().sum())
        if df[col].dtype != "object":
            info[col]["mean"] = df[col].mean()
            info[col]["variance"] = df[col].var()
        else:
            info[col]["mean"] = None
            info[col]["variance"] = None

    return info, df


#  A5 
def a5_jaccard_smc(df):
    temp = df.copy()

    for col in temp.columns:
        if temp[col].dtype == "object":
            values = set(temp[col].dropna().unique())
            if values == {'t', 'f'} or values == {'f', 't'}:
                temp[col] = temp[col].map({'t': 1, 'f': 0})

    bin_cols = []
    for col in temp.columns:
        if temp[col].dtype != "object":
            uniq = set(temp[col].dropna().unique())
            if uniq.issubset({0, 1}):
                bin_cols.append(col)

    v1 = temp.loc[0, bin_cols].values
    v2 = temp.loc[1, bin_cols].values

    f11 = f10 = f01 = f00 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        else:
            f00 += 1

    jc = 0 if (f11 + f10 + f01) == 0 else f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)

    return jc, smc


#  A6 
def a6_cosine_similarity(df):
    num_df = df.select_dtypes(include=['int64', 'float64'])

    v1 = num_df.iloc[0].values
    v2 = num_df.iloc[1].values

    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        return 0
    return dot / norm


#  A7 
def a7_heatmap(df):
    data = df.iloc[:20].select_dtypes(include=['int64', 'float64'])
    sns.heatmap(data.corr(), annot=True)
    plt.show()


#  MAIN 
def main():
    file_name = "Lab Session Data.xlsx"

    # A1
    X, Y, rank, cost = a1_purchase_analysis(file_name)
    print("Matrix X")
    print(X)
    print("\nMatrix Y")
    print(Y)
    print("\nRank =", rank)
    print("Cost of products:", cost)

    # A2
    print("\nCustomer classification")
    for r in a2_rich_poor_classification(Y):
        print(r)

    # A3
    mean_np, var_np, mt, vt, wed_m, apr_m, loss_p, profit_wed, cond_p, df_stock = a3_irctc_analysis(file_name)
    print("\nMean:", mean_np)
    print("Variance:", var_np)
    print("Mean time:", mt)
    print("Variance time:", vt)
    print("Wednesday mean:", wed_m)
    print("April mean:", apr_m)
    print("Loss probability:", loss_p)
    print("Profit on Wednesday:", profit_wed)
    print("Conditional probability:", cond_p)

    plt.scatter(df_stock["Day"], df_stock.iloc[:, 8])
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.show()

    # A4
    info, df_thyroid = a4_thyroid_exploration(file_name)
    print("\nThyroid data summary")
    for col in info:
        print("\nAttribute:", col)
        print("Datatype :", info[col]["datatype"])
        print("Missing  :", info[col]["missing"])
        print("Mean     :", info[col]["mean"])
        print("Variance :", info[col]["variance"])

    # A5
    jc, smc = a5_jaccard_smc(df_thyroid)
    print("\nJaccard:", jc)
    print("SMC:", smc)

    # A6
    cos = a6_cosine_similarity(df_thyroid)
    print("Cosine similarity:", cos)

    # A7
    a7_heatmap(df_thyroid)


main()
