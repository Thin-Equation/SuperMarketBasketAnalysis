import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set
from pyspark.ml.fpm import FPGrowth

class MarketBasketAnalysis:
    def __init__(self, spark_app_name, warehouse_dir, executor_memory, driver_memory, data_path):
        self.spark = SparkSession.builder \
            .appName(spark_app_name) \
            .config("spark.sql.warehouse.dir", warehouse_dir) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.driver.memory", driver_memory) \
            .getOrCreate()
        self.data_path = data_path
        self.df_spark = None
        self.df_transactions = None
        self.model = None

    def load_data(self):
        df_pandas = pd.read_excel(self.data_path)
        self.df_spark = self.spark.createDataFrame(df_pandas)
        self.df_spark.show(5)

    def clean_data(self):
        self.df_spark = self.df_spark.dropna(subset=["CustomerID", "Description"])
        self.df_spark = self.df_spark.dropDuplicates()
        self.df_spark = self.df_spark.filter(self.df_spark["Quantity"] > 0)
        self.df_spark.show(5)

    def prepare_transactions(self):
        self.df_transactions = self.df_spark.groupBy("InvoiceNo").agg(collect_set("Description").alias("items"))
        self.df_transactions.show(5)

    def apply_fp_growth(self, min_support=0.02, min_confidence=0.2):
        fp_growth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
        self.model = fp_growth.fit(self.df_transactions)
        self.model.freqItemsets.show(5)
        self.model.associationRules.show(5)

    def visualize_frequent_itemsets(self):
        freq_itemsets = self.model.freqItemsets.toPandas()
        plt.figure(figsize=(10, 6))
        freq_itemsets = freq_itemsets.sort_values(by="freq", ascending=False).head(10)
        plt.barh(freq_itemsets['items'].astype(str), freq_itemsets['freq'], color='skyblue')
        plt.xlabel('Frequency')
        plt.ylabel('Itemsets')
        plt.title('Top 10 Frequent Itemsets')
        plt.show()

    def recommend_products(self, purchased_items, n_recommendations=3):
        rules_pandas = self.model.associationRules.toPandas()
        recommendations = rules_pandas[rules_pandas['antecedent'].apply(lambda x: all(item in purchased_items for item in x))]
        recommendations = recommendations.sort_values(by='confidence', ascending=False).head(n_recommendations)
        return recommendations['consequent'].tolist()

# Example usage
if __name__ == "__main__":
    mba = MarketBasketAnalysis(
        spark_app_name="Market Basket Analysis",
        warehouse_dir="/tmp",
        executor_memory="2g",
        driver_memory="2g",
        data_path="/content/Online Retail.xlsx"
    )
    mba.load_data()
    mba.clean_data()
    mba.prepare_transactions()
    mba.apply_fp_growth()
    mba.visualize_frequent_itemsets()
    purchased_items = ["LUNCH BAG RED RETROSPOT"]
    recommended = mba.recommend_products(purchased_items)
    print("Recommended Products: ", recommended)