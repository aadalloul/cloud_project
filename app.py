# ===============================
# Cloud-Based Distributed Data Processing Platform
# Author: Alaa Dalloul
# Python 3, PySpark, Streamlit
# MongoDB Integration Included
# ===============================

import streamlit as st
import pandas as pd
import time
import io
import pdfplumber

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import col, isnan, split, avg, to_date

from pymongo import MongoClient
from urllib.parse import quote_plus

# --------------------------
# MongoDB connection
# --------------------------
# ضع بياناتك هنا
username = quote_plus("alaa")
password = quote_plus("a120142001@A")
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.weqkj6j.mongodb.net/cloud_project?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["cloud_project"]
collection = db["results"]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Cloud ML Platform", layout="wide")
st.title("☁️ Cloud-Based Distributed Data Processing Platform")
st.write("ارفع ملف بيانات (CSV, Excel, TXT أو PDF) ليتم تحليله باستخدام Apache Spark و ML")

uploaded_file = st.file_uploader("📂 اختر ملف Dataset", type=["csv", "xls", "xlsx", "txt", "pdf"])

# --------------------------
# User selects processing
# --------------------------
st.subheader("⚙️ اختر ما تريد تنفيذه")
stat_check = st.checkbox("Descriptive Statistics", True)
kmeans_check = st.checkbox("KMeans Clustering", True)
lr_check = st.checkbox("Linear Regression", True)
fp_check = st.checkbox("FP-Growth", True)
ts_check = st.checkbox("Time Series Aggregation", True)

nodes_list = st.multiselect("اختيار عدد الWorkers", [1,2,4,8], default=[1,2,4,8])

if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()
    df = None

    # --------------------------
    # قراءة الملفات
    # --------------------------
    try:
        if ext == "csv":
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except:
                df = pd.read_csv(uploaded_file, encoding="latin1")
        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif ext == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t", engine="python")
        elif ext == "pdf":
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            df = pd.DataFrame({"text": text.splitlines()})
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {e}")

    if df is not None:
        st.success("تم تحميل الملف بنجاح")

        # --------------------------
        # إنشاء Spark Session
        # --------------------------
        spark = SparkSession.builder \
            .appName("CloudML") \
            .master("local[*]") \
            .getOrCreate()
        spark_df = spark.createDataFrame(df)

        # --------------------------
        # Descriptive Statistics
        # --------------------------
        if stat_check:
            st.subheader("🔍 عينة من البيانات")
            st.dataframe(spark_df.limit(5).toPandas())

            st.subheader("📊 أنواع الأعمدة")
            st.dataframe(pd.DataFrame(spark_df.dtypes, columns=["Column", "Type"]))

            # Missing values %
            st.subheader("⚠️ القيم المفقودة")
            total = spark_df.count()
            missing = []
            for c in spark_df.columns:
                m = spark_df.filter(col(c).isNull() | isnan(col(c))).count()
                missing.append([c, (m / total) * 100])
            st.dataframe(pd.DataFrame(missing, columns=["Column", "Missing %"]))

            # Min, Max, Mean
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            stats_df = pd.DataFrame()
            if numeric_cols:
                stats_df["Column"] = numeric_cols
                stats_df["Min"] = [df[c].min() for c in numeric_cols]
                stats_df["Max"] = [df[c].max() for c in numeric_cols]
                stats_df["Mean"] = [df[c].mean() for c in numeric_cols]
            st.subheader("📈 إحصاءات رقمية")
            st.dataframe(stats_df)

            # Unique values
            st.subheader("🔢 عدد القيم الفريدة لكل عمود")
            unique_df = pd.DataFrame({"Column": df.columns, "Unique Values": [df[c].nunique() for c in df.columns]})
            st.dataframe(unique_df)

        # --------------------------
        # Machine Learning Jobs
        # --------------------------
        results = []
        base_time = {}

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # --- KMeans ---
        if kmeans_check and len(numeric_cols) >= 2:
            assembler = VectorAssembler(inputCols=numeric_cols[:2], outputCol="features")
            km_df = assembler.transform(spark_df)
            for n in nodes_list:
                start = time.time()
                model = KMeans(k=3, featuresCol="features").fit(km_df)
                t = time.time() - start
                results.append(["KMeans", n, t])
                if n == 1: base_time["KMeans"] = t

        # --- Linear Regression ---
        if lr_check and len(numeric_cols) >= 2:
            assembler = VectorAssembler(inputCols=[numeric_cols[0]], outputCol="features")
            lr_df = assembler.transform(spark_df)
            for n in nodes_list:
                start = time.time()
                model = LinearRegression(featuresCol="features", labelCol=numeric_cols[1]).fit(lr_df)
                t = time.time() - start
                results.append(["LinearRegression", n, t])
                if n == 1: base_time["LinearRegression"] = t

        # --- FP-Growth ---
        if fp_check and "text" in df.columns:
            fp_df = spark_df.withColumn("items", split(col("text"), " "))
            for n in nodes_list:
                start = time.time()
                try:
                    model = FPGrowth(itemsCol="items", minSupport=0.2).fit(fp_df)
                except:
                    pass
                t = time.time() - start
                results.append(["FPGrowth", n, t])
                if n == 1: base_time["FPGrowth"] = t

        # --- Time-Series Aggregation ---
        if ts_check:
            date_cols = [c for c in df.columns if "date" in c.lower()]
            if date_cols:
                date_col = date_cols[0]
                ts_df = spark_df.withColumn("date_only", to_date(col(date_col)))
                for n in nodes_list:
                    start = time.time()
                    agg_df = ts_df.groupBy("date_only").agg(*[avg(c).alias(f"{c}_avg") for c in numeric_cols])
                    _ = agg_df.collect()
                    t = time.time() - start
                    results.append(["TimeSeriesAgg", n, t])
                    if n == 1: base_time["TimeSeriesAgg"] = t

        # --------------------------
        # Results Table
        # --------------------------
        final = []
        for task, nodes, t in results:
            speedup = base_time.get(task, t) / t
            efficiency = speedup / nodes
            final.append([task, nodes, t, speedup, efficiency])

        results_df = pd.DataFrame(final, columns=["Task", "Nodes", "Time", "Speedup", "Efficiency"])
        st.subheader("🚀 نتائج الأداء")
        st.dataframe(results_df)

        # --------------------------
        # Save to MongoDB
        # --------------------------
        try:
            collection.insert_many(results_df.to_dict(orient="records"))
            st.success("تم حفظ النتائج في MongoDB")
        except Exception as e:
            st.error(f"خطأ عند حفظ النتائج في MongoDB: {e}")

        # --------------------------
        # Download CSV
        # --------------------------
        buffer = io.StringIO()
        results_df.to_csv(buffer, index=False)
        st.download_button("⬇️ تحميل النتائج CSV", buffer.getvalue(), "results.csv", "text/csv")

        spark.stop()
