import streamlit as st
import pandas as pd
import os
import joblib

from sklearn.preprocessing import MultiLabelBinarizer


# MultiLabel Binarizer for categorical columns
class CustomMultiLabelBinarizer(MultiLabelBinarizer):

    def __init__(self):
        super().__init__()
        self.classes = None

    def fit(self, y):
        return super().fit(y)

    def transform(self, y):
        return super().transform(y)

    def fit_transform(self, X, y):
        return super().fit_transform(X)


class Recommender:

    def __init__(self, recommender):
        self.recommender = recommender

    def predict(self, data):
        return self.recommender.predict(data)

    def get_k_recommendations(self, k, cluster):
        recommendations = data[data["cluster"] == cluster]
        if recommendations.shape[0] > 0:
            recommendations = recommendations.sample(k)
        return recommendations


def load_data():
    data_file = os.path.abspath("../data/final.csv")
    data = pd.read_csv(data_file, index_col=0)
    data.Storage = data.Storage.apply(lambda x: tuple(eval(x)))
    return data


parent_dir = "/".join(__file__.split("/")[:-2])
st.write(os.path.abspath(parent_dir))
model = joblib.load(os.path.join(parent_dir, "/model/recommender.obj"))

data = load_data()

storage_sizes = set()
for storage in data.Storage.unique():
    for sto in storage:
        storage_sizes.add(sto)

st.write("# Recommend Laptops")
features = {
    "Rating": st.number_input("Minimum Rating"),  # TODO: textbox
    "MRP": st.number_input("Minimum MRP"),  # TODO: textbox
    "Processor": st.selectbox("Processor", data.Processor.unique()),
    "RAM": st.selectbox("RAM", data.RAM.unique()),
    "OS": st.selectbox("Select OS", data.OS.unique()),
    "Storage": (st.selectbox("Storage", storage_sizes), ),
    "OS_arch": st.selectbox("OS Architecure", data.OS_arch.unique()),
    "RAM_Type": st.selectbox("Select RAM Type", data.RAM_Type.unique()),
}


def predict():
    test = [list(features.values())]
    pred = int(model.predict(test))
    recommendations = model.get_k_recommendations(5, pred)
    recommendations = st.dataframe(recommendations)


st.button("Estimate", on_click=predict)
