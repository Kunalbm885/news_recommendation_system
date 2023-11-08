import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Loading dataset
@st.cache_data
def load_data():
    data_df = pd.read_csv('NewsCategorizer.csv')  # Replace 'news_dataset.csv' with your dataset file
    return data_df


data_new = load_data()
data_news = pd.DataFrame()

#Selecting 1000 rows for each cate..... Sirf 1000 dega!!!!
for category, group in data_new.groupby('category'):
    sampled_group = group.sample(n=1000, random_state=42)  # Adjust karsakte N but nahi no need to get extra data 
    data_news = pd.concat([data_news, sampled_group])

# Reset the index of the data_news
data_news.reset_index(drop=True, inplace=True)





tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data_news['headline'] + " " + data_news['short_description'])

# Calculate cosine similarity between articles
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



def recommend_news(selected_category, num_recommendations=20):

    category_mask = data_news['category'] == selected_category
    category_indices = data_news.index[category_mask].tolist()
    similarity_scores = cosine_sim[category_indices, :].mean(axis=0)  # Calculate the mean similarity scores


    top_indices = np.argsort(similarity_scores)[::-1][:num_recommendations]
    return data_news.iloc[top_indices]





#APP CHALUUUUUUU .....
st.title("News Recommendation System")




search_query = st.text_input("Search News by Keyword")

if search_query:

    filtered_news = data_news[data_news['headline'].str.contains(search_query, case=False, regex=False)]

    #need 10 news bass!!!!
    similar_headlines = filtered_news['headline'].unique()[:10]
    selected_news = st.selectbox("Select a similar news headline:", similar_headlines)

    if selected_news:
        # News Dikhao ,Puchta hai Bharat !!
        selected_news_details = filtered_news[filtered_news['headline'] == selected_news]
        # st.write(f"Selected News Details:")
        st.write(f"{selected_news_details['headline'].values[0]}")
        st.write(f"Category: {selected_news_details['category'].values[0]}")
        st.write(f"{selected_news_details['short_description'].values[0]}")
        st.write(f"URL: {selected_news_details['links'].values[0]}")






# Sidebar with category
category_list = data_news['category'].unique()
selected_category = st.sidebar.selectbox("Select a category:", category_list)


st.markdown("---")
if selected_category:

    st.markdown(
        f'<h3>Recommended News in {selected_category} category:</h3>',
        unsafe_allow_html = True
    )

    recommendations = recommend_news(selected_category)


    for idx, (index, row) in enumerate(recommendations.iterrows(), start=1):
        st.markdown(
            f'<div style="border-radius: 10px; background-color: rgb(38 39 48); padding: 10px 25px; margin: 10px;">'
            f'<h3>{idx}. ) {row["headline"]}</h3>'
            # f'<p><strong>Category:</strong> {row["category"]}</p>' Mujhe nahi dikhana category yaha, kuch recommendation issssue hua tohhh ??? risk nahi leneka hai !!!!
            f'<p><strong>Description:</strong> {row["short_description"]}</p>'
            f'<p><strong>URL:</strong> <a href="{row["links"]}">{row["links"]}</a></p>'
            f'</div>',
            unsafe_allow_html=True
        )




