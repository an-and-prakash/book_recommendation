import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st
import random

@st.cache_data
def load_data():
    books = pd.read_csv('data/BX-Books.csv', sep=';', encoding='latin-1')
    ratings = pd.read_csv('data/BX-Book-Ratings-Subset.csv', sep=';', encoding='latin-1')
    users = pd.read_csv('data/BX-Users.csv', sep=';', encoding='latin-1')
    return books, ratings, users

def create_user_item_matrix(ratings):
    return pd.pivot_table(
        ratings,
        index='User-ID',
        columns='ISBN',
        values='Book-Rating',
        fill_value=0
    )

def get_favorite_users(user_id, ratings, threshold=8):
    user_ratings = ratings[ratings['User-ID'] == user_id]
    highly_rated_books = user_ratings[user_ratings['Book-Rating'] >= threshold]['ISBN']
    favorite_users = ratings[ratings['ISBN'].isin(highly_rated_books) & 
                             (ratings['User-ID'] != user_id)]['User-ID'].unique()
    return favorite_users

def get_favorite_publishers(user_id, ratings, books, threshold=8):
    user_ratings = ratings[ratings['User-ID'] == user_id]
    highly_rated_books = user_ratings[user_ratings['Book-Rating'] >= threshold].merge(books, on='ISBN')
    favorite_publishers = highly_rated_books['Publisher'].value_counts().index.tolist()
    return favorite_publishers



def get_recommendations_with_favorites(user_id, ratings, books, n_neighbors=5, n_recommendations=10):
    user_item_matrix = create_user_item_matrix(ratings)
    scaler = StandardScaler()
    user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)
    
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    knn.fit(user_item_matrix_scaled)
    
    user_idx = user_item_matrix.index.get_loc(user_id)
    distances, neighbor_indices = knn.kneighbors([user_item_matrix_scaled[user_idx]])
    
    neighbors = user_item_matrix.index[neighbor_indices.flatten()]
    neighbor_ratings = ratings[ratings['User-ID'].isin(neighbors)]
    
    favorite_users = get_favorite_users(user_id, ratings)
    favorite_publishers = get_favorite_publishers(user_id, ratings, books)
    
    book_scores = neighbor_ratings.groupby('ISBN')['Book-Rating'].mean().reset_index()
    book_scores.columns = ['ISBN', 'knn_score']
    
    book_scores = book_scores.merge(books, on='ISBN', how='left')
    book_scores['publisher_score'] = book_scores['Publisher'].isin(favorite_publishers).astype(int) * 1.5
    book_scores['favorite_user_score'] = book_scores['ISBN'].isin(ratings[ratings['User-ID'].isin(favorite_users)]['ISBN']).astype(int) * 2
    
    book_scores['final_score'] = book_scores['knn_score'] + book_scores['publisher_score'] + book_scores['favorite_user_score']
    
    user_read_isbns = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
    recommendations = book_scores[~book_scores['ISBN'].isin(user_read_isbns)]
    
    top_recommendations = recommendations.nlargest(n_recommendations, 'final_score')
    return top_recommendations

def get_active_users(ratings, users, n_users=10):
    user_activity = ratings.groupby('User-ID').agg({
        'ISBN': 'count',
        'Book-Rating': 'mean'
    }).reset_index()
    
    user_activity.columns = ['User-ID', 'Books-Rated', 'Avg-Rating']
    
    active_users = user_activity[user_activity['Books-Rated'] >= 20]
    
    active_users = active_users.nlargest(n_users, 'Books-Rated')
    
    valid_users = active_users.merge(users, on='User-ID', how='inner')
    
    return valid_users[['User-ID', 'Books-Rated', 'Avg-Rating']]

def display_user_suggestions(ratings, users):
    st.sidebar.header('🎯 Suggested User IDs')
    active_users = get_active_users(ratings, users)
    st.sidebar.markdown("**Active Users for Testing:**")
    for _, user in active_users.iterrows():
        user_info = f"""
        🔹 **User ID**: {int(user['User-ID'])}  
        - **Books Rated**: {int(user['Books-Rated'])}  
        - **Avg Rating**: {user['Avg-Rating']:.1f}/10
        """
        st.sidebar.markdown(user_info)

def display_friend_choices(user_id, ratings, books):
    st.subheader("📚 Your Friend's Choices")
    friend_ids = st.text_input("Enter Friend's User IDs (comma-separated)").strip()
    
    if friend_ids:
        friend_ids = list(map(int, friend_ids.split(',')))
        friend_ratings = ratings[ratings['User-ID'].isin(friend_ids)]
        
        top_friend_books = friend_ratings.groupby('ISBN')['Book-Rating'].mean().reset_index()
        top_friend_books = top_friend_books.nlargest(10, 'Book-Rating')
        
        friend_books = top_friend_books.merge(books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Image-URL-L']], on='ISBN', how='left')
        
        # Display friend's top books in 3 per row layout
        cols = st.columns(3)
        for i, (index, row) in enumerate(friend_books.iterrows()):
            with cols[i % 3]:
                st.image(row['Image-URL-L'], width=150)
                st.write(f"**{row['Book-Title']}** by {row['Book-Author']}")
                st.write(f"Publisher: {row['Publisher']}")

def display_book_recommendations(recommendations):
    st.subheader("📚 Recommended Books for You:")
    
    # Display 3 books per row with enough spacing
    cols = st.columns(3)
    for i, (index, rec) in enumerate(recommendations.iterrows()):
        with cols[i % 3]:
            st.image(rec['Image-URL-L'], width=150)
            st.write(f"**{rec['Book-Title']}** by {rec['Book-Author']}")
            st.write(f"Publisher: {rec['Publisher']}")
            st.write(f"Score: {round(rec['final_score'], 2)}")

def main():
    st.title('📚 Enhanced Book Recommendation System')
    books, ratings, users = load_data()
    display_user_suggestions(ratings, users)
    
    user_id = st.number_input("Enter Your User ID:", min_value=1, max_value=users['User-ID'].max())
    
    if user_id in ratings['User-ID'].unique():
        st.success(f"User ID {user_id} is valid!")
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding your perfect books..."):
                recommendations = get_recommendations_with_favorites(user_id, ratings, books)
                if recommendations.empty:
                    st.warning("No recommendations found. Try rating more books!")
                else:
                    # Display recommendations with images and details
                    display_book_recommendations(recommendations)
                
                # Show friend's top-rated books
        display_friend_choices(user_id, ratings, books)
    else:
        st.error("Invalid User ID. Check the suggested IDs in the sidebar!")

if __name__ == '__main__':
    main()
