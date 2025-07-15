# Movie-Recommendation-System
Develop a recommendation engine using the MovieLens dataset to suggest relevant movies to users. Implement both collaborative filtering and content-based filtering, evaluate their performance, and deploy the model as an API with a simple frontend.

Final Report:
✨Assignment: AI-Powered Movie Recommendation System

📌Objective

The objective of this project was to build a movie recommendation system using the MovieLens 100k dataset. The system uses both collaborative filtering and content-based filtering techniques to generate 
And suggest relevant movies for users. Users can interact with the model via a simple frontend interface.

Dataset used the MovieLens dataset: ml-100k

🧭 Summary of the steps

Step 1: Data Collection & Pre-processing

Dataset Preparation
● Downloaded the MovieLens 100k dataset
● Loaded the dataset using Pandas. 


Data Pre-processing

● Handled missing values, duplicate entries, and outliers. 

● Extracted relevant features (e.g., movie ID, user ID, ratings, genres, and timestamps). 



Step 2: Exploratory Data Analysis (EDA)

● Analysed the distribution of ratings and user interactions.
 
● Visualized trends using Matplotlib/Seaborn (e.g., most-rated movies, rating distribution, genre preferences). 

● Identified patterns in user-movie interactions. 


Step 3: Model Development

● Built a Content-Based Filtering model using TF-IDF vectors of movie genres/titles.
● Implemented Collaborative Filtering using the Alternating Least Squares (ALS) algorithm from the implicit library.

● Trained the model on user-movie interactions (ratings). 


Step 4: Model Evaluation

● Compared the performance of collaborative filtering and content-based filtering techniques. 

● Used metrics like Root Mean Square Error (RMSE), Precision, Recall, or Mean Absolute Error (MAE) to evaluate model performance. 
● Used Mean Absolute Error (MAE) to evaluate prediction accuracy.
● Performed cross-validation on user-item interactions to test model generalization.


Step 5: Model Deployment
Deploy as an API

● Converted the recommendation model into a Flask/FastAPI backend API with endpoints for recommendations.

● Created an endpoint that takes user ID and returns a list of recommended movies. 

● Built a Streamlit-based frontend allowing users to input their user ID and get recommended movies.


📈  Key findings 
● Hybrid approach (combining both methods) provided the most balanced recommendations.
● Achieved a mean MAE of approximately 0.72 on validation data, which is good and competitive.


🔧 Tools, methods, or technologies 
Dataset
MovieLens 100k

Model Algorithms
ALS (implicit), TF-IDF (scikit-learn)

Evaluation
MAE, Cross-validation (train-test split)

Backend
Flask / FastAPI
python recommendation_flask.py
http://localhost:5000/recommend?user_id=1
Frontend 
Streamlit 
run streamlit_app.py
http://192.168.1.6:8501

Programming Language
Python

🧗 Challenges & Solutions

● Cold Start Problem: Collaborative filtering requires prior user interaction. Solved using content-based filtering
● Data sparsity: Ratings data is sparse; used matrix factorization with ALS to handle it.
● Deployment Compatibility: Ensured smooth communication between frontend and backend using CORS and RESTful API design.
● User Feedback: Created a responsive UI using Streamlit for users, enables easy interaction with the model.

👓 Conclusion 
The project successfully implemented and deployed an AI-powered recommendation system. 
By combining collaborative and content-based filtering, the system was able to suggest relevant movies to users. 


🎖 References
•	MovieLens Dataset: https://grouplens.org/datasets/movielens/
•	Implicit Library (ALS): https://github.com/benfred/implicit
•	Scikit-learn: https://scikit-learn.org/
•	Streamlit: https://streamlit.io/
•	Flask: https://flask.palletsprojects.com/


