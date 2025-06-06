import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import ttk
import requests
from io import BytesIO
import zipfile
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


movies = pd.read_csv('https://raw.githubusercontent.com/Williams-W/movie_ml/main/complete_movies.csv')
selected_columns = ['Name', 'Country', 'Date', 'Rate', 'Certificate', 'Pred_Certificate', 'Genre', 'Nudity', 'Frightening', 'Violence', 'Profanity', 'Alcohol', 'Summarized_Text']
movies = movies[selected_columns]
###########################################################################

# Select relevant columns and filter rows
movies_cluster2 = movies[['Nudity', 'Frightening', 'Violence', 'Profanity', 'Alcohol', 'Certificate']]
movies_cluster2 = movies_cluster2.query("Certificate != 'Not Rated' & Certificate != 'None'")
movies_cluster2 = movies_cluster2[~movies_cluster2[['Nudity', 'Frightening', 'Violence', 'Profanity', 'Alcohol']].eq('No Rate').all(axis=1)]

# Replace ordinal values with numerical values
ordinal_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
movies_cluster2.replace(ordinal_mapping, inplace=True)

# Fill missing values with 0
movies_cluster2 = movies_cluster2.fillna(0)

# Prepare features and target variable
X = movies_cluster2[['Nudity', 'Frightening', 'Violence', 'Profanity', 'Alcohol']]
y = movies_cluster2['Certificate']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [500],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [4]
}


# Grid search for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, verbose=10)
grid_search_rf.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search_rf.best_estimator_

def predict_certificate(nudity, frightening, violence, profanity, alcohol, model = best_rf_model):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[nudity, frightening, violence, profanity, alcohol]],
                              columns=['Nudity', 'Frightening', 'Violence', 'Profanity', 'Alcohol'])

    # Make predictions using the trained model
    predicted_certificate = model.predict(input_data)

    # Return the predicted certificate
    return predicted_certificate

###################################################################
# GitHub raw content URL of the ZIP file
zip_url = 'https://github.com/Williams-W/movie_ml/raw/main/app_sim_compressed.zip'

# Download the ZIP file
response = requests.get(zip_url)
zip_file = zipfile.ZipFile(BytesIO(response.content))

# Extract the CSV file from the ZIP archive
csv_filename = 'app_sim_compressed.csv'  # Adjust the path based on the structure inside the ZIP file
with zip_file.open(csv_filename) as csv_file:
    # Read the CSV file directly from the ZIP archive
    app_sim = pd.read_csv(csv_file)

def movie_recommender(movie, filtered):
    if filtered:
        # Assuming 'movies' is your DataFrame with a column 'Certificates'
        filtered_indices = movies.query("Certificate == 'R' | Certificate == 'X'").index
        app_sim_filtered = app_sim.copy()  # Make a copy to avoid modifying the original dataframe
        app_sim_filtered.loc[:, app_sim.columns[filtered_indices]] = 0
    else:
        app_sim_filtered = app_sim

    movie_title = movie['Name'].values[0].lower()
    movie_idx = movies[movies['Name'].str.lower() == movie_title].index[0]
    
    movie_similarities = app_sim_filtered.iloc[movie_idx].values
    similar_movie_idxs = np.argsort(-movie_similarities)[1:7]
    
    similar_movies = movies.iloc[similar_movie_idxs]['Name'].values

    recommendations = f"{', '.join(similar_movies).title()}"
    return recommendations


###################################################################

def main():
    global window
    window = tk.Tk()
    window.title("Wyatt's Movie System")
    window.geometry("1400x700+75+50")

    home_page(window)

    window.mainloop()

def home_page(window):
    global selections_frame
    selections_frame = tk.Frame(window, bg= "#f7f3df")
    selections_frame.pack(fill='both', expand=1)

    # Title Label at the top
    tk.Label(selections_frame, text="Wyatt's Movie System", font=("Courier", 60), pady=100, bg= "#f7f3df").pack()

    # Search Bar Entry and Button in the same row
    entry_frame = tk.Frame(selections_frame, pady=0, bg="#f7f3df")
    entry_frame.pack()

    tk.Label(entry_frame, text="Movie Name", font=("Courier", 30), bg= "#f7f3df", padx=45).pack(side=tk.LEFT)

    global e
    e = tk.StringVar()
    e_entry = tk.Entry(entry_frame, textvariable=e, width=20, bg="lightgray", font=("Comic Sans MS", 25))
    e_entry.bind("<Return>", (lambda event: search_movie()))
    e_entry.pack(side=tk.LEFT)
    e_entry.focus_set()

    search_button = tk.Button(entry_frame, text="Search", width=15, height=2, font=("Comic Sans MS", 12),
                              bg="#cccccc", fg="black", command=search_movie)
    search_button.pack(side=tk.LEFT, padx=90)

    # Check button to filter out mature movie recommendations
    global filter_var
    filter_var = tk.BooleanVar(value=True)
    filter_checkbox = tk.Checkbutton(selections_frame, text="Filter Out Mature Movie Recommendations",
                                     font=("Comic Sans", 15), variable=filter_var, bg="#f7f3df", pady=10, padx=10)
    filter_checkbox.pack()

    # Button to go to the game page
    game_button = tk.Button(selections_frame, text="Go to Model", width=15, height=2, font=("Comic Sans", 15),
                             bg="#cccccc", fg="Black", command=lambda: go_to_game(window))
    game_button.pack(pady=125)

def search_movie():
    movie_name = e.get()
    result_movie = movies[movies['Name'].str.contains(movie_name, case=False)]
    filtered = filter_var.get()
    if not result_movie.empty:
        display_movie_info(result_movie, filtered)
    else:
        print("Movie not found")

def display_movie_info(movie, filtered):
    # Destroy previous frames
    for widget in selections_frame.winfo_children():
        widget.destroy()

    # Display movie information in separate frames
    info_frame = tk.Frame(selections_frame, width=1400, height=700, pady=10, bg ="#f7f3df")
    info_frame.pack()

    # Display the title with the movie name
    title_label = tk.Label(info_frame, text=f"Movie Information for {movie['Name'].values[0].title()}",
                           font=("Cancun", 30), bg ="#f7f3df")
    title_label.pack()

    # Create a frame for the left and right columns
    columns_frame = tk.Frame(info_frame, width=1400, height=500, pady=10, bg ="#f7f3df")
    columns_frame.pack(side=tk.TOP, pady=10)

    # Create frames for the left and right columns within the grouped frame
    left_column_frame = tk.Frame(columns_frame, width=500, height=10, padx=50, bg ="#f7f3df")  # Adjusted width and padding
    left_column_frame.pack(side=tk.LEFT)

    right_column_frame = tk.Frame(columns_frame, width=300, height=10, padx=50, bg ="#f7f3df")  # Adjusted width and padding
    right_column_frame.pack(side=tk.LEFT)

    # Create a new frame for the additional column to the right of right_column_frame
    additional_column_frame = tk.Frame(columns_frame, width=300, height=10, padx=50, bg ="#f7f3df")  # Adjusted width and padding
    additional_column_frame.pack(side=tk.LEFT)

    # Exclude the last column
    for i, column in enumerate(movie.columns[1:-1], start=1):
        font_size = 15

        # Determine the target frame (left, right, or additional)
        target_frame = left_column_frame if i <= 6 else right_column_frame

        # Create a frame for the column
        column_frame = tk.Frame(target_frame, bg ="#f7f3df")
        column_frame.pack(side=tk.TOP, anchor="w")

        # Create a label for the column name
        column_name_label = tk.Label(column_frame, text=f"{column}:", font=("Courier", font_size), anchor="w", bg ="#f7f3df")
        column_name_label.pack(side=tk.LEFT)

        # Create a label for the column value with text wrap
        column_value_label = tk.Label(column_frame, text=str(movie[column].values[0]), font=("Courier", font_size), bg ="#f7f3df",
                                       wraplength=300)  # Adjust wrap length as needed
        column_value_label.pack(side=tk.LEFT)

    # Create a frame for the "Summary" label
    summary_label_frame = tk.Frame(info_frame, width=1400, height=30, bg ="#f7f3df")
    summary_label_frame.pack(side=tk.TOP, pady=5)

    # Create a label for the "Summary" text
    summary_label = tk.Label(summary_label_frame, text="Summary:", font=("Courier", 15), bg ="#f7f3df")
    summary_label.pack()

    # Create a frame for the last column
    last_column_frame = tk.Frame(info_frame, width=1400, height=70)
    last_column_frame.pack(side=tk.TOP, anchor="w", pady=5)

    # Create a label for the last column value with text wrap
    last_column_value_label = tk.Label(last_column_frame, text=str(movie[movie.columns[-1]].values[0]),
                                       font=("Courier", 13), wraplength=1300, anchor="w", justify="left", bg ="#f7f3df")
    last_column_value_label.pack()

    # Run the movie_recommender function
    recommended_movies = movie_recommender(movie, filtered)

    # Create a label in the additional_column_frame for the recommendations
    recommendation_label = tk.Label(additional_column_frame, text=f"Recommendations:\n{recommended_movies}",
                                    font=("Courier", 15), wraplength=300, bg ="#f7f3df")
    recommendation_label.pack(side=tk.LEFT)

    # Add a button to go back to the home page
    back_button = tk.Button(selections_frame, text="Back to Home", width=15, height=2,
                            bg="#cccccc", fg="Black", command=go_to_home)
    back_button.pack(side=tk.BOTTOM, pady=(0, 30))  # Adjust padding as needed


# Rest of your code remains unchanged

def go_to_home():
    selections_frame.destroy()
    home_page(window)

def go_to_predictions():
    selections_frame.destroy()
    game_page(window)

def go_to_summaries():
    game_frame.destroy()
    home_page(window)

def go_to_game(win):
    selections_frame.destroy()
    game_page(win)

def game_page(window):
    global game_frame
    game_frame = tk.Frame(window, bg="#f7f3df")
    game_frame.pack(fill='both', expand=1)

    tk.Label(game_frame, text="What Should it Be Rated - ML Model\n", font=("Courier", 30), bg="#f7f3df").pack()

    # Import the ttk module
    from tkinter import ttk

    maturity_ratings = ['Nudity', 'Frightening', 'Violence', 'Profanity', 'Alcohol']

    # Create and pack 5 dropdown boxes with text headings
    dropdown_boxes = []

    for i, rating in enumerate(maturity_ratings):
        # Create a label for the heading
        heading_label = tk.Label(game_frame, text=f"Parents Guide for {rating}:", font=("Helvetica", 12), pady=18, bg="#f7f3df")
        heading_label.pack()

        # Create and configure the dropdown box
        options = ["None", "Mild", "Moderate", "Severe"]
        dropdown = ttk.Combobox(game_frame, values=options, state="readonly")
        dropdown.current(0)  # Set default selection
        dropdown.pack()

        dropdown_boxes.append(dropdown)

    # Create a frame for buttons
    button_frame = tk.Frame(game_frame, bg="#f7f3df")
    button_frame.pack(side=tk.BOTTOM, pady=(10, 10))

    # Create a button to retrieve information and call the predict_certificate function
    predict_button = tk.Button(button_frame, text="Predict Certificate", width=20, height=2,
                               bg="#cccccc", fg="Black", command=lambda: predict_and_display(dropdown_boxes))
    predict_button.pack(side=tk.LEFT, padx=10)  # Adjust padding as needed

    # Create a label to display the prediction result
    result_label = tk.Label(game_frame, text="", font=("Helvetica", 20), pady=10, bg="#f7f3df")
    result_label.pack()

    def predict_and_display(dropdown_boxes):
        # Get selected values from the dropdown boxes
        selected_values = [box.get() for box in dropdown_boxes]

        # Convert selected values to numerical format (e.g., "None" to 0)
        ordinal_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
        selected_values_numeric = [ordinal_mapping[val] for val in selected_values]

        # Call the predict_certificate function (assuming it's defined globally)
        prediction = predict_certificate(*selected_values_numeric, best_rf_model)

        # Update the result label
        result_label.config(text=f"Predicted Certificate: {prediction[0]}", pady=35)

    # Button to return to the home page
    back_button = tk.Button(button_frame, text="Back to Home", width=20, height=2,
                            bg="#cccccc", fg="Black", command=go_to_summaries)
    back_button.pack(side=tk.RIGHT, padx=10)  # Adjust padding as needed

main()
