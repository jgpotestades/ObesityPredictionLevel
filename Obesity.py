# --- Prediction of Obesity Levels Based On Eating Habits and Physical Activites ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(data_filepath, preprocess=True):
    try:
        df = pd.read_csv(data_filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_filepath}")
        return None, None, None, None, None

    # --- Data Cleaning (Handle inconsistencies) ---
    for col in ['Sex', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'FAMHIS', 'CAEC', 'MTRANS', 'Classif']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
                if col == 'Sex':
                    df[col] = df[col].str.lower().replace({'male.': 'male', 'female.': 'female'})
                elif col == 'MTRANS':
                    df[col] = df[col].str.replace('_', ' ').str.lower()
                elif col == 'Classif':
                    df[col] = df[col].str.replace('_', ' ').str.lower()
                elif col in ['CALC', 'FAVC', 'SCC', 'SMOKE', 'FAMHIS', 'CAEC']:
                    df[col] = df[col].str.lower()

    label_encoder = LabelEncoder()
    categorical_features = ['Sex', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'FAMHIS', 'CAEC', 'MTRANS', 'Classif']

    for column in categorical_features:
        if column in df.columns:
            df[column] = label_encoder.fit_transform(df[column])
        else:
            print(f"Warning: Column '{column}' not found in CSV.")
            return None, None, None, None, None

    sex_label_encoder = LabelEncoder()
    if 'Sex' in df.columns:
        df['Sex'] = sex_label_encoder.fit_transform(df['Sex'])

    df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df.dropna(subset=['Height', 'Weight'], inplace=True)
    df['BMI'] = df['Weight'] / (df['Height']**2)

    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
    scaler = StandardScaler()

    if preprocess:
        try:
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
        except ValueError as e:
            print(f"Error during scaling: {e}")
            return None, None, None, None, None

    return df, scaler, numerical_features, label_encoder, sex_label_encoder

# --- Model Training and Evaluation ---
def train_and_evaluate_model(df, numerical_features, algorithm):
    X = df.drop('Classif', axis=1)
    y = df['Classif']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif algorithm == "KNN":
        model = KNeighborsClassifier()
    else:
        messagebox.showerror("Error", "Invalid algorithm selected.")
        return None, None, None, None

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return model, accuracy, report, cm
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during model training/evaluation: {e}")
        return None, None, None, None

# --- Plotting Functions ---
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def plot_feature_importance(model, X):  # For models with feature_importances_
    try:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.show()
    except AttributeError:
        messagebox.showerror("Error", "Model does not have feature importances attribute.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during plotting: {e}")

def no_feature_importance_plot(model):  # For models without feature_importances_
    messagebox.showinfo("Information", "Feature importance is not available for this model.")

# --- GUI (Tkinter) ---
root = tk.Tk()
root.title("Obesity Risk Assessment")

data_file = ""
df = None
scaler = None
numerical_features = None
model = None
label_encoder = None
sex_label_encoder = None
categorical_features = None  # Define it here, and it will be updated later

# --- Define functions FIRST ---
def browse_file():
    global data_file, df, scaler, numerical_features, label_encoder, sex_label_encoder, categorical_features
    filename = filedialog.askopenfilename(initialdir=".", title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if filename:
        data_file = filename
        data_label.config(text=f"Selected File: {data_file}")
        try:
            df, scaler, numerical_features, label_encoder, sex_label_encoder = load_and_preprocess_data(data_file)
            if df is None:
                messagebox.showerror("Error", "Failed to load or preprocess data. Check the file and data format.")
                return  # Exit early if data loading fails

            # --- Define categorical features based on the loaded data ---
            categorical_features = ['Sex', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'FAMHIS', 'CAEC', 'MTRANS', 'Classif']

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during data loading: {e}")

def analyze_and_train():
    global df, model, numerical_features, scaler, label_encoder, sex_label_encoder, categorical_features, rf_accuracy, knn_accuracy, rf_report, knn_report, rf_cm, knn_cm

    if df is None:
        messagebox.showerror("Error", "No data loaded. Please select a CSV file.")
        return

    try:
        # --- Preprocessing Selection ---
        preprocess = messagebox.askyesno("Preprocessing", "Apply preprocessing (scaling)?")

        df, scaler, numerical_features, label_encoder, sex_label_encoder = load_and_preprocess_data(data_file, preprocess)
        if df is None:
            messagebox.showerror("Error", "Failed to load or preprocess data.")
            return

        plot_correlation_matrix(df)

        def evaluate_algorithm(algorithm):
            model, accuracy, report, cm = train_and_evaluate_model(df, numerical_features, algorithm)
            if model is None:
                return None, None, None

            if algorithm == "Random Forest":
                plot_feature_importance(model, df.drop('Classif', axis=1))
            elif algorithm == "KNN":
                no_feature_importance_plot(model)

            return accuracy, report, cm

        rf_accuracy, rf_report, rf_cm = evaluate_algorithm("Random Forest")
        knn_accuracy, knn_report, knn_cm = evaluate_algorithm("KNN")

        if rf_accuracy is not None and knn_accuracy is not None:
            # --- Results Window (Now populated) ---
            def show_results():
                if root.winfo_exists():
                    results_window = tk.Toplevel(root)
                    results_window.title("Analysis Results")

                    comparison_label = tk.Label(results_window, text="")
                    if rf_accuracy > knn_accuracy:
                        comparison_label.config(text=f"Random Forest has better accuracy ({rf_accuracy:.4f} vs {knn_accuracy:.4f})")
                    elif knn_accuracy > rf_accuracy:
                        comparison_label.config(text=f"KNN has better accuracy ({knn_accuracy:.4f} vs {rf_accuracy:.4f})")
                    else:
                        comparison_label.config(text="Both algorithms have the same accuracy.")
                    comparison_label.pack()

                    # --- Random Forest Results ---
                    rf_frame = tk.LabelFrame(results_window, text="Random Forest")
                    rf_frame.pack(pady=10)

                    rf_accuracy_label = tk.Label(rf_frame, text=f"Accuracy: {rf_accuracy:.4f}")
                    rf_accuracy_label.pack()

                    rf_report_text = tk.Text(rf_frame, wrap=tk.WORD, height=10)
                    rf_report_text.insert(tk.END, rf_report)
                    rf_report_text.pack()

                    if rf_cm is not None:
                        plt.figure()
                        sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
                        plt.title("Random Forest Confusion Matrix")
                        plt.xlabel("Predicted Label")
                        plt.ylabel("True Label")
                        plt.show(block=False)

                    # --- KNN Results ---
                    knn_frame = tk.LabelFrame(results_window, text="KNN")
                    knn_frame.pack(pady=10)

                    knn_accuracy_label = tk.Label(knn_frame, text=f"Accuracy. {knn_accuracy:.4f}")
                    knn_accuracy_label.pack()

                    knn_report_text = tk.Text(knn_frame, wrap=tk.WORD, height=10)
                    knn_report_text.insert(tk.END, knn_report)
                    knn_report_text.pack()

                    if knn_cm is not None:
                        plt.figure()
                        sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues")
                        plt.title("KNN Confusion Matrix")
                        plt.xlabel("Predicted Label")
                        plt.ylabel("True Label")
                        plt.show(block=False)

                else:
                    messagebox.showerror("Error", "Main window has been closed.")

            results_button = tk.Button(root, text="Show Results", command=show_results)
            results_button.pack(pady=10)

            # --- Prediction Input (One Window) ---

            # --- Define predict_obesity function at the TOP LEVEL ---
            def predict_obesity():  # Now a global function
                try:
                    predict_window = tk.Toplevel(root)
                    predict_window.title("Prediction Input")

                    input_vars = {}
                    row_num = 0  # Initialize row number for prediction window

                    for feature, label_text in input_labels.items():
                        label = tk.Label(predict_window, text=label_text)
                        label.grid(row=row_num, column=0, padx=5, pady=5, sticky="w")

                        if feature == 'Sex':
                            var = tk.StringVar(predict_window)
                            var.set("Male")  # Default Value
                            dropdown = ttk.Combobox(predict_window, textvariable=var, values=["Male", "Female"], state='readonly')  # Fixed values
                            dropdown.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
                            input_vars[feature] = var
                        elif feature in ['Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Age']:
                            var = tk.DoubleVar(predict_window)
                            entry = tk.Entry(predict_window, textvariable=var)
                            entry.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
                            input_vars[feature] = var
                        else:  # Other categorical features
                            var = tk.StringVar(predict_window)
                            var.set("no")  # Default Value
                            categories = ["no", "yes"]  # Fixed values
                            dropdown = ttk.Combobox(predict_window, textvariable=var, values=categories, state='readonly')
                            dropdown.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
                            input_vars[feature] = var

                        row_num += 1

                    def predict_action():
                        global scaler, model, numerical_features, label_encoder, sex_label_encoder, categorical_features, results_label

                        input_data = {}
                        for feature in ['Age', 'Sex', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 'FAMHIS', 'FAF', 'TUE', 'CAEC', 'MTRANS']:
                            value = input_vars[feature].get()
                            input_data[feature] = value

                        input_df = pd.DataFrame([input_data])

                        # --- Correctly handle categorical features ---
                        for col in categorical_features:  # Now categorical_features is accessible
                            if col in input_df.columns and col != 'Classif':
                                if col == 'Sex':
                                    try:
                                        encoded_value = sex_label_encoder.transform([input_df[col].iloc[0]])[0]
                                        input_df[col] = encoded_value
                                    except ValueError:
                                        messagebox.showerror("Error", f"Invalid input for Sex. Please select a valid option.")
                                        return
                                else:
                                    try:
                                        encoded_value = label_encoder.transform([input_df[col].iloc[0]])[0]
                                        input_df[col] = encoded_value
                                    except ValueError:
                                        messagebox.showerror("Error", f"Invalid input for {col}. Please select a valid option.")
                                        return

                        input_df['Height'] = pd.to_numeric(input_df['Height'], errors='coerce')
                        input_df['Weight'] = pd.to_numeric(input_df['Weight'], errors='coerce')

                        if pd.isna(input_df['Height'][0]) or pd.isna(input_df['Weight'][0]):
                            messagebox.showerror("Error", "Invalid Height or Weight input for prediction.")
                            return

                        # --- Convert cm to meters BEFORE calculating BMI ---
                        input_df['Height'] = input_df['Height'] / 100  # Convert cm to meters
                        input_df['BMI'] = input_df['Weight'] / (input_df['Height'] ** 2)

                        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

                        try:
                            prediction = model.predict(input_df.drop('Classif', axis=1))
                            predicted_class = label_encoder.inverse_transform(prediction)[0]
                            results_label.config(text=f"Predicted Obesity Level: {predicted_class}")  # Update results label
                            predict_window.destroy()  # Close the prediction window
                        except Exception as e:
                            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

                    predict_button_inner = tk.Button(predict_window, text="Predict", command=predict_action)
                    predict_button_inner.grid(row=row_num + 1, column=0, columnspan=2, pady=10)  # Place below input fields

                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred during prediction window creation: {e}")

            predict_button = tk.Button(root, text="Predict Obesity Level", command=predict_obesity)
            predict_button.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during analysis: {e}")


# --- Create UI Elements (AFTER defining functions) ---
file_frame = tk.LabelFrame(root, text="1. Select Your Data File")
file_frame.pack(pady=10, padx=10, fill=tk.X)

browse_button = tk.Button(file_frame, text="Browse CSV File", command=browse_file)
browse_button.pack(pady=5)

data_label = tk.Label(file_frame, text="No file selected")
data_label.pack()

analysis_frame = tk.LabelFrame(root, text="2. Analyze and Train Model")
analysis_frame.pack(pady=10, padx=10, fill=tk.X)

analyze_button = tk.Button(analysis_frame, text="Start Analysis", command=analyze_and_train)
analyze_button.pack(pady=5)

prediction_frame = tk.LabelFrame(root, text="3. Predict Your Risk")
prediction_frame.pack(pady=10, padx=10, fill=tk.X)

input_labels = {  # Dictionary for easier management
    'Age': 'Age (Years)',
    'Sex': 'Sex',
    'Height': 'Height (cm)',  # Units provided
    'Weight': 'Weight (Kg)',
    'CALC': 'Consumption of Alcohol',  # More descriptive
    'FAVC': 'Frequent Consumption of High Caloric Food',
    'FCVC': 'Frequency of Consumption of Vegetables',
    'NCP': 'Number of Main Meals',
    'SCC': 'Consumption of Food Between Meals',
    'SMOKE': 'Smoke',
    'CH2O': 'Consumption of Water Daily',
    'FAMHIS': 'Family History of Overweight',
    'FAF': 'Physical Activity Frequency',
    'TUE': 'Time Using Electronic Devices',
    'CAEC': 'Consumption of Food Between Meals',
    'MTRANS': 'Transportation Method Used',
}

input_vars = {}
row_num = 0

for feature, label_text in input_labels.items():
    label = tk.Label(prediction_frame, text=label_text)
    label.grid(row=row_num, column=0, padx=5, pady=5, sticky="w")  # Sticky for alignment

    if feature == 'Sex':
        var = tk.StringVar(prediction_frame)
        var.set("Male")  # Default Value
        dropdown = ttk.Combobox(prediction_frame, textvariable=var, values=["Male", "Female"], state='readonly')  # Fixed values
        dropdown.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")  # Sticky for alignment
        input_vars[feature] = var
    elif feature in ['Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Age']:
        var = tk.DoubleVar(prediction_frame)
        entry = tk.Entry(prediction_frame, textvariable=var)
        entry.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")  # Sticky for alignment
        input_vars[feature] = var
    else:  # Other categorical features
        var = tk.StringVar(prediction_frame)
        var.set("no")  # Default Value
        categories = ["no", "yes"]  # Fixed values
        dropdown = ttk.Combobox(prediction_frame, textvariable=var, values=categories, state='readonly')
        dropdown.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")  # Sticky for alignment
        input_vars[feature] = var

    row_num += 1

#predict_button = tk.Button(prediction_frame, text="Calculate Risk", command=predict_obesity)  # Clearer button text
#predict_button.grid(row=row_num + 1, column=0, columnspan=2, pady=10)  # Centered button

results_label = tk.Label(prediction_frame, text="", font=("Arial", 14, "bold"))  # Larger, bolder font
results_label.grid(row=row_num + 2, column=0, columnspan=2, pady=(0, 10))  # Space above

root.mainloop()