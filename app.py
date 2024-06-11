import pickle
import os
import zipfile

from flask import send_from_directory
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
import tensorflow as tf
from keras import layers, models
import os
import PIL
import io

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

model = LinearRegression()
modelForClassification = RandomForestClassifier(n_estimators=100, random_state=42)
encoded_data = {}
X = None
position_of_target=0
col_list = []
input_features = []
list1 = []
prediction = 0
current_directory = os.getcwd()
UPLOAD_FOLDER = 'uploaded_folders'


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def get_features(file, target_variable):
    global list1
    col_list = []
    data = file
    data.dropna(inplace=True)
    target_input = target_variable
    for col in data.columns:
        col_list.append(col.lower())
    target_variable = data.columns[col_list.index(target_input.lower())]
    input_features = [col for col in data.columns if col != target_variable]
    list1 = input_features.copy()
    return data[input_features], data[target_variable]


def load_and_clean_dataset(df, missing_values=['?', np.nan, '']):
    # Load the dataset
    numeric_columns = df.select_dtypes(include='number').columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    categorical_columns = df.select_dtypes(exclude='number').columns
    if not categorical_columns.empty:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df


def label_encode_features(df):
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(exclude='number').columns
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df


def min_max_scaling(df):
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def reduce_dimensionality_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df.select_dtypes(include='number'))
    # Convert NumPy array to DataFrame
    reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i + 1}' for i in range(n_components)])
    return reduced_df


def normalize_data(df):
    if isinstance(df, pd.DataFrame):
        df = min_max_scaling(df)
    elif isinstance(df, np.ndarray):
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
    return df


def log_transformation(data):
    if isinstance(data, pd.DataFrame):
        numeric_columns = data.select_dtypes(include='number').columns
        data[numeric_columns] = np.log1p(data[numeric_columns])
    elif isinstance(data, np.ndarray):
        data = np.log1p(data)
    else:
        raise ValueError("Unsupported data type. Please provide a pandas DataFrame or a NumPy array.")
    return data

#----------------------Image classification methods---------------------------
def load_and_clean_image_dataset(dataset_path):
    clean_dataset_path = dataset_path + "_clean"
    os.makedirs(clean_dataset_path, exist_ok=True)

    problematic_images = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            try:
                img = PIL.Image.open(os.path.join(root, file))
                img.verify()  # Verify image file integrity
            except (IOError, SyntaxError, PIL.UnidentifiedImageError) as e:
                print(f"Error loading image: {os.path.join(root, file)} - {e}")
                problematic_images.append(os.path.join(root, file))
            else:
                img.close()
                destination_path = os.path.join(clean_dataset_path, os.path.relpath(root, dataset_path), file)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                os.replace(os.path.join(root, file), destination_path)

    if problematic_images:
        print("Removing problematic images...")
        for image_path in problematic_images:
            os.remove(image_path)

    print("Dataset cleaned successfully.")
    return clean_dataset_path


# Function to identify classes automatically from the dataset folder structure
def identify_image_classes(dataset_path):
    classes = os.listdir(dataset_path)
    return classes

# Function to load and preprocess the dataset
# Function to load and preprocess the dataset
# Function to load and preprocess the dataset
def load_image_dataset(dataset_path, img_height, img_width, batch_size, validation_split=0.2):
    def safe_load_img(image_array):
        return image_array

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=safe_load_img,  # Apply safe_load_img function to each image
        validation_split=validation_split  # Splitting the data into training and validation sets
    )
    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Use training subset
    )
    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Use validation subset
    )
    return train_data, val_data, train_data.class_indices


# Function to create the CNN model
def create_image_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_image_classification_model(model, train_data, val_data, epochs=10):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    for epoch in range(epochs):
        print("Epoch", epoch+1, "/", epochs)
        try:
            model.fit(train_data, epochs=1, validation_data=val_data)
        except PIL.UnidentifiedImageError as e:
            print("Error loading image during training. Skipping this image.")
            print("Error:", e)
            continue

#-----------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cardsindex.html')
def next_page():
    return render_template('cardsindex.html')


def feature_selection(csv_file, target_column_name):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    data.dropna(inplace=True)
    num_columns = len(data.columns)
    if num_columns < 5:
        k = num_columns - 1
    elif 5 < num_columns <= 10:
        k = num_columns - 2
    elif 10 < num_columns <= 15:
        k = num_columns - 3
    elif 15 < num_columns <= 20:
        k = num_columns - 4
    else:
        k = num_columns - 6
    # Separate features and target variable
    X = data.drop(columns=[target_column_name])
    y = data[target_column_name]

    # Perform feature selection using SelectKBest with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get the names of selected features
    selected_features = X.columns[selected_indices]

    # Create a new DataFrame with selected features and target variable
    new_data = pd.concat([pd.DataFrame(X_new, columns=selected_features), y], axis=1)

    return new_data


@app.route('/regression.html', methods=['GET', 'POST'])
def regression():
    global model, prediction
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('regression.html', error='No selected file')

        try:
            target_variable = request.form['selectedFeature']

            new_data = feature_selection(file, target_variable)
            x, y = get_features(new_data, target_variable)
            prediction = y
            model.fit(x, y)
            pickle.dump(model, open('model.pkl', 'wb'))
            return redirect(url_for('show'))

        except Exception as e:
            return render_template('regression.html', error=str(e))

    return render_template('regression.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global model
    if request.method == 'POST':
        try:
            input_features = [float(request.form[f'{item}']) for item in list1]
            y_pred = model.predict([input_features])
            return render_template('predictionOfRegression.html', input_features=list1, y_pred=y_pred[0])
        except Exception as e:
            return render_template('predictionOfRegression.html', error=str(e))
    return render_template('predictionOfRegression.html', list=list1)


@app.route('/preprocessing.html', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('preprocessing.html', error='No selected file')

        df = pd.read_csv(file)
        selected_options = request.form.getlist('option')

        preprocessed_data = load_and_clean_dataset(df)

        for technique in selected_options:
            # if technique == 'One-Hot Encode Features' and not isinstance(preprocessed_data, pd.DataFrame):
            if technique == 'One-Hot Encode Features' and not isinstance(preprocessed_data, pd.DataFrame):
                print("Error: Cannot perform One-Hot Encoding on reduced dimensionality data. Skipping technique 2.")
                continue
            elif technique == 'Min-Max Scaling' and isinstance(preprocessed_data, np.ndarray):
                print("Error: Cannot perform Min-Max Scaling on reduced dimensionality data. Skipping technique 3.")
                continue
            elif technique == 'PCA':
                preprocessed_data = reduce_dimensionality_pca(preprocessed_data)
            elif technique == 'Normalize Data' and isinstance(preprocessed_data, np.ndarray):
                print("Error: Cannot perform Normalization on reduced dimensionality data. Skipping technique 5.")
                continue
            elif technique == 'Log Transformation' and not isinstance(preprocessed_data, pd.DataFrame):
                print("Error: Cannot perform Log Transformation on reduced dimensionality data. Skipping technique 6.")
                continue

            # if technique == 'One-Hot Encode Features':
            #     preprocessed_data = one_hot_encode_features(preprocessed_data)

            # if technique == 'One-Hot Encode Features':
            if technique == 'Label Encode Features':
                preprocessed_data = label_encode_features(preprocessed_data)
            elif technique == 'Min-Max Scaling':
                preprocessed_data = min_max_scaling(preprocessed_data)
            elif technique == 'Normalize Data':
                preprocessed_data = normalize_data(preprocessed_data)
            elif technique == 'Log Transformation':
                preprocessed_data = log_transformation(preprocessed_data)

        processed_data_html = preprocessed_data.to_html() if isinstance(preprocessed_data, pd.DataFrame) else str(
            preprocessed_data)
        processed_data_path = os.path.join(os.getcwd(), "processedData.csv")
        preprocessed_data.to_csv(processed_data_path, index=False)
        return render_template('preprocessingResult.html', processed_data=processed_data_html)

    return render_template('preprocessing.html')


@app.route('/download', methods=['GET'])
def download_file():
    filename = 'processedData.csv'
    return send_from_directory(os.getcwd(), filename, as_attachment=True)


@app.route('/show')
def show():
    return render_template('show.html')


@app.route('/visualize.html')
def visualize():
    return render_template('visualize.html')


@app.route('/classification')
def classification_file_upload():
    return render_template('classification_upload.html')


@app.route('/ClassificationInputs', methods=['GET', 'POST'])
def ClassificationInputs():
    global modelForClassification, encoded_data,X,position_of_target
    if request.method == 'POST':
        try:
            # Given data

            # new_data = ['0438010', 'Abandoned Property Accounts Auditor Trainee 1', 0, 0, 'Competitive', '05',
            #             'PUBLIC EMPLOYEE FEDERATION', 2, 'Professionals', 'T', 'Trainee', 'Approved', '02000',
            #             'Comptroller, Office of', '007/01/1998']
            new_data = []

            # Retrieve feature names
            feature_names = request.form.getlist('feature_names[]')

            # Loop through feature names and get corresponding values
            # for feature in feature_names:
            #     value = request.form.get(feature)
            #     # Trim and remove spaces
            #     value = value.strip()
            #     new_data.append(value)
            for feature in feature_names:
                value = request.form.get(feature)
                # Trim and remove spaces
                value = value.strip()
                # Check if the value is numeric
                try:
                    value = float(value)
                except ValueError:
                    pass  # If not numeric, keep it as string
                new_data.append(value)
            # new_data.insert(position_of_target, 0)
            print(new_data)

            # Map values to their encoded counterparts
            mapped_data = []
            for i, value in enumerate(new_data):
                if isinstance(value, str) and not value.isdigit():  # Check if value is a string and not a digit
                    encoded_value = None
                    for column, encodings in encoded_data.items():
                        if value in encodings:
                            encoded_value = encodings[value]
                            break
                    if encoded_value is None:
                        mean_value = X[X.columns[i]].mean()
                        mapped_data.append(mean_value)
                    else:
                        mapped_data.append(encoded_value)
                else:
                    mapped_data.append(float(value))
            # Print the prediction list

            input_data = np.array(mapped_data).reshape(1, -1)

            result = modelForClassification.predict(input_data)

            # Find the predicted grade in the encoded_data dictionary
            predicted_grade = None
            for key, value in encoded_data[position_of_target].items():
                if value == result[0]:
                    predicted_grade = key
                    break
            return render_template('Classification_prediction.html', input_features=list1, y_pred=predicted_grade)
        except Exception as e:
            return render_template('Classification_prediction.html', error=str(e))
    return render_template('Classification_prediction.html', input_features=list1)


def get_data_type_info(data):
    data_type_info = {}
    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical data
            data_type_info[column] = list(data[column].unique())
        else:  # Numeric data
            data_type_info[column] = "Numeric"
    return data_type_info

@app.route('/classification_train', methods=['POST'])
def classification_train():
    global modelForClassification, encoded_data,X,list1,position_of_target
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            print("inside empty file if")
            return render_template('classification_upload.html', error='No selected file')

        try:
            target_variable = request.form['selectedFeature']
            data = pd.read_csv(file)
            position_of_target = target_variable
            data.dropna(inplace=True)
            data_without_target = data.copy()

            # Remove the target_variable column from the copy
            if target_variable in data_without_target.columns:
                data_without_target.drop(columns=[target_variable], inplace=True)

            # Pass the modified data to get_data_type_info function
            list1 = get_data_type_info(data_without_target)
            # x, y = get_features(data, target_variable)
            label_encoders = {}
            for column in data.select_dtypes(include=['object']).columns:
                label_encoders[column] = LabelEncoder()
                data[column] = label_encoders[column].fit_transform(data[column])
                # Create mapping dictionary
                encoded_data[column] = dict(zip(label_encoders[column].classes_,
                                                label_encoders[column].transform(label_encoders[column].classes_)))
            # Step 3: Feature Scaling
            scaler = StandardScaler()
            X = data.drop(target_variable, axis=1)
            y = data[target_variable]
            X_scaled = scaler.fit_transform(X)
            modelForClassification.fit(X_scaled, y)
            return redirect(url_for('ClassificationInputs'))

        except Exception as e:
            return render_template('classification_upload.html', error=str(e))

    return render_template('classification_upload.html')

@app.route('/imageClassification')
def image_classification_zip_file_upload():
    return render_template('image_classification.html')

@app.route('/imageClassificationUpload')
def image_classification_upload():
    return render_template('image_classification_upload.html')

@app.route('/image_classification_train', methods=['POST'])
def image_classification_train():
    global class_indices, model
    print("Inside image classification train")
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            print("inside empty file if")
            return render_template('classification_upload.html', error='No selected file')

        if file and file.filename.endswith('.zip'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            print("File path:", file_path)

        extract_to = os.path.join(UPLOAD_FOLDER, os.path.splitext(file.filename)[0])
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)



        IMG_HEIGHT, IMG_WIDTH = 128, 128
        dataset_path = extract_to + '/' + extract_to[17:]

        # C:/Users/Shubhada/Downloads/classification
        num_classes = len(identify_image_classes(dataset_path))
        batch_size = 32
        print("Number of classes:", num_classes)

        clean_dataset_path = load_and_clean_image_dataset(dataset_path)
        print("Cleaned dataset:", clean_dataset_path)

        # Load and preprocess dataset, and generate validation data
        train_data, val_data, class_indices = load_image_dataset(clean_dataset_path, IMG_HEIGHT, IMG_WIDTH, batch_size)
        print(class_indices)
        # Create and compile the model
        model = create_image_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
        pickle.dump(model, open('model.pkl', 'wb'))
        # Train the model
        train_image_classification_model(model, train_data, val_data)


    return render_template('image_classification_show.html')


@app.route('/imageClassificationPrediction', methods=['POST'])
def image_classification_prediction():
    if 'file' not in request.files:
        return render_template('image_classification_upload.html', error='No file part in the request')

    file = request.files['file']
    print(file)
    if file.filename == '':
        return render_template('image_classification_upload.html', error='No selected file')

    try:
        IMG_HEIGHT, IMG_WIDTH = 128, 128
        print("Inside prediction try")
        class_names = {v: k for k, v in class_indices.items()}
        print("class names:", class_names)
        file_bytes = file.read()
        img = tf.keras.preprocessing.image.load_img(io.BytesIO(file_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
        print("image loaded")
        print("image loaded")
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        print("image to array")
        img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
        print("Create batch axis")
        predictions = model.predict(img_array)
        print("Prediction:", predictions)
        predicted_class = class_names[np.argmax(predictions)]
        print(predicted_class)
        return render_template('image_classification_prediction.html', output_class=predicted_class)
    except Exception as e:
        print("Inside prediction exception")
        return render_template('image_classification_upload.html', error=str(e))

@app.route('/upload', methods=['POST'])
def upload():
    # Save the uploaded file to the uploads directory
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            print("inside empty file if")
            return render_template('visualize.html', error='No selected file')

        try:
            target_variable = request.form['selectedFeature']
            df = pd.read_csv(file)

            # Generate histogram
            plt.hist(df[target_variable])
            plt.xlabel(target_variable)
            plt.ylabel('Frequency')
            plt.title('Histogram of ' + target_variable)

            # Save plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Encode the plot to base64 string
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plot_url = 'data:image/png;base64,' + plot_data

            # Generate bar plot
            plt.figure()
            df[target_variable].value_counts().plot(kind='bar')
            plt.xlabel(target_variable)
            plt.ylabel('Frequency')
            plt.title('Bar Plot')
            bar_buffer = BytesIO()
            plt.savefig(bar_buffer, format='png')
            bar_buffer.seek(0)
            bar_data = base64.b64encode(bar_buffer.getvalue()).decode('utf-8')
            bar_url = 'data:image/png;base64,' + bar_data

            # Generate box plot
            plt.figure()
            df.boxplot(column=target_variable)
            plt.title('Box Plot')
            box_buffer = BytesIO()
            plt.savefig(box_buffer, format='png')
            box_buffer.seek(0)
            box_data = base64.b64encode(box_buffer.getvalue()).decode('utf-8')
            box_url = 'data:image/png;base64,' + box_data
            #
            # Generate heatmap
            plt.figure()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            plt.title('Heatmap')
            heatmap_buffer = BytesIO()
            plt.savefig(heatmap_buffer, format='png')
            heatmap_buffer.seek(0)
            heatmap_data = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
            heatmap_url = 'data:image/png;base64,' + heatmap_data

            # Generate scatter plot
            plt.figure()
            plt.scatter(df[target_variable], df[df.columns[0]])
            plt.xlabel(target_variable)
            plt.ylabel(df.columns[0])
            plt.title('Scatter Plot')
            # Save scatter plot to a BytesIO object
            scatter_buffer = BytesIO()
            plt.savefig(scatter_buffer, format='png')
            scatter_buffer.seek(0)

            # Encode scatter plot to base64 string
            scatter_data = base64.b64encode(scatter_buffer.getvalue()).decode('utf-8')
            scatter_url = 'data:image/png;base64,' + scatter_data

            return render_template('visualizationResult.html',
                                   plot_url=plot_url, bar_url=bar_url, box_url=box_url, heatmap_url=heatmap_url,
                                   scatter_url=scatter_url)

        except Exception as e:
            return render_template('visualize.html', error=str(e))

    return render_template('visualize.html')


# if __name__ == '__main__':
#     app.run(debug=True)
