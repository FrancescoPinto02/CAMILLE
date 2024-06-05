INSERT INTO effect (name, description) VALUES
    ('Efficiency', 'Efficiency issues description...'),
    ('Error-prone', 'Error-prone issues description...'),
    ('Robustness', 'Robustness issues description...'),
    ('Reproducibility', 'Repoducibility issues description...'),
    ('Memory Issue', 'Memory Issue issues description...'),
    ('Readability', 'Readability issues description...');
    
    
INSERT INTO pipeline_stage (name, description) VALUES
    ('Data Cleaning', 'Data Cleaning description...'),
    ('Feature Engineering', 'Feature Engineering description...'),
    ('Model Training', 'Model Training description...'),
    ('Model Evaluation', 'Model Evaluation description...');
    
INSERT INTO codesmell (name, description, problems, solution, bad_example, good_example, type) VALUES
    ('Unnecessary Iteration',
     'The code smell known as ''Unnecessary Iteration'' emphasizes the importance of replacing loops with vectorized solutions, especially in data-intensive tasks common in machine learning.',
     'Unnecessary iteration poses a challenge due to its time-consuming nature and potential for increased code complexity. This can lead to slower execution times and reduced performance, particularly in data-intensive machine learning applications.',
     'The solution to ''Unnecessary Iteration'' is to adopt vectorized solutions instead of loops. Utilizing built-in methods like join and groupby in Pandas, along with APIs like tf.reduce_sum() in TensorFlow, can significantly improve program efficiency and reduce code complexity.',
     'import pandas as pd

# Sample DataFrame
data = {
    ''category'': [''A'', ''B'', ''A'', ''B'', ''C'', ''A''],
    ''value'': [10, 20, 30, 40, 50, 60]
}
df = pd.DataFrame(data)

# Unnecessary iteration
result = {}
for category in df[''category''].unique():
    total = 0
    for index, row in df.iterrows():
        if row[''category''] == category:
            total += row[''value'']
    result[category] = total

print(result)
',
     'import pandas as pd

# Sample DataFrame
data = {
    ''category'': [''A'', ''B'', ''A'', ''B'', ''C'', ''A''],
    ''value'': [10, 20, 30, 40, 50, 60]
}
df = pd.DataFrame(data)

# Vectorized solution using groupby
result = df.groupby(''category'')[''value''].sum().to_dict()

print(result)
',
     'Generic'),
    ('NaN Equivalence Comparison Misused',
     'The code smell known as ''NaN Equivalence Comparison Misused'' highlights the difference between NaN equivalence comparison and None comparison, particularly in libraries like NumPy and Pandas.',
     'Misusing NaN equivalence comparison can result in unintended behaviors due to the difference in how NaN values are treated compared to None. Developers may encounter unexpected results, particularly when comparing DataFrame elements with np.nan, as it always returns False, potentially leading to bugs in the code.',
     'The solution to ''NaN Equivalence Comparison Misused'' involves understanding the difference between NaN and None comparison and handling NaN values appropriately. Developers should be aware that np.nan == np.nan evaluates to False and utilize functions like np.isnan() for NaN checks instead of direct comparison.',
     'import numpy as np
import pandas as pd

# Sample DataFrame with NaN values
data = {
    ''column1'': [1, 2, np.nan, 4],
    ''column2'': [np.nan, 2, 3, 4]
}
df = pd.DataFrame(data)

# Incorrect NaN comparison
result = df.applymap(lambda x: x == np.nan)

print(result)
',
     'import numpy as np
import pandas as pd

# Sample DataFrame with NaN values
data = {
    ''column1'': [1, 2, np.nan, 4],
    ''column2'': [np.nan, 2, 3, 4]
}
df = pd.DataFrame(data)

# Correct NaN comparison using np.isnan
result = df.applymap(lambda x: np.isnan(x))

print(result)
',
     'Generic'),
    ('Chain Indexing',
     'The code smell known as ''Chain Indexing'' refers to the use of multiple sequential indexing operations in Pandas, such as df[''one''][''two''], which can result in inefficient code and potential errors.',
     'The problem with chain indexing in Pandas lies in its potential performance issues and error-prone nature. For instance, using df[''one''][''two''] triggers two separate indexing events, whereas df.loc[:, (''one'', ''two'')] accomplishes the same task with a single call, resulting in significant performance differences. Additionally, assigning values to the result of chain indexing can yield unpredictable outcomes due to Pandas'' ambiguity regarding whether it returns a view or a copy',
     'The solution to ''Chain Indexing'' is straightforward: developers using Pandas should refrain from employing chain indexing. Instead, they should opt for more efficient and reliable indexing methods, such as using .loc or .iloc for accessing DataFrame elements.',
     'import pandas as pd

# Sample DataFrame
data = {
    ''one'': [1, 2, 3, 4],
    ''two'': [5, 6, 7, 8],
    ''three'': [9, 10, 11, 12]
}
df = pd.DataFrame(data)

# Chain indexing to access elements in ''two'' column
result = df[''two''][df[''one''] > 2]

print(result)
',
     'import pandas as pd

# Sample DataFrame
data = {
    ''one'': [1, 2, 3, 4],
    ''two'': [5, 6, 7, 8],
    ''three'': [9, 10, 11, 12]
}
df = pd.DataFrame(data)

# Proper indexing using .loc
result = df.loc[df[''one''] > 2, ''two'']

print(result)
',
     'API-Specific'),
    ('Columns and DataType Not Explicitly Set',
     'The code smell known as ''Columns and DataType Not Explicitly Set'' highlights the importance of explicitly selecting columns and setting their data types when importing data, to avoid unexpected behavior in subsequent data processing steps.',
     'The problem with not explicitly setting columns and data types during data import is that it can lead to confusion and errors in the downstream data schema. When columns are not explicitly selected, developers may be unsure of the data structure. Similarly, if data types are not set explicitly, the default type conversion might silently pass unexpected inputs, causing errors later in the processing pipeline.',
     'To address ''Columns and DataType Not Explicitly Set'', it is recommended to explicitly specify the columns and their data types when importing data. This practice helps in maintaining a clear data schema and ensures that the data types are correctly assigned, thus preventing unexpected behavior and potential errors in downstream tasks',
     'import pandas as pd

# Import data from a CSV file without specifying columns and data types
df = pd.read_csv(''data.csv'')

# Print the DataFrame
print(df.info())
',
     'import pandas as pd

# Specify the columns to be imported and their data types
column_names = [''id'', ''name'', ''age'', ''salary'']
data_types = {
    ''id'': ''int64'',
    ''name'': ''object'',
    ''age'': ''int64'',
    ''salary'': ''float64''
}

# Import data from a CSV file with specified columns and data types
df = pd.read_csv(''data.csv'', usecols=column_names, dtype=data_types)

# Print the DataFrame
print(df.info())
',
     'Generic'),
    ('Empty Column Misinitialization',
     'The code smell known as ''Empty Column Misinitialization'' involves the incorrect practice of initializing new empty columns in Pandas DataFrames with zeros or empty strings rather than using NumPy''s NaN value.',
     'Initializing new empty columns with zeros or empty strings in Pandas can cause issues because it prevents the correct use of methods such as .isnull() or .notnull(). This can result in problems when trying to identify and handle missing data, potentially leading to errors in the data analysis process.',
     'The solution to ''Empty Column Misinitialization'' is to use NumPy''s NaN value when creating new empty columns in a Pandas DataFrame. This ensures that methods like .isnull() and .notnull() work correctly, allowing for proper identification and handling of missing data.',
     'import pandas as pd

# Sample DataFrame
data = {
    ''column1'': [1, 2, 3],
    ''column2'': [4, 5, 6]
}
df = pd.DataFrame(data)

# Incorrectly initializing new empty columns with zeros or empty strings
df[''new_column_zeros''] = 0
df[''new_column_empty_strings''] = ''''

# Print the DataFrame
print(df)

# Check for missing values
print(df.isnull())
',
     'import pandas as np
import pandas as pd

# Sample DataFrame
data = {
    ''column1'': [1, 2, 3],
    ''column2'': [4, 5, 6]
}
df = pd.DataFrame(data)

# Correctly initializing new empty columns with NumPy''s NaN
df[''new_column_nan''] = np.nan

# Print the DataFrame
print(df)

# Check for missing values
print(df.isnull())
',
     'Generic'),
    ('Merge API Parameter Not Explicitly Set',
     'The code smell you''re referring to is called ''Merge API Parameter Not Explicitly Set''. It involves the practice of not explicitly specifying key parameters such as ''on'', ''how'', and ''validate'' when performing merge operations using the df.merge() API in Pandas.',
     'Not explicitly setting parameters for df.merge() can cause significant issues. It makes the code harder to read and understand, and may lead to incorrect merges if assumptions about the data are wrong. For instance, if the merge keys are not unique and the ''validate'' parameter is not set, the merge might silently produce incorrect results.',
     'To address ''Merge API Parameter Not Explicitly Set'', developers should always explicitly set the ''on'', ''how'', and ''validate'' parameters when performing a merge with df.merge(). This approach ensures clarity in the merge logic, enhances code readability, and helps catch potential issues early, preventing silent errors in data merging.',
     'import pandas as pd

# Sample DataFrames
data1 = {
    ''id'': [1, 2, 3],
    ''value1'': [''A'', ''B'', ''C'']
}
df1 = pd.DataFrame(data1)

data2 = {
    ''id'': [1, 2, 4],
    ''value2'': [''D'', ''E'', ''F'']
}
df2 = pd.DataFrame(data2)

# Merge without explicitly setting parameters
merged_df = df1.merge(df2)

# Print the merged DataFrame
print(merged_df)
',
     'import pandas as pd

# Sample DataFrames
data1 = {
    ''id'': [1, 2, 3],
    ''value1'': [''A'', ''B'', ''C'']
}
df1 = pd.DataFrame(data1)

data2 = {
    ''id'': [1, 2, 4],
    ''value2'': [''D'', ''E'', ''F'']
}
df2 = pd.DataFrame(data2)

# Merge with explicitly setting parameters
merged_df = df1.merge(df2, on=''id'', how=''inner'', validate=''one_to_one'')

# Print the merged DataFrame
print(merged_df)
',
     'Generic'),
    ('In-Place APIs Misused',
     'The code smell known as ''In-Place APIs Misused'' highlights the incorrect use of in-place operations. It happens when developers do not assign the result of an operation to a variable or fail to set the in-place parameter, mistakenly believing the original data structure is modified directly.',
     'The problem with misusing in-place APIs is that it can lead to unexpected results where changes do not affect the original data structure. For example, in Pandas, failing to assign the result of df.dropna() to a variable or not setting the in-place parameter means the original DataFrame remains unchanged. This can cause confusion and errors in data manipulation.',
     'To address ''In-Place APIs Misused'', developers should either assign the result of the operation to a new variable or explicitly set the in-place parameter in the API. This practice ensures that the intended changes are applied to the data structure. Understanding the specific behavior of methods in libraries like Pandas and TensorFlow is crucial for accurate data manipulation.',
     'import pandas as pd

# Sample DataFrame
data = {
    ''A'': [1, 2, None, 4],
    ''B'': [5, None, 7, 8],
    ''C'': [None, 10, 11, 12]
}
df = pd.DataFrame(data)

# Misuse of in-place operation: Drop NaN values without assigning the result
df.dropna(inplace=True)

# Original DataFrame remains unchanged
print(df)
',
     'import pandas as pd

# Sample DataFrame
data = {
    ''A'': [1, 2, None, 4],
    ''B'': [5, None, 7, 8],
    ''C'': [None, 10, 11, 12]
}
df = pd.DataFrame(data)

# Proper use of in-place operation: Assign the result to a new variable
df_cleaned = df.dropna()

# Original DataFrame remains unchanged
print(df)

# The result is stored in a new variable
print(df_cleaned)
',
     'Generic'),
    ('Dataframe Conversion API Misused',
     'The code smell known as ''DataFrame Conversion API Misused'' highlights the incorrect use of the df.values() method for converting a Pandas DataFrame to a NumPy array, which can lead to inconsistencies. The recommended method is df.to_numpy().',
     'Using df.values() for DataFrame to NumPy array conversion has an inconsistency problem. It''s unclear whether df.values() will return the actual array, a transformed version, or a custom Pandas array. This uncertainty can lead to errors in data manipulation and analysis.',
     'The solution to ''DataFrame Conversion API Misused'' is to use df.to_numpy() instead of df.values() when converting a DataFrame to a NumPy array. The df.to_numpy() method provides a consistent and reliable conversion, ensuring predictable results.',
     'import pandas as pd

# Sample DataFrame
data = {
    ''A'': [1, 2, 3],
    ''B'': [4, 5, 6],
    ''C'': [7, 8, 9]
}
df = pd.DataFrame(data)

# Misuse of DataFrame conversion API: Using df.values()
numpy_array = df.values

# Print the NumPy array
print(numpy_array)
',
     'import pandas as pd

# Sample DataFrame
data = {
    ''A'': [1, 2, 3],
    ''B'': [4, 5, 6],
    ''C'': [7, 8, 9]
}
df = pd.DataFrame(data)

# Proper use of DataFrame conversion API: Using df.to_numpy()
numpy_array = df.to_numpy()

# Print the NumPy array
print(numpy_array)
',
     'API-Specific'),
    ('Matrix Multiplication API Misused',
     'The code smell you''re referring to is called ''Matrix Multiplication API Misused''. It involves using the np.dot() function for matrix multiplication in NumPy instead of the more semantically appropriate np.matmul() function.',
     'The problem with using np.dot() for two-dimensional matrix multiplication is that it can cause confusion due to its mathematical semantics. In mathematics, the dot product usually results in a scalar, not a matrix. Using np.dot() for matrix multiplication can mislead developers and result in code that''s harder to understand.',
     'To address ''Matrix Multiplication API Misused'', developers should prefer np.matmul() over np.dot() for two-dimensional matrix multiplication in NumPy. This practice ensures that the operation''s semantics are clear and consistent with mathematical conventions, reducing potential confusion.',
     'import numpy as np

# Sample matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Misuse of matrix multiplication API: Using np.dot()
result = np.dot(matrix1, matrix2)

# Print the result
print(result)
',
     'import numpy as np

# Sample matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Proper use of matrix multiplication API: Using np.matmul()
result = np.matmul(matrix1, matrix2)

# Print the result
print(result)
',
     'API-Specific'),
    ('No Scaling before Scaling-Sensitive Operation',
     'The code smell you''re referring to is called ''No Scaling before Scaling-Sensitive Operation''. It occurs when feature scaling is not applied before performing operations that are sensitive to the scale of the input data.',
     'Skipping feature scaling before scaling-sensitive operations can cause significant issues. For instance, in PCA, if features are not scaled, the variable with the larger scale will dominate the principal components, leading to incorrect conclusions. Similarly, SVM, SGD, and other algorithms may perform poorly or give misleading results without proper scaling.',
     'To address ''No Scaling before Scaling-Sensitive Operation'', always check and apply appropriate feature scaling techniques such as standardization or normalization before running scaling-sensitive algorithms. This practice helps in obtaining accurate results and prevents one feature from disproportionately influencing the outcome.',
     'from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Perform PCA without feature scaling
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print the transformed data
print(X_pca)
',
     'from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA with feature scaling
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print the transformed data
print(X_pca)
',
     'Generic'),
    ('Hyperparameter Not Explicitly Set',
     'The code smell known as ''Hyperparameter Not Explicitly Set'' highlights the practice of not explicitly setting hyperparameters for machine learning models, which can lead to suboptimal performance and issues with reproducibility.',
     'The problem with not explicitly setting hyperparameters is that the default values provided by machine learning libraries may not be optimal for your specific dataset or problem. This can result in suboptimal model performance, such as getting stuck in local optima. Additionally, default parameters can change between library versions, causing inconsistencies in results and making it difficult to replicate the model in a different environment or programming language.',
     'The solution to ''Hyperparameter Not Explicitly Set'' is to explicitly define and tune hyperparameters for your machine learning models. By doing so, you can optimize the model''s performance for your specific data and problem. Explicitly setting hyperparameters also enhances the reproducibility of your results, making it easier to replicate the model in different environments or programming languages.',
     'from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier without explicitly setting hyperparameters
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
',
     'from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier with explicitly set hyperparameters
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
',
     'Generic'),
    ('Memory Not Freed',
     'The code smell known as ''Memory Not Freed'' indicates a situation where memory resources are not appropriately managed during machine learning training.',
     'The problem with not freeing memory during machine learning training is that it can lead to memory exhaustion, causing the training process to fail. This is particularly problematic given the memory constraints typically imposed on machines.',
     'To address ''Memory Not Freed'', developers should ensure they use memory management APIs provided by machine learning libraries effectively. This includes using clear_session() in TensorFlow when creating models in a loop and utilizing .detach() in PyTorch to release tensors from the computational graph when appropriate. These practices help prevent memory exhaustion and ensure smoother training processes.',
     'import tensorflow as tf

# Loop to create and train multiple models
for _ in range(5):
    # Define and compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=''relu'', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation=''softmax'')
    ])
    model.compile(optimizer=''adam'',
                  loss=''sparse_categorical_crossentropy'',
                  metrics=[''accuracy''])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Clear the TensorFlow session to free up memory
    tf.keras.backend.clear_session()
',
     'import torch
import torch.nn as nn
import torch.optim as optim

# Loop to create and train multiple models
for _ in range(5):
    # Define the model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1)
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Release tensors from the computational graph
    outputs.detach()
',
     'Generic'),
    ('Deterministic Algorithm Option Not Used',
     'The code smell you''re referring to is called ''Deterministic Algorithm Option Not Used''. It occurs when deterministic algorithms, which can improve reproducibility, are not utilized during the development process.',
     'The problem with not using deterministic algorithms during development is that it can result in non-repeatable results, making debugging inconvenient. This lack of reproducibility can hinder the debugging process and lead to inefficiencies in resolving issues.',
     'To address ''Deterministic Algorithm Option Not Used'', developers should enable deterministic algorithm options, such as torch.use_deterministic_algorithms(True) in PyTorch, during the development phase to enhance reproducibility and simplify debugging. However, it''s essential to switch to performance-optimized options for deployment to minimize any performance impact.',
     'import torch

# Your PyTorch code for model training, evaluation, etc. goes here
',
     'import torch

# Enable deterministic algorithms for reproducibility during development
torch.use_deterministic_algorithms(True)

# Your PyTorch code for model training, evaluation, etc. goes here
',
     'Generic'),
    ('Randomness Uncontrolled',
     'The code smell known as ''Randomness Uncontrolled'' highlights the absence of explicitly setting random seeds in applications involving random procedures. ',
     'Failing to set random seeds can lead to various issues, such as unpredictable results and difficulty in reproducing experiments. Without fixed random seeds, algorithms relying on randomness may produce different outcomes each time they run, making it harder to debug and replicate results.',
     'To address ''Randomness Uncontrolled'', it is recommended to set global random seeds explicitly during the development process. This ensures reproducibility in libraries like Scikit-Learn, PyTorch, Numpy, and others. Additionally, specific components like DataLoader in PyTorch should be initialized with random seeds to ensure consistent data splitting and loading.',
     'import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Randomness is not controlled, leading to unpredictable results
data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, size=(100,))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Your further code for model training, evaluation, etc. goes here
',
     'import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Randomness is controlled, ensuring reproducibility
data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, size=(100,))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random_seed)

# Your further code for model training, evaluation, etc. goes here
',
     'Generic'),
    ('Missing the Mask of Invalid Value',
     'The code smell known as ''Missing the Mask of Invalid Value'' arises when invalid values, such as values approaching zero, are not masked or handled appropriately in operations like the log function.',
     'Failing to handle potential invalid values, such as values approaching zero, can lead to errors during the execution of operations like the log function. These errors are difficult to diagnose and may not provide clear indications of their source, prolonging the debugging process.',
     'The solution to ''Missing the Mask of Invalid Value'' involves checking for potential invalid values in operations like the log function and adding a mask to handle them appropriately. Utilizing functions like tf.clip_by_value() to ensure input values remain within a valid range can prevent errors and streamline the debugging process. By addressing this code smell proactively, developers can improve the robustness of their code and reduce debugging efforts.',
     'import tensorflow as tf

# Perform a log operation without handling potential invalid values
input_tensor = tf.constant([-0.1, 0.5, 1.0, 10.0])
result = tf.math.log(input_tensor)
print(result)
',
     'import tensorflow as tf

# Handle potential invalid values by adding a mask
input_tensor = tf.constant([-0.1, 0.5, 1.0, 10.0])
masked_input = tf.clip_by_value(input_tensor, 1e-8, 1e8)  # Clip input values to a valid range
result = tf.math.log(masked_input)
print(result)
',
     'Generic'),
    ('Broadcasting Feature Not Used',
     'The code smell you''re referring to is called ''Broadcasting Feature Not Used''. It occurs when the broadcasting feature available in deep learning libraries like PyTorch and TensorFlow is not utilized, leading to potential inefficiencies in memory usage.',
     'By not leveraging the broadcasting feature, deep learning code may consume more memory than necessary, especially when performing operations like tiling tensors. This inefficiency can hinder performance and scalability, particularly in memory-constrained environments.',
     'The solution to ''Broadcasting Feature Not Used'' involves embracing the broadcasting feature in deep learning code, even though it may introduce some trade-offs in terms of debugging. By leveraging broadcasting, developers can achieve more memory-efficient operations, enhancing the performance and scalability of their models',
     'import torch

# Create two tensors with different shapes
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([10, 20, 30])

# Perform element-wise multiplication without utilizing broadcasting
result = tensor1 * tensor2
print(result)
',
     'import torch

# Create two tensors with different shapes
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([10, 20, 30])

# Utilize broadcasting for element-wise multiplication
result = tensor1 * tensor2.unsqueeze(0)  # Unsqueeze tensor2 to match the shape of tensor1
print(result)
',
     'Generic'),
    ('TensorArray Not Used',
     'The code smell known as ''TensorArray Not Used'' indicates a missed opportunity to use tf.TensorArray() in TensorFlow 2 for scenarios where the value of the array needs to be modified iteratively within a loop.',
     'The problem with not using tf.TensorArray() in TensorFlow 2 is that attempting to modify the value of an array initialized with tf.constant() within a loop can lead to errors. While the issue can be addressed using low-level APIs like tf.while_loop(), this approach can be inefficient and result in the creation of numerous intermediate tensors.',
     'To address ''TensorArray Not Used'', developers should utilize tf.TensorArray() in TensorFlow 2 when the value of the array needs to change dynamically within a loop. By using tf.TensorArray(), developers can avoid errors and inefficiencies associated with modifying arrays initialized with tf.constant() within loops.',
     'import tensorflow as tf

# Initialize a tensor using tf.constant()
tensor = tf.constant([1, 2, 3, 4, 5])

# Attempt to modify the tensor value within a loop
for i in range(5):
    tensor[i] += 1  # Trying to modify a tensor initialized with tf.constant() will result in an error
',
     'import tensorflow as tf

# Initialize a TensorArray to dynamically modify values within a loop
tensor_array = tf.TensorArray(tf.int32, size=5)
initial_values = tf.constant([1, 2, 3, 4, 5])

# Write initial values to the TensorArray
for i in range(5):
    tensor_array = tensor_array.write(i, initial_values[i])

# Modify values of the TensorArray within a loop
for i in range(5):
    tensor_array = tensor_array.write(i, tensor_array.read(i) + 1)

# Read values from the TensorArray
result = tensor_array.stack()
print(result)
',
     'API-Specific'),
    ('Training / Evaluation Mode Improper Toggling',
     'The code smell known as ''Training / Evaluation Mode Improper Toggling'' arises when developers fail to switch back to the training mode after using the evaluation mode in deep learning code.',
     'The problem with improper toggling of training and evaluation modes is that it can lead to inconsistencies in the behavior of layers like Dropout. Forgetting to toggle back to the training mode after inference may cause Dropout layers to remain deactivated during subsequent training steps, potentially impacting the training results.',
     'The solution to ''Training / Evaluation Mode Improper Toggling'' involves correctly managing the toggling between training and evaluation modes in deep learning code. Developers should call the training mode in the appropriate place and remember to switch back to training mode after the inference step to ensure consistent behavior of layers like Dropout.',
     'import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Instantiate the model
model = Model()

# Set the model to evaluation mode
model.eval()

# Perform inference without toggling back to training mode
input_data = torch.randn(5, 10)
output = model(input_data)
',
     'import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Instantiate the model
model = Model()

# Set the model to evaluation mode
model.eval()

# Perform inference
input_data = torch.randn(5, 10)
output = model(input_data)

# Toggle back to training mode
model.train()
',
     'Generic'),
    ('Pytorch Call Method Misused',
     'The code smell known as ''PyTorch Call Method Misused'' arises when developers incorrectly use self.net.forward() instead of self.net() to forward input to the network in PyTorch.',
     'The problem with misusing the call method in PyTorch is that using self.net.forward() instead of self.net() may neglect certain functionalities such as registered hooks. This oversight can lead to unintended behavior or missed opportunities for customization during the forward pass.',
     'The solution to ''PyTorch Call Method Misused'' involves using self.net() instead of self.net.forward() in PyTorch for forwarding input to the network. By using self.net(), developers ensure that all relevant functionalities like registered hooks are accounted for during the forward pass, leading to more accurate and customizable model behavior.',
     'import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

# Instantiate the model
model = Model()

# Misuse of the call method
input_data = torch.randn(5, 10)
output = model.forward(input_data)
',
     'import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

# Instantiate the model
model = Model()

# Correct usage of the call method
input_data = torch.randn(5, 10)
output = model(input_data)
',
     'API-Specific'),
    ('Gradients Not Cleared before Backward Propagation',
     'The code smell you''re referring to is called ''Gradients Not Cleared before Backward Propagation''. It occurs when developers forget to use optimizer.zero_grad() before calling loss_fn.backward() in PyTorch',
     'Failure to clear gradients before backward propagation in PyTorch can result in the accumulation of gradients from previous iterations. This accumulation can lead to issues like gradient explosion, causing the training process to fail and hindering model convergence and performance.',
     'The solution to ''Gradients Not Cleared before Backward Propagation'' involves following the correct sequence of steps in PyTorch: optimizer.zero_grad(), loss_fn.backward(), and optimizer.step(). Developers should remember to use optimizer.zero_grad() before loss_fn.backward() to clear gradients and maintain the stability and effectiveness of the training process.',
     'import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Instantiate the model and define loss function and optimizer
model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Perform forward pass
input_data = torch.randn(5, 10)
output = model(input_data)

# Calculate loss
target = torch.tensor([1, 0, 1, 0, 1])  # Example target
loss = loss_fn(output, target)

# Backward propagation without clearing gradients
loss.backward()  # Gradients from previous iterations are accumulated
',
     'import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Instantiate the model and define loss function and optimizer
model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Perform forward pass
input_data = torch.randn(5, 10)
output = model(input_data)

# Calculate loss
target = torch.tensor([1, 0, 1, 0, 1])  # Example target
loss = loss_fn(output, target)

# Clear gradients before backward propagation
optimizer.zero_grad()

# Backward propagation
loss.backward()

# Update weights
optimizer.step()
',
     'API-Specific'),
    ('Data Leakage',
     'The code smell you''re referring to is called ''Data Leakage''. It occurs when the data used for training a machine learning model contains information that should not be available at the time of prediction.',
     'Data Leakage poses a significant challenge in machine learning as it can result in misleading experimental results and suboptimal real-world performance. Whether due to leaky predictors or leaky validation strategies, data leakage undermines the effectiveness and reliability of machine learning models.',
     'To address ''Data Leakage'', developers should carefully segregate training and validation data to prevent contamination. In Scikit-Learn, one effective approach is to use the Pipeline() API, which helps prevent data leakage by encapsulating preprocessing steps within the model pipeline.',
     'import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data.csv")

# Assume ''leaky_feature'' is a leaky predictor
X = data.drop(columns=[''target'', ''leaky_feature''])
y = data[''target'']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Assume ''leaky_feature'' is used during training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Assume ''leaky_feature'' is inadvertently used during validation
y_pred = model.predict(X_valid)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_pred)
print("Validation Accuracy:", accuracy)
',
     'from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data.csv")

# Separate features and target
X = data.drop(columns=[''target''])
y = data[''target'']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    (''scaler'', StandardScaler())
])

# Define model
model = RandomForestClassifier()

# Create pipeline
preprocessor = ColumnTransformer(transformers=[
    (''num'', numeric_transformer, X_train.columns)
])

pipeline = Pipeline(steps=[
    (''preprocessor'', preprocessor),
    (''model'', model)
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on validation set
y_pred = pipeline.predict(X_valid)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_pred)
print("Validation Accuracy:", accuracy)
',
     'Generic'),
    ('Threshold-Dependent Validation',
     'The code smell known as ''Threshold-Dependent Validation'' arises when model evaluation metrics are based on specific thresholds, which can lead to less interpretable results.',
     'The problem with ''Threshold-Dependent Validation'' is that relying on metrics tied to specific thresholds, such as F-measure, can be challenging to interpret and may not generalize well across different datasets or contexts. This can result in suboptimal model evaluation and decision-making.',
     'The solution to ''Threshold-Dependent Validation'' involves favoring threshold-independent metrics over threshold-dependent ones for model evaluation. By relying on metrics like Area Under the Curve (AUC), developers can ensure more reliable and interpretable assessments of model performance across different contexts and datasets.',
     'import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict probabilities and convert to binary predictions using a threshold
y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = np.where(y_pred_proba > threshold, 1, 0)

# Evaluate model using F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
',
     'from sklearn.metrics import roc_auc_score

# Evaluate model using AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC Score:", auc_roc)
',
     'Generic');