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
    
INSERT INTO codesmell (name, description, type) VALUES 
    ('Unnecessary Iteration', 'Unnecessary Iteration description...', 'Generic'),
    ('NaN Equivalence Comparison Misused', 'NaN Equivalence Comparison Misused description...', 'Generic'),
    ('Chain Indexing', 'Chain Indexing description...', 'API-Specific'),
    ('Columns and DataType Not Explicitly Set', 'Columns and DataType Not Explicitly Set description...', 'Generic'),
    ('Empty Column Misinitialization', 'Empty Column Misinitialization description...', 'Generic'),
    ('Merge API Parameter Not Explicitly Set', 'Merge API Parameter Not Explicitly Set description...', 'Generic'),
    ('In-Place APIs Misused', 'In-Place APIs Misused description...', 'Generic'),
    ('Dataframe Conversion API Misused', 'Dataframe Conversion API Misused description...', 'API-Specific'),
    ('Matrix Multiplication API Misused', 'Matrix Multiplication API Misused description...', 'API-Specific'),
    ('No Scaling before Scaling-Sensitive Operation', 'No Scaling before Scaling-Sensitive Operation description...', 'Generic'),
    ('Hyperparameter Not Explicitly Set', 'Hyperparameter Not Explicitly Set description...', 'Generic'),
    ('Memory Not Freed', 'Memory Not Freed description...', 'Generic'),
    ('Deterministic Algorithm Option Not Used', 'Deterministic Algorithm Option Not Used description...', 'Generic'),
    ('Randomness Uncontrolled', 'Randomness Uncontrolled description...', 'Generic'),
    ('Missing the Mask of Invalid Value', 'Missing the Mask of Invalid Value description...', 'Generic'),
    ('Broadcasting Feature Not Used', 'Broadcasting Feature Not Used description...', 'Generic'),
    ('TensorArray Not Used', 'TensorArray Not Used description...', 'API-Specific'),
    ('Training / Evaluation Mode Improper Toggling', 'Training / Evaluation Mode Improper Toggling description...', 'Generic'),
    ('Pytorch Call Method Misused', 'Pytorch Call Method Misused description...', 'API-Specific'),
    ('Gradients Not Cleared before Backward Propagation', 'Gradients Not Cleared before Backward Propagation description...', 'API-Specific'),
    ('Data Leakage', 'Data Leakage description...', 'Generic'),
    ('Threshold-Dependent Validation', 'Threshold-Dependent Validation description...', 'Generic');