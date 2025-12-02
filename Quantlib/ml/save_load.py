# Save and Load ML Models
import joblib
def save_model(model, filename):
    """Saves a machine learning model to a file using joblib.
    
    Args:
        model: The machine learning model to save.
        filename (str): The path to the file where the model will be saved.
    """
    joblib.dump(model, filename)    
    print(f"Model saved to {filename}")
    
def load_model(filename):
    """Loads a machine learning model from a file using joblib.
    
    Args:
        filename (str): The path to the file from which the model will be loaded.
        
    Returns:
        The loaded machine learning model.
    """
    print(f"Loading model from {filename}...")
    return joblib.load(filename)
