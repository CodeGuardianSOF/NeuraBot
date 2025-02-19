# ...existing code...

def train_ai_model(data, labels, model):
    """
    Train the AI model with the provided data and labels.
    
    Parameters:
    data (array-like): Training data
    labels (array-like): Target labels
    model (object): The AI model to be trained
    
    Returns:
    object: Trained model
    """
    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Example usage
if __name__ == "__main__":
    # ...existing code...
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    
    # Load example data
    iris = load_iris()
    data, labels = iris.data, iris.target
    
    # Initialize the model
    model = RandomForestClassifier()
    
    # Train the model
    trained_model = train_ai_model(data, labels, model)
    # ...existing code...
