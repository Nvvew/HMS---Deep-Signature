import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import scikitplot as skplt
import matplotlib.pyplot as plt
from esig import tosig
from sigkernel import Signature
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from torchcde import cdeint
import torch

def prepare_data(X_train, X_test):
    """
    Prepare the data by scaling and converting to signatures.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        
    Returns:
        tuple: Scaled and signature-transformed training and testing features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_sig = tosig.stream2sig(X_train_scaled, depth=2)
    X_test_sig = tosig.stream2sig(X_test_scaled, depth=2)
    
    return X_train_scaled, X_test_scaled, X_train_sig, X_test_sig

def train_signature_svm(X_train_sig, X_test_sig, y_train, y_test):
    """
    Train an SVM model using signature kernels.
    
    Args:
        X_train_sig (numpy.ndarray): Signature-transformed training features.
        X_test_sig (numpy.ndarray): Signature-transformed testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        
    Returns:
        float: Accuracy of the trained model on the test set.
    """
    signature_kernel = Signature()
    svm_model = SVC(kernel=signature_kernel.fit_transform)
    svm_model.fit(X_train_sig, y_train)
    
    y_pred = svm_model.predict(X_test_sig)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Signature SVM Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.show()
    
    return accuracy

def train_signature_random_forest(X_train_sig, X_test_sig, y_train, y_test):
    """
    Train a Random Forest model using signatures.
    
    Args:
        X_train_sig (numpy.ndarray): Signature-transformed training features.
        X_test_sig (numpy.ndarray): Signature-transformed testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        
    Returns:
        float: Accuracy of the trained model on the test set.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_sig, y_train)
    
    y_pred = rf_model.predict(X_test_sig)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Signature Random Forest Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.show()
    
    return accuracy

def train_neural_cde(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train a Neural CDE model.
    
    Args:
        X_train_scaled (numpy.ndarray): Scaled training features.
        X_test_scaled (numpy.ndarray): Scaled testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        
    Returns:
        float: Accuracy of the trained model on the test set.
    """
    class CDEFunc(torch.nn.Module):
        def __init__(self, input_channels, hidden_channels, output_channels):
            super(CDEFunc, self).__init__()
            self.input_channels = input_channels
            self.hidden_channels = hidden_channels
            self.output_channels = output_channels
            
            self.linear1 = torch.nn.Linear(hidden_channels, 128)
            self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)
        
        def forward(self, t, z):
            z = self.linear1(z)
            z = torch.relu(z)
            z = self.linear2(z)
            z = z.view(z.size(0), self.hidden_channels, self.input_channels)
            return z
    
    input_channels = X_train_scaled.shape[1]
    hidden_channels = 32
    output_channels = len(y_train.unique())
    
    cde_func = CDEFunc(input_channels, hidden_channels, output_channels)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    coeffs = cdeint(cde_func, X_train_tensor, torch.tensor([0.0, 1.0]))
    outputs = coeffs[:, -1, :].squeeze()
    loss = torch.nn.functional.cross_entropy(outputs, y_train_tensor)
    
    optimizer = torch.optim.Adam(cde_func.parameters(), lr=0.001)
    
    num_epochs = 50
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        coeffs = cdeint(cde_func, X_train_tensor, torch.tensor([0.0, 1.0]))
        outputs = coeffs[:, -1, :].squeeze()
        loss = torch.nn.functional.cross_entropy(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    with torch.no_grad():
        coeffs = cdeint(cde_func, X_test_tensor, torch.tensor([0.0, 1.0]))
        outputs = coeffs[:, -1, :].squeeze()
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Neural CDE Accuracy: {accuracy}")
        print("Classification Report:\n", classification_report(y_test_tensor.numpy(), predicted.numpy()))
        print("Confusion Matrix:\n", confusion_matrix(y_test_tensor.numpy(), predicted.numpy()))
        skplt.metrics.plot_confusion_matrix(y_test_tensor.numpy(), predicted.numpy(), normalize=True)
        plt.show()
    
    return accuracy

def main():
    """
    Main function to execute the model training process.
    """
    combined_df = pd.read_parquet('combined_data.parquet')
    X = combined_df.drop('Brain Activity Numerical', axis=1)
    y = combined_df['Brain Activity Numerical']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    X_train_scaled, X_test_scaled, X_train_sig, X_test_sig = prepare_data(X_train, X_test)
    
    svm_accuracy = train_signature_svm(X_train_sig, X_test_sig, y_train, y_test)
    rf_accuracy = train_signature_random_forest(X_train_sig, X_test_sig, y_train, y_test)
    ncde_accuracy = train_neural_cde(X_train_scaled, X_test_scaled, y_train, y_test)
    
    accuracies = {
        'Signature SVM': svm_accuracy,
        'Signature Random Forest': rf_accuracy,
        'Neural CDE': ncde_accuracy
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()