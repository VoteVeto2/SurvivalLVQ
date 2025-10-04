import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from itertools import combinations 

# LVQ class
class LVQ:
    def __init__(self, n_prototypes, learning_rate=0.01, decay_rate=0.95):
        """
        Initialize LVQ classifier
        
        Parameters:
        - n_prototypes: Number of prototypes per class
        - learning_rate: Initial learning rate
        - decay_rate: Learning rate decay factor
        """
        self.n_prototypes = n_prototypes
        self.initial_lr = learning_rate
        self.decay_rate = decay_rate
        
    def _initialize_prototypes(self, X, y):
        """
        Initialize prototypes using k-means++ strategy
        """
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        prototypes = []
        
        for class_label in np.unique(y):
            class_data = X[y == class_label]
            class_prototypes = self._kmeans_plus_plus(class_data, self.n_prototypes)
            prototypes.extend(class_prototypes)
        
        return np.array(prototypes)
    
    def _kmeans_plus_plus(self, X, n_centers):
        """
        K-means++ initialization for better prototype placement
        """
        centers = []
        
        # Choose first center randomly
        centers.append(X[np.random.randint(X.shape[0])])
        
        for _ in range(n_centers - 1):
            # Calculate distances to nearest center
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centers]) for x in X])
            # Choose next center with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers.append(X[j])
                    break
        
        return centers
    
    def _assign_labels(self, X, y):
        """
        Assign labels to prototypes based on nearest training examples
        """
        labels = []
        n_classes = len(np.unique(y))
        
        for class_label in np.unique(y):
            for _ in range(self.n_prototypes):
                labels.append(class_label)
        
        return np.array(labels)
        
    def fit(self, X, y, epochs=100):
        """
        Train the LVQ model
        
        Parameters:
        - X: Training features
        - y: Training labels
        - epochs: Number of training epochs
        """
        # Initialize prototypes using k-means++ strategy
        self.prototypes = self._initialize_prototypes(X, y)
        self.prototype_labels = self._assign_labels(X, y)
        
        # Store training history
        self.training_errors = []
        
        for epoch in range(epochs):
            lr = self.initial_lr * (self.decay_rate ** epoch)
            indices = np.random.permutation(len(X))
            epoch_errors = 0
            
            for idx in indices:
                # Winner-take-all: find nearest prototype
                distances = np.linalg.norm(self.prototypes - X[idx], axis=1)
                winner_idx = np.argmin(distances)
                
                # LVQ update rule
                if self.prototype_labels[winner_idx] == y[idx]:
                    # Correct classification: move prototype closer
                    self.prototypes[winner_idx] += lr * (X[idx] - self.prototypes[winner_idx])
                else:
                    # Incorrect classification: move prototype away
                    self.prototypes[winner_idx] -= lr * (X[idx] - self.prototypes[winner_idx])
                    epoch_errors += 1
            
            self.training_errors.append(epoch_errors / len(X))
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Error Rate: {epoch_errors/len(X):.4f}, LR: {lr:.6f}")
    
    def predict(self, X):
        """
        Predict labels for new data
        
        Parameters:
        - X: Test features
        
        Returns:
        - predictions: Predicted labels
        """
        predictions = []
        
        for x in X:
            # Find nearest prototype
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            winner_idx = np.argmin(distances)
            predictions.append(self.prototype_labels[winner_idx])
        
        return np.array(predictions)
    
    def get_prototypes(self):
        """
        Return current prototypes and their labels
        """
        return self.prototypes, self.prototype_labels