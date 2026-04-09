"""
Model wrappers for computing SHAP values across different ML frameworks.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import lightgbm as lgb
from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the model."""
        pass
    
    @abstractmethod
    def compute_shap(self, X_test, task="binary"):
        """
        Compute SHAP values for test data.
        
        Returns
        -------
        shap_values : np.ndarray
            For binary/regression: shape (n_samples, n_features)
            For multiclass: shape (n_samples, n_classes, n_features)
        """
        pass


class XGBoostWrapper(ModelWrapper):
    """Wrapper for XGBoost models with native SHAP support."""
    
    def __init__(self, params, num_boost_round=100):
        self.params = params.copy()
        self.num_boost_round = num_boost_round
        self.model = None
        
    def fit(self, X_train, y_train):
        """Train XGBoost model."""
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            verbose_eval=False,
        )
        return self
    
    def compute_shap(self, X_test, task="binary"):
        """Compute SHAP values using XGBoost's native implementation."""
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        shap_values = self.model.predict(dtest, pred_contribs=True, strict_shape=True)
        
        # Extract feature contributions (remove bias term)
        if task == "multiclass":
            # Shape: (n_samples, n_classes, n_features + 1)
            return shap_values[:, :, :-1]
        else:
            # Shape: (n_samples, n_features + 1)
            return shap_values[:, :-1]


class SklearnWrapper(ModelWrapper):
    """
    Wrapper for sklearn models using SHAP TreeExplainer, LinearExplainer, or KernelExplainer.
    
    Memory Optimization:
    - For KernelExplainer, uses k-means sampling to select ~100 representative 
      background samples instead of using the entire training dataset
    - Processes test samples in batches to reduce peak memory usage
    - Creates explainer just-in-time to allow garbage collection
    - For linear models (LogisticRegression, LinearRegression, etc.), uses LinearExplainer
    """
    
    def __init__(self, model_class, model_params=None, use_tree_explainer=True, use_linear_explainer=False):
        """
        Parameters
        ----------
        model_class : class
            Sklearn model class (e.g., RandomForestClassifier, LogisticRegression)
        model_params : dict
            Parameters to pass to model_class
        use_tree_explainer : bool
            If True, uses TreeExplainer (for tree-based models).
        use_linear_explainer : bool
            If True, uses LinearExplainer (for linear models - much faster than KernelExplainer).
            If False and use_tree_explainer is False, uses KernelExplainer.
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.use_tree_explainer = use_tree_explainer
        self.use_linear_explainer = use_linear_explainer
        self.model = None
        self.background_data = None
        
    def fit(self, X_train, y_train):
        """Train sklearn model."""
        self.model = self.model_class(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Store training data for explainer creation
        if self.use_linear_explainer:
            # LinearExplainer needs the full training data
            self.background_data = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        elif self.use_tree_explainer:
            self.background_data = None
        else:
            # For KernelExplainer, use a sampled background dataset
            X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            n_background = min(10, len(X_train_array))  # Reduced from 100 to 10 for speed
            self.background_data = shap.sample(X_train_array, n_background, random_state=42)
        
        return self
    
    def compute_shap(self, X_test, task="binary"):
        """Compute SHAP values using SHAP library with optimized memory usage."""
        # Convert to numpy array if DataFrame
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Create explainer just-in-time to allow garbage collection after use
        if self.use_tree_explainer:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test_array)
        elif self.use_linear_explainer:
            # LinearExplainer for linear models (very fast and exact!)
            explainer = shap.LinearExplainer(self.model, self.background_data)
            shap_values = explainer.shap_values(X_test_array)
        else:
            # Use sampled background data for KernelExplainer (much lower memory)
            explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                self.background_data
            )
            
            # Process in batches to reduce memory peaks
            batch_size = min(50, len(X_test_array))
            shap_values_list = []
            
            for i in range(0, len(X_test_array), batch_size):
                batch = X_test_array[i:i+batch_size]
                batch_shap = explainer.shap_values(batch)
                shap_values_list.append(batch_shap)
            
            # Combine batches
            if isinstance(shap_values_list[0], list):
                # Multiclass case: list of arrays
                n_classes = len(shap_values_list[0])
                shap_values = [np.vstack([batch[c] for batch in shap_values_list]) 
                              for c in range(n_classes)]
            else:
                # Binary/regression case
                shap_values = np.vstack(shap_values_list)
        
        # Handle different return formats
        if task == "multiclass":
            # SHAP returns list of arrays for multiclass, convert to 3D array
            if isinstance(shap_values, list):
                # Shape: [(n_samples, n_features)] * n_classes -> (n_samples, n_classes, n_features)
                shap_values = np.array(shap_values).transpose(1, 0, 2)
            # If already 3D, ensure correct shape
            elif len(shap_values.shape) == 3:
                # Check if it's (n_samples, n_features, n_classes) and needs transposing
                # KernelExplainer often returns (n_samples, n_features, n_classes)
                # but we need (n_samples, n_classes, n_features)
                if shap_values.shape[2] < shap_values.shape[1]:
                    # Last dim is likely n_classes (smaller), so transpose
                    shap_values = shap_values.transpose(0, 2, 1)
            else:
                raise ValueError(f"Unexpected SHAP values shape for multiclass: {shap_values.shape}")
        else:
            # For binary classification, TreeExplainer might return just class 1
            if isinstance(shap_values, list):
                # Use positive class SHAP values
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            # Ensure 2D
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, 1)
        
        return shap_values


class LightGBMWrapper(ModelWrapper):
    """Wrapper for LightGBM models."""
    
    def __init__(self, params, num_boost_round=100):
        self.params = params.copy()
        self.num_boost_round = num_boost_round
        self.model = None
        self.categorical_columns = []
        self.category_levels = {}
        self.background_data = None

    # def _prepare_frame(self, X, fit=False):
    #     """Convert object/category columns to stable integer category codes."""
    #     if not isinstance(X, pd.DataFrame):
    #         return X

    #     X_prepared = X.copy()

    #     if fit:
    #         self.categorical_columns = [
    #             column
    #             for column in X_prepared.columns
    #             if pd.api.types.is_object_dtype(X_prepared[column])
    #             or pd.api.types.is_categorical_dtype(X_prepared[column])
    #         ]
    #         self.category_levels = {}
    #         for column in self.categorical_columns:
    #             categorical = pd.Categorical(X_prepared[column])
    #             self.category_levels[column] = list(categorical.categories)
    #             X_prepared[column] = categorical.codes.astype(np.int32)
    #         return X_prepared

    #     for column in self.categorical_columns:
    #         categorical = pd.Categorical(
    #             X_prepared[column],
    #             categories=self.category_levels.get(column, []),
    #         )
    #         X_prepared[column] = categorical.codes.astype(np.int32)

    #     return X_prepared
        
    def fit(self, X_train, y_train):
        """Train LightGBM model."""
        
        # X_train_prepared = self._prepare_frame(X_train, fit=True)
        # if isinstance(X_train_prepared, pd.DataFrame):
        #     self.background_data = X_train_prepared.sample(
        #         n=min(100, len(X_train_prepared)),
        #         random_state=42,
        #     )
        # else:
        #     self.background_data = X_train_prepared[: min(100, len(X_train_prepared))]
        dtrain = lgb.Dataset(
            # X_train_prepared,
            X_train,
            label=y_train,
            categorical_feature=self.categorical_columns,
        )
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            # verbose_eval=False,
        )
        return self
    
    def compute_shap(self, X_test, task="binary"):
        """Compute SHAP values using LightGBM's native implementation."""
        if self.model is None:
            raise ValueError("Model has not been fit yet.")

        shap_values = self.model.predict(X_test, pred_contrib=True)
        
        # Extract feature contributions (remove bias term)
        if task == "multiclass":
            # LightGBM returns shape: (n_samples, n_classes * (n_features + 1))
            # Need to reshape to: (n_samples, n_classes, n_features + 1)
            n_samples = shap_values.shape[0]
            num_class = self.params.get('num_class', 2)
            n_features_with_bias = shap_values.shape[1] // num_class
            
            # Reshape to (n_samples, n_classes, n_features + 1)
            shap_values = shap_values.reshape(n_samples, num_class, n_features_with_bias)
            
            # Remove bias term (last column)
            return shap_values[:, :, :-1]
        else:
            # Binary: Shape is (n_samples, n_features + 1)
            return shap_values[:, :-1]


class CatBoostWrapper(ModelWrapper):
    """Wrapper for CatBoost models."""
    
    def __init__(self, params, num_boost_round=100):
        self.params = params.copy()
        self.num_boost_round = num_boost_round
        self.model = None
        
    def fit(self, X_train, y_train):
        """Train CatBoost model."""
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        
        self.model = CatBoostClassifier(
            iterations=self.num_boost_round,
            verbose=False,
            **self.params
        )
        self.model.fit(X_train, y_train)
        return self
    
    def compute_shap(self, X_test, task="binary"):
        """Compute SHAP values using CatBoost's implementation."""
        try:
            from catboost import Pool
        except ImportError:
            raise ImportError("CatBoost not installed")
        
        pool = Pool(X_test)
        shap_values = self.model.get_feature_importance(
            pool, 
            type='ShapValues'
        )
        
        # Remove bias term (last column)
        if task == "multiclass":
            # Shape: (n_samples, n_features + 1) for each class
            # CatBoost returns different format, need to verify
            return shap_values[:, :-1]
        else:
            # Shape: (n_samples, n_features + 1)
            return shap_values[:, :-1]


class PyTorchWrapper(ModelWrapper):
    """Wrapper for PyTorch neural network models."""
    
    def __init__(self, model_fn, model_params=None, train_params=None, explainer_type="deep"):
        """
        Parameters
        ----------
        model_fn : callable
            Function that returns a PyTorch model instance
        model_params : dict
            Parameters to pass to model_fn
        train_params : dict
            Training parameters (epochs, lr, batch_size, etc.)
        explainer_type : str
            Type of SHAP explainer: "deep", "gradient", or "kernel"
        """
        self.model_fn = model_fn
        self.model_params = model_params or {}
        self.train_params = train_params or {
            "epochs": 100,
            "lr": 0.001,
            "batch_size": 32,
            "verbose": False
        }
        self.explainer_type = explainer_type
        self.model = None
        self.background_data = None
        
    def fit(self, X_train, y_train):
        """Train PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle random_state for reproducibility
        model_params = self.model_params.copy()
        random_state = model_params.pop('random_state', None)
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        # Create model and move to device
        self.model = self.model_fn(**model_params)
        self.model = self.model.to(self.device)
        
        # Convert data to tensors and move to device
        X_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train.values if isinstance(y_train, pd.Series) else y_train).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_params.get("batch_size", 32),
            shuffle=True
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_params.get("lr", 0.001))
        
        # Train
        self.model.train()
        epochs = self.train_params.get("epochs", 100)
        verbose = self.train_params.get("verbose", False)
        
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                # Data already on device from TensorDataset
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Store background data for SHAP explainer (move to CPU for SHAP)
        self.model.eval()
        if self.explainer_type in ["deep", "gradient"]:
            # Keep on device for deep/gradient explainers
            self.background_data = X_tensor[:min(10, len(X_tensor))]
        else:  # kernel - use shap.sample (on CPU)
            X_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            n_background = min(10, len(X_train))  # Use 10 samples for speed
            self.background_data = shap.sample(X_array, n_background, random_state=0)
        
        return self
    
    def compute_shap(self, X_test, task="binary"):
        """Compute SHAP values using specified explainer with optimized memory usage."""
        import torch
        
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Create explainer just-in-time
        if self.explainer_type == "deep":
            explainer = shap.DeepExplainer(self.model, self.background_data)
            X_test_tensor = torch.FloatTensor(X_test_array).to(self.device)
            shap_values = explainer.shap_values(X_test_tensor)
        elif self.explainer_type == "gradient":
            explainer = shap.GradientExplainer(self.model, self.background_data)
            X_test_tensor = torch.FloatTensor(X_test_array).to(self.device)
            shap_values = explainer.shap_values(X_test_tensor)
        else:  # kernel - process in batches
            # For kernel explainer, use CPU model prediction
            def model_predict(x):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    output = self.model(x_tensor)
                    return output.cpu().numpy()
            
            explainer = shap.KernelExplainer(model_predict, self.background_data)
            
            # Process in batches to reduce memory
            batch_size = min(50, len(X_test_array))
            shap_values_list = []
            for i in range(0, len(X_test_array), batch_size):
                batch = X_test_array[i:i+batch_size]
                batch_shap = explainer.shap_values(batch)
                shap_values_list.append(batch_shap)
            
            # Combine batches
            if isinstance(shap_values_list[0], list):
                n_classes = len(shap_values_list[0])
                shap_values = [np.vstack([batch[c] for batch in shap_values_list]) 
                              for c in range(n_classes)]
            else:
                shap_values = np.vstack(shap_values_list)
        
        # Handle different return formats
        if task == "multiclass":
            if isinstance(shap_values, list):
                # List of arrays per class: [(n_samples, n_features), ...] -> (n_samples, n_classes, n_features)
                shap_values = np.array(shap_values).transpose(1, 0, 2)
            elif len(shap_values.shape) == 3:
                # KernelExplainer may return (n_samples, n_features, n_classes) for multiclass
                # Transpose to standard format (n_samples, n_classes, n_features)
                if shap_values.shape[1] > shap_values.shape[2]:
                    # Likely (n_samples, n_features, n_classes) - transpose
                    shap_values = shap_values.transpose(0, 2, 1)
        else:
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        return shap_values


class KerasWrapper(ModelWrapper):
    """Wrapper for TensorFlow/Keras neural network models."""
    
    def __init__(self, model_fn, model_params=None, train_params=None, explainer_type="deep"):
        """
        Parameters
        ----------
        model_fn : callable
            Function that returns a compiled Keras model
        model_params : dict
            Parameters to pass to model_fn
        train_params : dict
            Training parameters (epochs, batch_size, etc.)
        explainer_type : str
            Type of SHAP explainer: "deep", "gradient", or "kernel"
        """
        self.model_fn = model_fn
        self.model_params = model_params or {}
        self.train_params = train_params or {
            "epochs": 100,
            "batch_size": 32,
            "verbose": 0
        }
        self.explainer_type = explainer_type
        self.model = None
        self.background_data = None
        
    def fit(self, X_train, y_train):
        """Train Keras model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        # Create model
        self.model = self.model_fn(**self.model_params)
        
        # Convert data
        X_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_array = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        # Train
        self.model.fit(
            X_array,
            y_array,
            epochs=self.train_params.get("epochs", 100),
            batch_size=self.train_params.get("batch_size", 32),
            verbose=self.train_params.get("verbose", 0)
        )
        
        # Store background data for SHAP explainer
        if self.explainer_type in ["deep", "gradient"]:
            self.background_data = X_array[:min(100, len(X_array))]
        else:  # kernel - use k-means sampling to reduce memory
            self.background_data = shap.kmeans(X_train, min(100, len(X_train)))
        
        return self
    
    def compute_shap(self, X_test, task="binary"):
        """Compute SHAP values using specified explainer with optimized memory usage."""
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Create explainer just-in-time
        if self.explainer_type == "deep":
            explainer = shap.DeepExplainer(self.model, self.background_data)
            shap_values = explainer.shap_values(X_test_array)
        elif self.explainer_type == "gradient":
            explainer = shap.GradientExplainer(self.model, self.background_data)
            shap_values = explainer.shap_values(X_test_array)
        else:  # kernel - process in batches
            explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            
            # Process in batches to reduce memory
            batch_size = min(50, len(X_test_array))
            shap_values_list = []
            for i in range(0, len(X_test_array), batch_size):
                batch = X_test_array[i:i+batch_size]
                batch_shap = explainer.shap_values(batch)
                shap_values_list.append(batch_shap)
            
            # Combine batches
            if isinstance(shap_values_list[0], list):
                n_classes = len(shap_values_list[0])
                shap_values = [np.vstack([batch[c] for batch in shap_values_list]) 
                              for c in range(n_classes)]
            else:
                shap_values = np.vstack(shap_values_list)
        
        # Handle different return formats
        if task == "multiclass":
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).transpose(1, 0, 2)
        else:
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        return shap_values


def create_model_wrapper(model_type, **kwargs):
    """
    Factory function to create model wrappers.
    
    Parameters
    ----------
    model_type : str
        One of: "xgboost", "sklearn", "lightgbm", "catboost", "pytorch", "keras"
    **kwargs : dict
        Arguments passed to the wrapper constructor
        
    Examples
    --------
    # XGBoost
    wrapper = create_model_wrapper("xgboost", params=xgb_params, num_boost_round=100)
    
    # Sklearn Random Forest
    from sklearn.ensemble import RandomForestClassifier
    wrapper = create_model_wrapper(
        "sklearn",
        model_class=RandomForestClassifier,
        model_params={"n_estimators": 100, "random_state": 42}
    )
    
    # LightGBM
    wrapper = create_model_wrapper("lightgbm", params=lgb_params, num_boost_round=100)
    
    # PyTorch
    def create_pytorch_model(input_dim, hidden_dim=64, num_classes=2):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    wrapper = create_model_wrapper(
        "pytorch",
        model_fn=create_pytorch_model,
        model_params={"input_dim": 10, "hidden_dim": 64, "num_classes": 2},
        train_params={"epochs": 50, "lr": 0.001}
    )
    
    # Keras/TensorFlow
    def create_keras_model(input_dim, num_classes=2):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    wrapper = create_model_wrapper(
        "keras",
        model_fn=create_keras_model,
        model_params={"input_dim": 10, "num_classes": 2},
        train_params={"epochs": 50}
    )
    """
    model_type = model_type.lower()
    
    if model_type == "xgboost":
        return XGBoostWrapper(**kwargs)
    elif model_type == "sklearn":
        return SklearnWrapper(**kwargs)
    elif model_type == "lightgbm":
        return LightGBMWrapper(**kwargs)
    elif model_type == "catboost":
        return CatBoostWrapper(**kwargs)
    elif model_type in ["pytorch", "torch"]:
        return PyTorchWrapper(**kwargs)
    elif model_type in ["keras", "tensorflow", "tf"]:
        return KerasWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Choose from: xgboost, sklearn, lightgbm, catboost, pytorch, keras")
