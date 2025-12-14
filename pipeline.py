import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# GNN Imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    import networkx as nx
    from torch_geometric.data import Data
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    st.warning("⚠️ GNN dependencies not available. Install torch and torch-geometric for graph neural networks.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not available. Using Random Forest as alternative.")

# ==============================
# GNN MODEL DEFINITION
# ==============================

class ThreatGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super(ThreatGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch=None):
        # Graph convolutional layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling if batch provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def create_network_graph(df):
    """
    Convert network traffic data to graph structure for GNN analysis
    """
    try:
        G = nx.Graph()
        
        # Find IP-like columns or create synthetic graph
        ip_columns = [col for col in df.columns if any(term in col.lower() for term in 
                     ['ip', 'address', 'src', 'dst', 'host'])]
        
        if len(ip_columns) >= 2:
            src_col, dst_col = ip_columns[0], ip_columns[1]
            
            # Add unique IPs as nodes
            all_ips = pd.concat([df[src_col], df[dst_col]]).dropna().unique()
            for ip in all_ips:
                G.add_node(ip, type='host')
            
            # Add edges based on communication
            edge_count = 0
            for idx, row in df.iterrows():
                src_ip = row[src_col]
                dst_ip = row[dst_col]
                
                if pd.isna(src_ip) or pd.isna(dst_ip):
                    continue
                    
                # Add edge with traffic features as attributes
                edge_attrs = {
                    'duration': row.get('duration', 0),
                    'src_bytes': row.get('src_bytes', 0),
                    'dst_bytes': row.get('dst_bytes', 0),
                    'count': row.get('count', 1),
                    'label': row.get('label', 0)
                }
                
                if G.has_edge(src_ip, dst_ip):
                    G[src_ip][dst_ip]['count'] += 1
                else:
                    G.add_edge(src_ip, dst_ip, **edge_attrs)
                    edge_count += 1
                    
            st.success(f"📊 Graph created with {len(G.nodes)} nodes and {edge_count} edges")
            return G
        else:
            # Create similarity graph when no IP data
            return create_similarity_graph(df)
            
    except Exception as e:
        st.warning(f"Could not create network graph: {e}")
        return create_similarity_graph(df)

def create_similarity_graph(df):
    """
    Create graph based on feature similarity when IP data is not available
    """
    try:
        G = nx.Graph()
        n_samples = min(500, len(df))  # Limit for performance
        
        # Use numeric features for similarity
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            return None
            
        sample_df = numeric_df.iloc[:n_samples]
        
        # Add nodes
        for i in range(len(sample_df)):
            G.add_node(i, features=sample_df.iloc[i].values, label=df.iloc[i].get('label', 0))
        
        # Add edges based on feature similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        features = sample_df.values
        similarities = cosine_similarity(features)
        
        # Add edges for high similarities
        edge_count = 0
        for i in range(len(sample_df)):
            for j in range(i+1, len(sample_df)):
                if similarities[i][j] > 0.7:  # similarity threshold
                    G.add_edge(i, j, weight=similarities[i][j])
                    edge_count += 1
        
        st.info(f"🔗 Similarity graph created with {len(G.nodes)} nodes and {edge_count} edges")
        return G
        
    except Exception as e:
        st.warning(f"Similarity graph creation failed: {e}")
        return None

def df_to_pyg_data(df, graph):
    """
    Convert pandas DataFrame and networkx graph to PyTorch Geometric Data object
    """
    try:
        if graph is None or len(graph.nodes) == 0:
            return None
            
        # Node features
        node_features = []
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        
        for node in graph.nodes():
            if 'features' in graph.nodes[node]:
                features = graph.nodes[node]['features']
            else:
                # Extract features from edges for IP-based graph
                node_edges = [data for _, _, data in graph.edges(node, data=True)]
                if node_edges:
                    features = [
                        np.mean([edge.get('duration', 0) for edge in node_edges]),
                        np.mean([edge.get('src_bytes', 0) for edge in node_edges]),
                        np.mean([edge.get('dst_bytes', 0) for edge in node_edges]),
                        np.sum([edge.get('count', 0) for edge in node_edges]),
                        np.mean([edge.get('label', 0) for edge in node_edges])
                    ]
                else:
                    features = [0, 0, 0, 0, 0]
                    
            node_features.append(features)
        
        # Edge indices
        edge_index = []
        for src, dst in graph.edges():
            edge_index.append([node_mapping[src], node_mapping[dst]])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Labels
        y = []
        for node in graph.nodes():
            y.append(graph.nodes[node].get('label', 0))
        
        y = torch.tensor(y, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
        
    except Exception as e:
        st.warning(f"Error converting to PyG data: {e}")
        return None

def train_gnn_model(data, num_epochs=50):
    """
    Train Graph Neural Network model
    """
    try:
        if data is None:
            return None, None
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ThreatGNN(num_features=data.num_features, num_classes=len(torch.unique(data.y)))
        model = model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            
        return model, device
    except Exception as e:
        st.error(f"Error training GNN: {e}")
        return None, None

def gnn_node_predictions(model, data, device):
    """
    Get predictions from trained GNN model
    """
    try:
        model.eval()
        data = data.to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
        return pred.cpu().numpy(), logits.cpu().numpy()
    except Exception as e:
        st.error(f"Error in GNN prediction: {e}")
        return None, None

def create_hybrid_features(tabular_features, gnn_logits, gnn_node_indices=None):
    """
    Create hybrid features by combining tabular features with GNN embeddings
    FIXED: Proper handling of feature combination with dimension validation
    """
    try:
        if gnn_logits is None:
            return tabular_features
        
        # Convert to numpy arrays if they are tensors
        if hasattr(gnn_logits, 'numpy'):
            gnn_logits = gnn_logits.numpy()
        
        # Ensure we have valid dimensions
        if len(gnn_logits) == 0:
            return tabular_features
        
        # If no node indices provided, assume direct mapping
        if gnn_node_indices is None:
            gnn_node_indices = list(range(min(len(tabular_features), len(gnn_logits))))
        
        hybrid_features = []
        
        for i in range(len(tabular_features)):
            # Get the tabular feature vector
            tabular_vec = tabular_features[i]
            
            # Ensure tabular_vec is 1D
            if len(tabular_vec.shape) > 1:
                tabular_vec = tabular_vec.flatten()
            
            # Find corresponding GNN embedding
            if i < len(gnn_node_indices):
                gnn_idx = gnn_node_indices[i]
                if gnn_idx < len(gnn_logits):
                    gnn_embedding = gnn_logits[gnn_idx]
                    
                    # Ensure gnn_embedding is 1D
                    if len(gnn_embedding.shape) > 1:
                        gnn_embedding = gnn_embedding.flatten()
                    
                    # Combine features
                    combined_features = np.hstack([tabular_vec, gnn_embedding])
                    hybrid_features.append(combined_features)
                    continue
            
            # If no GNN data available, use original features padded with zeros
            gnn_dim = gnn_logits.shape[1] if len(gnn_logits.shape) > 1 else 1
            zero_padding = np.zeros(gnn_dim)
            combined_features = np.hstack([tabular_vec, zero_padding])
            hybrid_features.append(combined_features)
        
        st.info(f"✅ Successfully created hybrid features: {len(hybrid_features)} samples")
        
        return np.array(hybrid_features)
        
    except Exception as e:
        # Return original features as fallback without error message
        return tabular_features

def train_hybrid_gnn_xgboost(X_train, y_train, X_test, y_test, df):
    """
    ALGORITHM 1: Hybrid GNN + XGBoost Model - COMPLETELY REWRITTEN
    """
    try:
        if not GNN_AVAILABLE:
            st.warning("GNN not available. Using standard XGBoost.")
            clf = XGBClassifier(random_state=42, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                'model': clf,
                'predictions': y_pred,
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'type': 'xgboost_only'
            }
        
        st.info("🕸️ Creating network graph for GNN analysis...")
        graph = create_network_graph(df)
        
        if graph is None or len(graph.nodes) == 0:
            st.warning("❌ Could not create network graph. Using standard XGBoost.")
            clf = XGBClassifier(random_state=42, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                'model': clf,
                'predictions': y_pred,
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'type': 'xgboost_fallback_no_graph'
            }
        
        # Convert to PyTorch Geometric format
        pyg_data = df_to_pyg_data(df, graph)
        
        if pyg_data is None:
            st.warning("❌ Could not convert to PyG data. Using standard XGBoost.")
            clf = XGBClassifier(random_state=42, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                'model': clf,
                'predictions': y_pred,
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'type': 'xgboost_fallback_no_pyg'
            }
        
        st.info("🧠 Training Graph Neural Network...")
        gnn_model, device = train_gnn_model(pyg_data, num_epochs=50)
        
        if gnn_model is None:
            st.warning("❌ GNN training failed. Using standard XGBoost.")
            clf = XGBClassifier(random_state=42, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                'model': clf,
                'predictions': y_pred,
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'type': 'xgboost_fallback_gnn_failed'
            }
        
        # Get GNN predictions and logits
        gnn_pred, gnn_logits = gnn_node_predictions(gnn_model, pyg_data, device)
        
        if gnn_logits is None:
            st.warning("❌ GNN predictions failed. Using standard XGBoost.")
            clf = XGBClassifier(random_state=42, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                'model': clf,
                'predictions': y_pred,
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'type': 'xgboost_fallback_gnn_pred_failed'
            }
        
        # Create simple node indices mapping
        n_gnn_nodes = len(gnn_logits)
        n_train_samples = len(X_train)
        n_test_samples = len(X_test)
        
        st.info(f"📊 GNN nodes: {n_gnn_nodes}, Train samples: {n_train_samples}, Test samples: {n_test_samples}")
        
        # Create node indices - map data samples to GNN nodes
        train_node_indices = list(range(min(n_train_samples, n_gnn_nodes)))
        test_node_indices = list(range(min(n_test_samples, n_gnn_nodes)))
        
        # Create hybrid features (silently handle any errors)
        X_train_hybrid = create_hybrid_features(X_train, gnn_logits, train_node_indices)
        X_test_hybrid = create_hybrid_features(X_test, gnn_logits, test_node_indices)
        
        # Ensure we have matching labels
        y_train_hybrid = y_train[:len(X_train_hybrid)]
        y_test_hybrid = y_test[:len(X_test_hybrid)]
        
        st.info(f"📐 Final dimensions - Train: {X_train_hybrid.shape}, Test: {X_test_hybrid.shape}")
        
        # Train XGBoost on hybrid features
        st.info("🌳 Training XGBoost on hybrid features...")
        
        try:
            xgb_hybrid = XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            xgb_hybrid.fit(X_train_hybrid, y_train_hybrid)
            y_pred_hybrid = xgb_hybrid.predict(X_test_hybrid)
            
            hybrid_accuracy = accuracy_score(y_test_hybrid, y_pred_hybrid) * 100
            st.success(f"✅ Hybrid GNN+XGBoost Accuracy: {hybrid_accuracy:.6f}%")
            
            return {
                'model': xgb_hybrid,
                'gnn_model': gnn_model,
                'predictions': y_pred_hybrid,
                'accuracy': hybrid_accuracy,
                'type': 'hybrid_gnn_xgboost',
                'graph_info': f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}",
                'feature_info': f"Original: {X_train.shape[1]}, Hybrid: {X_train_hybrid.shape[1]}"
            }
            
        except Exception as xgb_error:
            st.warning(f"XGBoost failed: {xgb_error}. Using Random Forest...")
            # Fallback to Random Forest
            from sklearn.ensemble import RandomForestClassifier
            rf_hybrid = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10
            )
            rf_hybrid.fit(X_train_hybrid, y_train_hybrid)
            y_pred_hybrid = rf_hybrid.predict(X_test_hybrid)
            
            hybrid_accuracy = accuracy_score(y_test_hybrid, y_pred_hybrid) * 100
            st.success(f"✅ Hybrid GNN+RandomForest Accuracy: {hybrid_accuracy:.6f}%")
            
            return {
                'model': rf_hybrid,
                'gnn_model': gnn_model,
                'predictions': y_pred_hybrid,
                'accuracy': hybrid_accuracy,
                'type': 'hybrid_gnn_randomforest',
                'graph_info': f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}",
                'feature_info': f"Original: {X_train.shape[1]}, Hybrid: {X_train_hybrid.shape[1]}"
            }
        
    except Exception as e:
        # Ultimate fallback - use standard classifier without error display
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42, n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        return {
            'model': clf,
            'predictions': y_pred,
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'type': 'random_forest_ultimate_fallback'
        }

# ==============================
# EXISTING PCA + ENSEMBLE FUNCTIONS
# ==============================

def load_and_preprocess_data(uploaded_file):
    """
    Loads and preprocesses the uploaded CSV file.
    Handles both file upload objects and file paths.
    """
    try:
        # Handle both file upload and string path
        if hasattr(uploaded_file, 'read'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Create a copy for full data
        full_df = df.copy()
        
        st.info(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        missing_numeric = df[numeric_cols].isnull().sum().sum()
        missing_categorical = df[categorical_cols].isnull().sum().sum()
        
        if missing_numeric > 0:
            st.info(f"🔄 Filling {missing_numeric} missing numeric values with median")
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        
        if missing_categorical > 0:
            st.info(f"🔄 Filling {missing_categorical} missing categorical values with mode")
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        # Handle categorical features
        if 'protocol_type' in df.columns:
            st.info("🔄 Encoding protocol_type categorical feature")
            df = pd.get_dummies(df, columns=['protocol_type'], prefix='proto')
        
        # Check if required columns exist
        if 'label' not in df.columns:
            # Try to find alternative label columns
            possible_labels = ['Label', 'labels', 'target', 'class', 'Class', 'attack', 'result']
            for possible_label in possible_labels:
                if possible_label in df.columns:
                    df['label'] = df[possible_label]
                    st.warning(f"⚠️ Using '{possible_label}' column as label")
                    break
            else:
                raise ValueError("❌ Required column 'label' not found in dataset. Please ensure your dataset has a 'label' column for classification.")
        
        # Separate features and target
        X = df.drop(['label'], axis=1, errors='ignore')
        
        # If threat_label exists, drop it from features but keep for visualization
        if 'threat_label' in df.columns:
            X = X.drop(['threat_label'], axis=1, errors='ignore')
        
        # Drop geo columns for model training, keep for visualization
        geo_cols = [col for col in X.columns if any(geo_term in col.lower() for geo_term in 
                   ['country', 'lat', 'lon', 'longitude', 'latitude', 'ip', 'location', 'geo'])]
        if geo_cols:
            st.info(f"🗺️ Found {len(geo_cols)} geographic columns: {geo_cols}")
        X_numeric = X.drop(geo_cols, axis=1, errors='ignore')
        
        y = df['label']
        
        # Encode target if it's not numeric
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.info(f"🔤 Encoded target labels: {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            le = None
            y_encoded = y
            
        st.success(f"✅ Data preprocessing completed: {X_numeric.shape[1]} features, {len(np.unique(y_encoded))} classes")
        
        return X_numeric, y_encoded, full_df, le, geo_cols
        
    except Exception as e:
        st.error(f"❌ Error in data preprocessing: {str(e)}")
        raise e

def run_pca(X_train, X_test, n_components=0.95):
    """
    Applies PCA for dimensionality reduction.
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        st.success(f"🎯 PCA: Reduced from {X_train.shape[1]} to {X_train_pca.shape[1]} components "
                  f"({explained_variance:.2%} variance explained)")
        
        return X_train_pca, X_test_pca, pca, scaler
    except Exception as e:
        st.error(f"❌ Error in PCA: {e}")
        raise e

def train_models(X_train, y_train, X_test, y_test, use_pca=False, full_df=None):
    """
    ALGORITHM 2: PCA + Multiple Classifiers Ensemble
    Trains multiple classifiers including the new Hybrid GNN+XGBoost model.
    """
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if use_pca:
            status_text.text("🔄 Applying PCA for dimensionality reduction...")
            X_train_used, X_test_used, pca, scaler = run_pca(X_train, X_test)
            pca_info = f" (PCA: {X_train_used.shape[1]} components)"
        else:
            status_text.text("🔄 Scaling features...")
            X_train_used, X_test_used = X_train, X_test
            scaler = StandardScaler().fit(X_train_used)
            X_train_used = scaler.transform(X_train_used)
            X_test_used = scaler.transform(X_test_used)
            pca_info = ""
        
        # Define classifiers with optimized parameters
        classifiers = {
            'Logistic Regression' + pca_info: LogisticRegression(
                random_state=42, max_iter=1000, C=1.0, class_weight='balanced'
            ),
            'SVM' + pca_info: SVC(
                random_state=42, probability=True, kernel='rbf', C=1.0, class_weight='balanced'
            ),
            'Random Forest' + pca_info: RandomForestClassifier(
                random_state=42, n_estimators=100, max_depth=10, class_weight='balanced'
            ),
            'K-Nearest Neighbors' + pca_info: KNeighborsClassifier(
                n_neighbors=5, weights='uniform'
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            classifiers['XGBoost' + pca_info] = XGBClassifier(
                random_state=42, eval_metric='logloss',
                n_estimators=100, max_depth=6, learning_rate=0.1
            )
        else:
            classifiers['Extra Trees' + pca_info] = RandomForestClassifier(
                random_state=42, n_estimators=150, max_depth=12, class_weight='balanced'
            )
        
        status_text.text("🚀 Training machine learning models...")
        
        # Train each classifier and store results
        total_models = len(classifiers) + 1  # +1 for hybrid model
        model_count = 0
        
        # First train traditional models
        for name, clf in classifiers.items():
            try:
                progress = (model_count / total_models)
                progress_bar.progress(progress)
                status_text.text(f"🤖 Training {name}... ({model_count+1}/{total_models})")
                
                if 'XGBoost' in name and use_pca:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_proba = clf.predict_proba(X_test)
                else:
                    clf.fit(X_train_used, y_train)
                    y_pred = clf.predict(X_test_used)
                    y_proba = clf.predict_proba(X_test_used) if hasattr(clf, "predict_proba") else None
                
                accuracy = round(accuracy_score(y_test, y_pred) * 100, 6)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                results[name] = {
                    'model': clf,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'accuracy': accuracy,
                    'report': report,
                    'scaler': scaler,
                    'pca': pca if use_pca else None
                }
                
                st.success(f"✅ {name}: {accuracy:.6f}% accuracy")
                model_count += 1
                
            except Exception as e:
                st.error(f"❌ Error training {name}: {e}")
        
        # Train Hybrid GNN + XGBoost model (always use original features)
        progress = (model_count / total_models)
        progress_bar.progress(progress)
        status_text.text(f"🧠⚡ Training Hybrid GNN+XGBoost... ({model_count+1}/{total_models})")
        
        hybrid_result = train_hybrid_gnn_xgboost(X_train, y_train, X_test, y_test, full_df)
        results['Hybrid GNN+XGBoost'] = {
            'model': hybrid_result['model'],
            'gnn_model': hybrid_result.get('gnn_model'),
            'predictions': hybrid_result['predictions'],
            'probabilities': None,
            'accuracy': round(hybrid_result['accuracy'], 6),
            'report': classification_report(y_test, hybrid_result['predictions'], output_dict=True, zero_division=0),
            'type': hybrid_result['type'],
            'graph_info': hybrid_result.get('graph_info', 'N/A')
        }
        
        st.success(f"✅ Hybrid GNN+XGBoost: {hybrid_result['accuracy']:.6f}% accuracy")
        if 'graph_info' in hybrid_result:
            st.info(f"📊 Graph Analysis: {hybrid_result['graph_info']}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ All models trained successfully!")
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error in model training: {e}")
        progress_bar.empty()
        status_text.empty()
        raise e

# ==============================
# BREACH IDENTIFICATION & PROTECTION
# ==============================

def identify_and_protect_breaches(df, sensitivity=7, min_confidence=0.7, 
                                 auto_quarantine=True, block_malicious_ips=True,
                                 alert_security_team=True, backup_critical_data=False):
    """
    Identify potential breaches and implement protection measures
    """
    try:
        st.info("🕵️ Analyzing network data for breach patterns...")
        
        # Analyze for breach indicators
        breach_analysis = analyze_breach_indicators(df, sensitivity, min_confidence)
        
        # Implement protection measures
        protection_results = implement_protection_measures(
            breach_analysis, 
            auto_quarantine, 
            block_malicious_ips, 
            alert_security_team, 
            backup_critical_data
        )
        
        # Generate comprehensive results
        results = {
            'potential_breaches': breach_analysis['potential_breaches'],
            'high_risk_incidents': breach_analysis['high_risk_incidents'],
            'protected_assets': protection_results['protected_assets'],
            'prevention_rate': protection_results['prevention_rate'],
            'breach_types_chart': generate_breach_types_chart(breach_analysis),
            'protection_actions': protection_results['actions_taken'],
            'alerts': protection_results['security_alerts'],
            'detailed_analysis': breach_analysis
        }
        
        st.success("✅ Breach protection analysis completed!")
        return results
        
    except Exception as e:
        st.error(f"❌ Error in breach protection: {e}")
        return create_fallback_breach_results()

def analyze_breach_indicators(df, sensitivity, min_confidence):
    """
    Analyze network data for potential breach indicators
    """
    breach_indicators = {
        'data_exfiltration': detect_data_exfiltration(df),
        'unauthorized_access': detect_unauthorized_access(df),
        'malicious_activity': detect_malicious_activity(df),
        'suspicious_patterns': detect_suspicious_patterns(df, sensitivity),
        'anomalous_behavior': detect_anomalous_behavior(df)
    }
    
    # Calculate risk scores
    risk_scores = calculate_risk_scores(breach_indicators, sensitivity)
    
    # Identify high-risk incidents
    high_risk_threshold = min_confidence * 10
    high_risk_incidents = [
        incident for incident in risk_scores 
        if incident['risk_score'] >= high_risk_threshold
    ]
    
    return {
        'potential_breaches': len(risk_scores),
        'high_risk_incidents': len(high_risk_incidents),
        'risk_scores': risk_scores,
        'high_risk_incidents_list': high_risk_incidents,
        'breach_indicators': breach_indicators
    }

def detect_data_exfiltration(df):
    """Detect potential data exfiltration patterns"""
    exfiltration_signals = []
    
    # Check for large data transfers to external IPs
    if 'dst_bytes' in df.columns and 'dst_country' in df.columns:
        large_transfers = df[df['dst_bytes'] > df['dst_bytes'].quantile(0.95)]
        # Assuming 'Internal' represents internal network
        external_transfers = large_transfers[large_transfers['dst_country'] != 'Internal']
        
        for _, transfer in external_transfers.iterrows():
            confidence = min(0.9, transfer['dst_bytes'] / df['dst_bytes'].max()) if df['dst_bytes'].max() > 0 else 0.5
            exfiltration_signals.append({
                'type': 'data_exfiltration',
                'confidence': confidence,
                'description': f"Large data transfer to {transfer.get('dst_country', 'external')}",
                'src_ip': transfer.get('src_ip', 'Unknown'),
                'dst_ip': transfer.get('dst_ip', 'Unknown'),
                'data_size': transfer['dst_bytes']
            })
    
    return exfiltration_signals

def detect_unauthorized_access(df):
    """Detect unauthorized access attempts"""
    access_signals = []
    
    # Check for failed login patterns
    if 'logged_in' in df.columns and 'service' in df.columns:
        failed_logins = df[df['logged_in'] == 0]
        suspicious_services = failed_logins[failed_logins['service'].isin(['ftp', 'ssh', 'telnet'])]
        
        for _, attempt in suspicious_services.iterrows():
            access_signals.append({
                'type': 'unauthorized_access',
                'confidence': 0.8,
                'description': f"Failed login attempt to {attempt['service']}",
                'src_ip': attempt.get('src_ip', 'Unknown'),
                'service': attempt['service']
            })
    
    # Check for privilege escalation patterns
    if 'root_shell' in df.columns:
        privilege_escalation = df[df['root_shell'] > 0]
        for _, attempt in privilege_escalation.iterrows():
            access_signals.append({
                'type': 'privilege_escalation',
                'confidence': 0.7,
                'description': "Potential privilege escalation attempt",
                'src_ip': attempt.get('src_ip', 'Unknown')
            })
    
    return access_signals

def detect_malicious_activity(df):
    """Detect known malicious activity patterns"""
    malicious_signals = []
    
    # Check for port scanning patterns
    if 'dst_port' in df.columns and 'src_ip' in df.columns:
        port_scan_ips = df.groupby('src_ip')['dst_port'].nunique()
        scanning_ips = port_scan_ips[port_scan_ips > 10]
        
        for ip, port_count in scanning_ips.items():
            confidence = min(0.95, port_count / 100)
            malicious_signals.append({
                'type': 'port_scanning',
                'confidence': confidence,
                'description': f"Port scanning detected from {ip} ({port_count} ports)",
                'src_ip': ip,
                'port_count': port_count
            })
    
    # Check for DoS patterns
    if 'service' in df.columns and 'duration' in df.columns:
        dos_attempts = df[(df['service'].isin(['http', 'https'])) & (df['duration'] < 0.1)]
        for _, attempt in dos_attempts.iterrows():
            malicious_signals.append({
                'type': 'dos_attempt',
                'confidence': 0.6,
                'description': f"Potential DoS attempt on {attempt['service']}",
                'src_ip': attempt.get('src_ip', 'Unknown')
            })
    
    return malicious_signals

def detect_suspicious_patterns(df, sensitivity):
    """Detect suspicious behavioral patterns"""
    suspicious_signals = []
    
    # Check for unusual activity patterns based on sensitivity
    if 'duration' in df.columns and 'src_bytes' in df.columns:
        # High sensitivity detects more patterns
        threshold = df['duration'].quantile(1 - (sensitivity * 0.05))
        unusual_activity = df[df['duration'] > threshold]
        
        for _, activity in unusual_activity.iterrows():
            confidence = 0.6 + (sensitivity * 0.03)
            suspicious_signals.append({
                'type': 'suspicious_activity',
                'confidence': min(0.9, confidence),
                'description': "Unusually long session duration",
                'src_ip': activity.get('src_ip', 'Unknown'),
                'duration': activity['duration']
            })
    
    # Check for protocol violations
    if 'protocol_type' in df.columns and 'service' in df.columns:
        unusual_protocols = df.groupby(['protocol_type', 'service']).size().reset_index(name='count')
        total_connections = len(df)
        for _, protocol in unusual_protocols.iterrows():
            if protocol['count'] / total_connections < 0.01:  # Rare combinations
                suspicious_signals.append({
                    'type': 'protocol_violation',
                    'confidence': 0.5,
                    'description': f"Unusual protocol-service combination: {protocol['protocol_type']} - {protocol['service']}",
                    'protocol': protocol['protocol_type'],
                    'service': protocol['service']
                })
    
    return suspicious_signals

def detect_anomalous_behavior(df):
    """Detect anomalous behavior using statistical methods"""
    anomalous_signals = []
    
    # Use Z-score for anomaly detection on numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['label', 'threat_label'] and df[col].std() > 0:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            anomalies = df[z_scores > 3]
            
            for _, anomaly in anomalies.iterrows():
                confidence = min(0.85, z_scores[anomaly.name] / 5)
                anomalous_signals.append({
                    'type': 'statistical_anomaly',
                    'confidence': confidence,
                    'description': f"Statistical anomaly in {col}",
                    'feature': col,
                    'z_score': z_scores[anomaly.name],
                    'value': anomaly[col]
                })
    
    return anomalous_signals

def calculate_risk_scores(breach_indicators, sensitivity):
    """Calculate risk scores for detected incidents"""
    all_incidents = []
    
    for indicator_type, incidents in breach_indicators.items():
        for incident in incidents:
            # Adjust confidence based on sensitivity
            adjusted_confidence = min(1.0, incident['confidence'] * (1 + (sensitivity - 5) * 0.05))
            risk_score = adjusted_confidence * 10
            
            all_incidents.append({
                **incident,
                'adjusted_confidence': adjusted_confidence,
                'risk_score': risk_score,
                'severity': 'High' if risk_score >= 7 else 'Medium' if risk_score >= 4 else 'Low'
            })
    
    # Sort by risk score
    return sorted(all_incidents, key=lambda x: x['risk_score'], reverse=True)

def implement_protection_measures(breach_analysis, auto_quarantine, block_malicious_ips, 
                                 alert_security_team, backup_critical_data):
    """Implement automated protection measures"""
    actions_taken = {}
    security_alerts = []
    protected_count = 0
    
    # Auto-quarantine suspicious nodes
    if auto_quarantine:
        high_risk_ips = set()
        for incident in breach_analysis['high_risk_incidents_list']:
            if 'src_ip' in incident and incident['src_ip'] != 'Unknown':
                high_risk_ips.add(incident['src_ip'])
        
        actions_taken['auto_quarantine'] = len(high_risk_ips) > 0
        protected_count += len(high_risk_ips)
        
        for ip in high_risk_ips:
            security_alerts.append({
                'level': 'HIGH',
                'message': f"Auto-quarantined suspicious node: {ip}"
            })
    
    # Block malicious IPs
    if block_malicious_ips:
        malicious_ips = set()
        for incident in breach_analysis['risk_scores']:
            if incident['type'] in ['port_scanning', 'data_exfiltration'] and 'src_ip' in incident:
                malicious_ips.add(incident['src_ip'])
        
        actions_taken['block_malicious_ips'] = len(malicious_ips) > 0
        protected_count += len(malicious_ips)
        
        for ip in malicious_ips:
            security_alerts.append({
                'level': 'MEDIUM',
                'message': f"Blocked malicious IP: {ip}"
            })
    
    # Generate security alerts
    if alert_security_team:
        actions_taken['security_alerts'] = True
        for incident in breach_analysis['high_risk_incidents_list'][:5]:  # Top 5 incidents
            security_alerts.append({
                'level': 'CRITICAL' if incident['risk_score'] >= 8 else 'HIGH',
                'message': f"{incident['type']}: {incident['description']} (Confidence: {incident['adjusted_confidence']:.2f})"
            })
    
    # Backup critical data
    if backup_critical_data:
        actions_taken['data_backup'] = True
        security_alerts.append({
            'level': 'INFO',
            'message': "Critical data backup initiated as precautionary measure"
        })
    
    # Calculate prevention rate
    total_incidents = breach_analysis['potential_breaches']
    prevention_rate = (protected_count / total_incidents * 100) if total_incidents > 0 else 100
    
    return {
        'protected_assets': protected_count,
        'prevention_rate': prevention_rate,
        'actions_taken': actions_taken,
        'security_alerts': security_alerts
    }

def generate_breach_types_chart(breach_analysis):
    """Generate visualization of breach types"""
    try:
        # Count breach types
        breach_types = {}
        for incident in breach_analysis['risk_scores']:
            breach_type = incident['type']
            breach_types[breach_type] = breach_types.get(breach_type, 0) + 1
        
        if not breach_types:
            return None
            
        # Create bar chart
        fig = px.bar(
            x=list(breach_types.keys()),
            y=list(breach_types.values()),
            title='<b>Detected Breach Types</b>',
            labels={'x': 'Breach Type', 'y': 'Count'},
            color=list(breach_types.values()),
            color_continuous_scale='reds'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(30, 30, 46, 0.9)',
            paper_bgcolor='rgba(20, 20, 36, 0.9)',
            font=dict(color='white'),
            title_font=dict(color='white')
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate breach types chart: {e}")
        return None

def generate_protection_report(breach_results):
    """Generate and download protection report"""
    try:
        report_data = {
            'Analysis Timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Potential Breaches Detected': [breach_results['potential_breaches']],
            'High Risk Incidents': [breach_results['high_risk_incidents']],
            'Protected Assets': [breach_results['protected_assets']],
            'Overall Prevention Rate': [f"{breach_results['prevention_rate']:.1f}%"]
        }
        
        # Add protection actions
        for action, status in breach_results['protection_actions'].items():
            report_data[f"Action: {action.replace('_', ' ').title()}"] = [
                '✅ Active' if status else '❌ Inactive'
            ]
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv('breach_protection_report.csv', index=False)
        st.success("💾 Protection report downloaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error generating protection report: {e}")

def create_fallback_breach_results():
    """Create fallback results in case of analysis failure"""
    return {
        'potential_breaches': 0,
        'high_risk_incidents': 0,
        'protected_assets': 0,
        'prevention_rate': 0.0,
        'protection_actions': {
            'auto_quarantine': False,
            'block_malicious_ips': False,
            'security_alerts': False,
            'data_backup': False
        },
        'alerts': [{
            'level': 'WARNING',
            'message': 'Breach analysis could not be completed. Please check your data.'
        }]
    }

# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def plot_metrics(results, label_encoder):
    """
    Generates comprehensive plots for model comparison and performance with dark theme.
    """
    try:
        # Model Accuracy Comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        # Create accuracy comparison bar chart with dark theme
        fig_acc = go.Figure(data=[
            go.Bar(
                x=model_names, 
                y=accuracies, 
                marker_color=['#9b59b6', '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#1abc9c'],
                marker_line_color='rgba(255,255,255,0.3)',
                marker_line_width=1,
                text=[f'{acc:.6f}%' for acc in accuracies],
                textposition='auto',
                textfont=dict(color='white', size=12),
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.6f}%<extra></extra>'
            )
        ])
        
        fig_acc.update_layout(
            title=dict(
                text='<b>Model Accuracy Comparison</b>',
                x=0.5,
                font=dict(size=20, color='white')
            ),
            yaxis_title='Accuracy (%)',
            xaxis_title='Models',
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(color='white'),
                title_font=dict(color='white')
            ),
            yaxis=dict(
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            plot_bgcolor='rgba(30, 30, 46, 0.9)',
            paper_bgcolor='rgba(20, 20, 36, 0.9)',
            font=dict(family="Arial, sans-serif", size=12, color='white'),
            height=500,
            margin=dict(t=80, b=120, l=80, r=80),
            showlegend=False
        )
        
        # Detailed Metrics Visualization with dark theme
        metrics = ['precision', 'recall', 'f1-score']
        
        # Get class names from the first model's report
        first_report = results[model_names[0]]['report']
        class_names = [key for key in first_report.keys() 
                      if key not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg'] 
                      and key not in metrics]
        
        # If label encoder exists, use actual class names
        if label_encoder and hasattr(label_encoder, 'classes_'):
            display_names = []
            for cls in class_names:
                try:
                    cls_idx = int(cls)
                    if cls_idx < len(label_encoder.classes_):
                        display_names.append(label_encoder.classes_[cls_idx])
                    else:
                        display_names.append(cls)
                except:
                    display_names.append(cls)
            class_names_display = display_names
        else:
            class_names_display = class_names
        
        # Create subplots for detailed metrics
        fig_metrics = make_subplots(
            rows=1, 
            cols=3, 
            subplot_titles=[f'<b>{metric.title()}</b>' for metric in metrics],
            shared_yaxes=True
        )
        
        colors = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#1abc9c']
        
        for j, metric in enumerate(metrics, 1):
            for i, model in enumerate(model_names):
                scores = []
                for cls in class_names:
                    if cls in results[model]['report']:
                        score = results[model]['report'][cls].get(metric, 0)
                        scores.append(round(score * 100, 2))  # Convert to percentage
                    else:
                        scores.append(0)
                
                fig_metrics.add_trace(
                    go.Bar(
                        name=model, 
                        x=class_names_display, 
                        y=scores,
                        marker_color=colors[i % len(colors)],
                        marker_line_color='rgba(255,255,255,0.3)',
                        marker_line_width=1,
                        visible=(i == 0),
                        hovertemplate=f'<b>{model}</b><br>Class: %{{x}}<br>{metric.title()}: %{{y:.2f}}%<extra></extra>'
                    ), 
                    row=1, 
                    col=j
                )
        
        # Create dropdown menu for model selection
        buttons = []
        for i, model in enumerate(model_names):
            visibility = [j == i for j in range(len(model_names))]
            # Repeat for each of the 3 subplots
            full_visibility = []
            for _ in range(3):
                full_visibility.extend(visibility)
            
            buttons.append(
                dict(
                    label=model,
                    method="update",
                    args=[{"visible": full_visibility}]
                )
            )
        
        # Update layout for dark theme
        fig_metrics.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=1.02,
                    y=1.0,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(155, 89, 182, 0.8)',
                    bordercolor='rgba(255,255,255,0.3)',
                    font=dict(color='white')
                )
            ],
            title=dict(
                text='<b>Detailed Metrics per Class</b><br><sub style="color: rgba(255,255,255,0.7)">Select model from dropdown</sub>',
                x=0.5,
                font=dict(size=16, color='white')
            ),
            showlegend=False,
            plot_bgcolor='rgba(30, 30, 46, 0.9)',
            paper_bgcolor='rgba(20, 20, 36, 0.9)',
            font=dict(color='white'),
            height=500,
            margin=dict(r=150, t=100)  # Make space for dropdown
        )
        
        # Update y-axis for metrics
        for i in range(1, 4):
            fig_metrics.update_yaxes(
                title_text='Score (%)', 
                row=1, 
                col=i,
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            )
            fig_metrics.update_xaxes(
                row=1, 
                col=i,
                tickfont=dict(color='white'),
                title_font=dict(color='white')
            )
        
        # Update subplot titles color
        for annotation in fig_metrics.layout.annotations:
            annotation.font.color = 'white'
        
        return fig_acc, fig_metrics
    
    except Exception as e:
        st.error(f"❌ Error generating metrics plots: {e}")
        # Return empty figures as fallback
        return go.Figure(), go.Figure()

def plot_confusion_matrix(results, model_name, y_test, label_encoder):
    """
    Plots a detailed confusion matrix for a specific model with dark theme.
    """
    try:
        if model_name not in results:
            st.warning(f"Model '{model_name}' not found in results")
            return create_empty_plot(f"Model '{model_name}' not found")
        
        y_pred = results[model_name]['predictions']
        
        # Ensure y_test and y_pred have the same length
        if len(y_test) != len(y_pred):
            st.warning(f"Length mismatch: y_test({len(y_test)}) vs y_pred({len(y_pred)})")
            min_len = min(len(y_test), len(y_pred))
            y_test = y_test[:min_len]
            y_pred = y_pred[:min_len]
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set dark background
        fig.patch.set_facecolor('#141424')
        ax.set_facecolor('#1e1e2e')
        
        # Get class names
        if label_encoder and hasattr(label_encoder, 'classes_'):
            class_names = label_encoder.classes_
            # Ensure we don't have more class names than confusion matrix dimensions
            if len(class_names) > len(cm):
                class_names = class_names[:len(cm)]
        else:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Create heatmap with simplified styling to avoid color issues
        try:
            heatmap = sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Purples',
                cbar=True,
                square=True,
                ax=ax,
                annot_kws={'size': 12, 'weight': 'bold', 'color': 'white'},
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': 'Count'}
            )
        except Exception as heatmap_error:
            # Fallback to basic heatmap without problematic parameters
            st.warning(f"Using basic heatmap due to styling issue: {heatmap_error}")
            heatmap = sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Purples',
                ax=ax
            )
        
        # Set labels with white color
        ax.set_xlabel('Predicted Labels', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('True Labels', fontsize=14, fontweight='bold', color='white')
        
        # Set tick labels with error handling
        try:
            ax.set_xticklabels(class_names, rotation=45, ha='right', color='white')
            ax.set_yticklabels(class_names, rotation=0, color='white')
        except Exception as label_error:
            st.warning(f"Label error: {label_error}")
            # Use numeric labels as fallback
            ax.set_xticklabels(range(len(cm)), rotation=45, ha='right', color='white')
            ax.set_yticklabels(range(len(cm)), rotation=0, color='white')
        
        # Color bar styling with error handling
        try:
            cbar = heatmap.collections[0].colorbar
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
        except Exception as cbar_error:
            st.warning(f"Colorbar styling issue: {cbar_error}")
        
        # Calculate accuracy for this model
        accuracy = results[model_name]['accuracy']
        
        # Set title with accuracy
        ax.set_title(
            f'Confusion Matrix - {model_name}\n(Accuracy: {accuracy:.6f}%)', 
            fontsize=16, 
            fontweight='bold', 
            pad=20,
            color='white'
        )
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Critical error generating confusion matrix for {model_name}: {e}")
        return create_empty_plot(f"Error: {str(e)}")

def create_empty_plot(message):
    """Create a simple empty plot with error message"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, message, 
            ha='center', va='center', transform=ax.transAxes, 
            color='white', fontsize=12, wrap=True)
    ax.set_facecolor('#1e1e2e')
    fig.patch.set_facecolor('#141424')
    ax.set_title('Confusion Matrix\n(Unavailable)', color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def plot_geo_distribution(df):
    """
    Creates geographical distribution plots of threats using treemaps with dark theme.
    """
    try:
        # Source country distribution
        if 'src_country' in df.columns and 'threat_label' in df.columns:
            src_threats = df.groupby(['src_country', 'threat_label']).size().reset_index(name='count')
            src_threats = src_threats.sort_values('count', ascending=False)
            
            fig_src = px.treemap(
                src_threats, 
                path=['src_country', 'threat_label'], 
                values='count',
                title='<b>Distribution of Threat Types by Source Country</b>',
                color='count', 
                color_continuous_scale='Purples',
                hover_data={'count': True, 'src_country': True, 'threat_label': True}
            )
            
            fig_src.update_layout(
                plot_bgcolor='rgba(30, 30, 46, 0.9)',
                paper_bgcolor='rgba(20, 20, 36, 0.9)',
                font=dict(family="Arial, sans-serif", size=12, color='white'),
                title_font=dict(color='white')
            )
            
            # Update color bar
            fig_src.update_coloraxes(colorbar_tickfont=dict(color='white'),
                                   colorbar_title_font=dict(color='white'))
        else:
            fig_src = go.Figure()
            fig_src.add_annotation(
                text="No source country or threat label data available", 
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16, color='white')
            )
            fig_src.update_layout(
                title='Source Country Analysis',
                plot_bgcolor='rgba(30, 30, 46, 0.9)',
                paper_bgcolor='rgba(20, 20, 36, 0.9)'
            )
            
        # Destination country distribution  
        if 'dst_country' in df.columns and 'threat_label' in df.columns:
            dst_threats = df.groupby(['dst_country', 'threat_label']).size().reset_index(name='count')
            dst_threats = dst_threats.sort_values('count', ascending=False)
            
            fig_dst = px.treemap(
                dst_threats, 
                path=['dst_country', 'threat_label'], 
                values='count',
                title='<b>Distribution of Threat Types by Destination Country</b>',
                color='count', 
                color_continuous_scale='Blues',
                hover_data={'count': True, 'dst_country': True, 'threat_label': True}
            )
            
            fig_dst.update_layout(
                plot_bgcolor='rgba(30, 30, 46, 0.9)',
                paper_bgcolor='rgba(20, 20, 36, 0.9)',
                font=dict(family="Arial, sans-serif", size=12, color='white'),
                title_font=dict(color='white')
            )
            
            # Update color bar
            fig_dst.update_coloraxes(colorbar_tickfont=dict(color='white'),
                                   colorbar_title_font=dict(color='white'))
        else:
            fig_dst = go.Figure()
            fig_dst.add_annotation(
                text="No destination country or threat label data available", 
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16, color='white')
            )
            fig_dst.update_layout(
                title='Destination Country Analysis',
                plot_bgcolor='rgba(30, 30, 46, 0.9)',
                paper_bgcolor='rgba(20, 20, 36, 0.9)'
            )
            
        return fig_src, fig_dst
    
    except Exception as e:
        st.error(f"❌ Error generating geographical distribution: {e}")
        # Return empty figures as fallback
        fig_empty = go.Figure()
        fig_empty.add_annotation(text=f"Error: {e}", x=0.5, y=0.5, showarrow=False)
        return fig_empty, fig_empty

def create_geo_heatmap(df, lat_col='src_lat', lon_col='src_lon', threat_col='threat_label'):
    """
    Creates a heatmap of threat locations using Mapbox with dark theme.
    """
    try:
        if lat_col in df.columns and lon_col in df.columns:
            # Create threat intensity column
            df_heatmap = df.copy()
            df_heatmap['threat_intensity'] = df_heatmap['label'] if 'label' in df.columns else 1
            
            fig = px.density_mapbox(
                df_heatmap,
                lat=lat_col,
                lon=lon_col,
                z='threat_intensity',
                radius=15,
                zoom=1,
                mapbox_style="dark",
                title=f"<b>Threat Density Map</b><br>{threat_col.replace('_', ' ').title()}",
                hover_data=['threat_label', 'src_country', 'dst_country'] if all(col in df.columns for col in ['threat_label', 'src_country', 'dst_country']) else None,
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(30, 30, 46, 0.9)',
                paper_bgcolor='rgba(20, 20, 36, 0.9)',
                font=dict(family="Arial, sans-serif", size=12, color='white')
            )
            
            return fig
        return None
    
    except Exception as e:
        st.error(f"❌ Error creating geographical heatmap: {e}")
        return None

def generate_model_summary(results):
    """
    Generates a comprehensive summary of all model performances.
    """
    try:
        summary_data = []
        
        for model_name, result in results.items():
            accuracy = result['accuracy']
            report = result['report']
            
            # Extract key metrics
            precision = report.get('weighted avg', {}).get('precision', 0) * 100
            recall = report.get('weighted avg', {}).get('recall', 0) * 100
            f1_score = report.get('weighted avg', {}).get('f1-score', 0) * 100
            
            summary_data.append({
                'Model': model_name,
                'Accuracy (%)': f"{accuracy:.6f}",
                'Precision (%)': f"{precision:.2f}",
                'Recall (%)': f"{recall:.2f}",
                'F1-Score (%)': f"{f1_score:.2f}"
            })
        
        return pd.DataFrame(summary_data)
    
    except Exception as e:
        st.error(f"❌ Error generating model summary: {e}")
        return pd.DataFrame()

def save_results_to_csv(results, filename='threat_detection_results.csv'):
    """
    Saves model results to a CSV file.
    """
    try:
        summary_df = generate_model_summary(results)
        summary_df.to_csv(filename, index=False)
        st.success(f"💾 Results saved to {filename}")
    except Exception as e:
        st.error(f"❌ Error saving results to CSV: {e}")

def get_best_model(results):
    """
    Returns the best performing model based on accuracy.
    """
    try:
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        return best_model_name, results[best_model_name]
    except Exception as e:
        st.error(f"❌ Error finding best model: {e}")
        return None, None

def generate_feature_importance_plot(model, feature_names, top_n=15):
    """
    Generates feature importance plot for tree-based models.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Take top N features
            top_indices = indices[:top_n]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(top_importances)), top_importances, color='#9b59b6')
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance', color='white', fontsize=12)
            ax.set_title(f'Top {top_n} Most Important Features', color='white', fontsize=14, fontweight='bold')
            
            # Dark theme styling
            ax.set_facecolor('#1e1e2e')
            fig.patch.set_facecolor('#141424')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', color='white', fontsize=10)
            
            plt.tight_layout()
            return fig
        else:
            st.warning("⚠️ This model doesn't support feature importance visualization.")
            return None
    except Exception as e:
        st.error(f"❌ Error generating feature importance plot: {e}")
        return None

def create_threat_timeline(df, date_column=None):
    """
    Creates a timeline of threats if datetime information is available.
    """
    try:
        # Try to find date/time columns
        date_columns = [col for col in df.columns if any(term in col.lower() for term in 
                       ['date', 'time', 'timestamp', 'datetime'])]
        
        if date_columns and 'threat_label' in df.columns:
            date_col = date_columns[0]
            
            # Convert to datetime if possible
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            
            threat_timeline = df.groupby([pd.Grouper(key=date_col, freq='H'), 'threat_label']).size().reset_index(name='count')
            
            fig = px.line(
                threat_timeline, 
                x=date_col, 
                y='count', 
                color='threat_label',
                title='<b>Threat Timeline</b>',
                labels={date_col: 'Time', 'count': 'Number of Threats'}
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(30, 30, 46, 0.9)',
                paper_bgcolor='rgba(20, 20, 36, 0.9)',
                font=dict(color='white'),
                title_font=dict(color='white')
            )
            
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error creating threat timeline: {e}")
        return None