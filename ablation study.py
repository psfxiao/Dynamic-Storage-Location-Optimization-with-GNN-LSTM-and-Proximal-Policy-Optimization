import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import gymnasium as gym
from gymnasium import spaces
import random
from datetime import datetime, timedelta
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
import math
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==================== USER CONFIGURATION ====================
# Change this variable to run a different model:
# "PPO_only" , "PPO_GNN" , "PPO_LSTM" , "Full"
SELECTED_MODEL = "PPO_only"  # <--- CHANGE THIS FOR EACH RUN
# ============================================================

# ==================== Column name constants ====================
COL_ORDER_NUMBER = 'order number'
COL_FABRIC_TOTAL = 'fabric total outbound volume (unit:volume)'
COL_SKU = 'SKU encodes'
COL_SKU_VOLUMES = 'the outbound quantity of each SKU'
COL_ORDER_CATEGORY = 'Order category'
COL_ORDER_DATE = 'order date'
COL_ORDER_TIME = 'order time'
CAT_OUTOFTOWN = 'outof-town order'
CAT_INTOWN = 'in-town order'

COL_OVERALL_RANK = 'Overall Rank'
COL_SKU_W = 'SKU'
COL_LOCATION = 'storage location code'
COL_DISTANCE = "storage location's Manhattan distance"
COL_FULLYEAR_RANK = 'Full-Year Rank'
COL_SPRINGSUMMER_RANK = 'Spring-Summer Rank'
COL_AUTUMNWINTER_RANK = 'Autumn-Winter Rank'

COL_NEED_ADJUST = 'need_adjust'
COL_HOT = 'hot_sku'
COL_SKU_ID = 'sku_id'
COL_RANK_DIFF = 'rank_diff'
COL_TOTAL_OUTBOUND = 'total_outbound'


def normalize_sku(sku):
    sku = str(sku).strip()
    if sku.isdigit():
        return str(int(sku))
    return sku


# ==================== Data Loaders ====================
class SalesDataLoader:
    def __init__(self, order_file_path):
        self.order_df = pd.read_excel(order_file_path)
        print("Order columns:", self.order_df.columns.tolist())
        self._preprocess_data()
        self._process_sku_sales()

    def _preprocess_data(self):
        required_cols = [COL_ORDER_DATE, COL_ORDER_TIME, COL_SKU, COL_SKU_VOLUMES, COL_ORDER_CATEGORY]
        missing = [c for c in required_cols if c not in self.order_df.columns]
        if missing:
            raise ValueError(f"Missing columns in order data: {missing}. "
                             f"Available columns: {self.order_df.columns.tolist()}")
        self.order_df[COL_ORDER_DATE] = pd.to_datetime(self.order_df[COL_ORDER_DATE], errors='coerce')
        self.order_df['order_time'] = pd.to_datetime(
            self.order_df[COL_ORDER_DATE].dt.strftime('%Y-%m-%d') + ' ' + self.order_df[COL_ORDER_TIME].astype(str),
            errors='coerce'
        )
        if self.order_df['order_time'].isna().any():
            print(f"Warning: {self.order_df['order_time'].isna().sum()} invalid date-time rows dropped")
            self.order_df = self.order_df.dropna(subset=['order_time'])
        self.order_df['sku_list'] = self.order_df[COL_SKU].apply(
            lambda x: [normalize_sku(s) for s in str(x).split(',')]
        )
        self.order_df['volume_list'] = self.order_df[COL_SKU_VOLUMES].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[')
                      else [int(v.strip()) for v in str(x).split(',')]
        )
        self.sku_sales = {}

    def _process_sku_sales(self):
        for _, row in self.order_df.iterrows():
            time = row['order_time']
            skus = row['sku_list']
            volumes = row['volume_list']
            order_type = row[COL_ORDER_CATEGORY]
            if len(skus) != len(volumes):
                continue
            for sku, volume in zip(skus, volumes):
                sku = normalize_sku(sku)
                if sku not in self.sku_sales:
                    self.sku_sales[sku] = {'times': [], 'volumes': [], 'types': []}
                self.sku_sales[sku]['times'].append(time)
                self.sku_sales[sku]['volumes'].append(volume)
                self.sku_sales[sku]['types'].append(order_type)
        for sku in self.sku_sales:
            times = self.sku_sales[sku]['times']
            volumes = self.sku_sales[sku]['volumes']
            types = self.sku_sales[sku]['types']
            sorted_indices = np.argsort(times)
            self.sku_sales[sku]['times'] = [times[i] for i in sorted_indices]
            self.sku_sales[sku]['volumes'] = [volumes[i] for i in sorted_indices]
            self.sku_sales[sku]['types'] = [types[i] for i in sorted_indices]

    def get_sales_features(self, sku_id, current_time, seq_length=5):
        features = np.zeros((seq_length, 3))
        sku_id = normalize_sku(sku_id)
        if sku_id not in self.sku_sales:
            return features
        sku_data = self.sku_sales[sku_id]
        times = sku_data['times']
        volumes = sku_data['volumes']
        types = sku_data['types']
        past_orders = [(t, v, ty) for t, v, ty in zip(times, volumes, types) if t < current_time]
        recent_orders = past_orders[-seq_length:]
        time_diffs = [0]
        for i in range(1, len(recent_orders)):
            delta = (recent_orders[i][0] - recent_orders[i - 1][0]).total_seconds() / 3600
            time_diffs.append(delta)
        start_idx = seq_length - len(recent_orders)
        for i, ((dt, vol, ty), diff) in enumerate(zip(recent_orders, time_diffs)):
            features[start_idx + i, 0] = vol
            features[start_idx + i, 1] = diff
            features[start_idx + i, 2] = 1 if ty == CAT_OUTOFTOWN else 0
        return features


class WarehouseDataLoader:
    def __init__(self, file_path, sales_data_loader):
        self.df = pd.read_excel(file_path)
        print("Warehouse columns:", self.df.columns.tolist())
        self.sku_count = len(self.df)
        self.sales_data_loader = sales_data_loader
        self._validate_data()
        self._preprocess_data()
        self._build_sku_mapping()
        print("WarehouseDataLoader initialized")

    def _validate_data(self):
        required = {COL_OVERALL_RANK, COL_SKU_W, COL_LOCATION, COL_DISTANCE,
                    COL_FULLYEAR_RANK, COL_SPRINGSUMMER_RANK, COL_AUTUMNWINTER_RANK}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in warehouse data: {missing}. "
                             f"Available: {self.df.columns.tolist()}")

    def _preprocess_data(self):
        self.df[COL_SKU_W] = self.df[COL_SKU_W].astype(str).str.strip()
        self.df[COL_SKU_W] = self.df[COL_SKU_W].apply(
            lambda x: str(int(x)) if x.isdigit() else x
        )
        self.df[COL_RANK_DIFF] = self.df[COL_OVERALL_RANK] - self.df[COL_SPRINGSUMMER_RANK]
        self.df[COL_NEED_ADJUST] = np.where(np.abs(self.df[COL_RANK_DIFF]) >= 2000, 1, 0)
        top_k = int(0.2 * len(self.df))
        self.df[COL_HOT] = 0
        self.df.loc[self.df[COL_SPRINGSUMMER_RANK].rank() <= top_k, COL_HOT] = 1
        self.df[COL_SKU_ID] = range(len(self.df))

        sku_total = {}
        for sku, data in self.sales_data_loader.sku_sales.items():
            sku_total[sku] = sum(data['volumes'])
        self.df[COL_TOTAL_OUTBOUND] = self.df[COL_SKU_W].map(sku_total).fillna(0)

        print(f"[DEBUG] Total outbound sum: {self.df[COL_TOTAL_OUTBOUND].sum()}")
        print(f"[DEBUG] Non-zero outbound count: {(self.df[COL_TOTAL_OUTBOUND] > 0).sum()}")
        if self.df[COL_TOTAL_OUTBOUND].sum() == 0:
            print("[WARNING] Total outbound is zero, using random positive values for testing.")
            self.df[COL_TOTAL_OUTBOUND] = np.random.randint(1, 100, len(self.df))
        print(f"Preprocessing done, SKUs needing adjustment: {self.df[COL_NEED_ADJUST].sum()}")

    def _build_sku_mapping(self):
        self.sku_to_location = dict(zip(self.df[COL_SKU_W], self.df[COL_LOCATION]))
        self.location_to_sku = dict(zip(self.df[COL_LOCATION], self.df[COL_SKU_W]))
        self.rank_mapping = dict(zip(self.df[COL_SKU_W], self.df[COL_SPRINGSUMMER_RANK]))
        self.distance_mapping = dict(zip(self.df[COL_LOCATION], self.df[COL_DISTANCE]))
        self.sku_to_id = dict(zip(self.df[COL_SKU_W], self.df[COL_SKU_ID]))
        self.id_to_sku = dict(zip(self.df[COL_SKU_ID], self.df[COL_SKU_W]))
        self.sku_to_hot = dict(zip(self.df[COL_SKU_W], self.df[COL_HOT]))
        print("SKU mappings built")

    def get_adjacency_matrix(self):
        n = self.sku_count
        adj_matrix = np.zeros((n, n))
        sku_to_index = {sku: idx for idx, sku in enumerate(self.df[COL_SKU_W])}
        for _, row in self.sales_data_loader.order_df.iterrows():
            skus = row['sku_list']
            for i in range(len(skus)):
                for j in range(i + 1, len(skus)):
                    if skus[i] in sku_to_index and skus[j] in sku_to_index:
                        idx_i = sku_to_index[skus[i]]
                        idx_j = sku_to_index[skus[j]]
                        adj_matrix[idx_i, idx_j] += 1
                        adj_matrix[idx_j, idx_i] += 1
        max_val = np.max(adj_matrix)
        if max_val > 0:
            adj_matrix = adj_matrix / max_val
        else:
            adj_matrix = np.random.rand(n, n) * 0.1
        hot_skus = [i for i, sku in enumerate(self.df[COL_SKU_W]) if self.sku_to_hot[sku]]
        for i in hot_skus:
            for j in hot_skus:
                if i != j:
                    adj_matrix[i, j] += 0.5
        return adj_matrix

    def get_initial_state(self):
        state = self.df[[COL_OVERALL_RANK, COL_DISTANCE, COL_SPRINGSUMMER_RANK, COL_NEED_ADJUST, COL_HOT]].values
        state[:, 0] = (state[:, 0] - np.min(state[:, 0])) / (np.max(state[:, 0]) - np.min(state[:, 0]) + 1e-8)
        state[:, 1] = (state[:, 1] - np.min(state[:, 1])) / (np.max(state[:, 1]) - np.min(state[:, 1]) + 1e-8)
        state[:, 2] = (state[:, 2] - np.min(state[:, 2])) / (np.max(state[:, 2]) - np.min(state[:, 2]) + 1e-8)
        return state.astype(np.float32)

    def get_lstm_data(self, current_time):
        lstm_data = np.zeros((len(self.df), 5, 3))
        for idx, row in self.df.iterrows():
            sku_id = row[COL_SKU_W]
            features = self.sales_data_loader.get_sales_features(sku_id, current_time, seq_length=5)
            lstm_data[idx] = features
        return lstm_data


# ==================== Neural Network Components ====================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1)))
        return torch.matmul(attention, V)


class WarehouseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.attention = AttentionLayer(output_dim)
        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x.size(1) != self.conv1.in_channels:
            x = x[:, :self.conv1.in_channels]
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.attention(x)
        return x


class SalesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(torch.cat((h_n[-2], h_n[-1]), dim=-1))


# ==================== Ablation Policies ====================
class PPOOnlyPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden=64):
        super().__init__()
        self.device = device
        self.num_skus, self.feat_dim = state_dim
        self.sku_mlp = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value_mlp = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.action_dim = action_dim

    def forward(self, states, lstm_datas):
        batch_size, num_skus, feat_dim = states.shape
        sku_logits = self.sku_mlp(states.view(-1, feat_dim)).view(batch_size, num_skus)
        global_feat = states.mean(dim=1)
        values = self.value_mlp(global_feat).squeeze(-1)
        return sku_logits, values

    def select_action(self, state, lstm_data, valid_actions_idx):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.forward(state_tensor, None)
            valid_logits = logits[0, valid_actions_idx]
            probs = torch.softmax(valid_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def log_prob(self, states, lstm_datas, actions, valid_actions_idx):
        logits, _ = self.forward(states, lstm_datas)
        valid_logits = logits[:, valid_actions_idx]
        log_probs = torch.log_softmax(valid_logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze()

    def value(self, states, lstm_datas):
        _, values = self.forward(states, lstm_datas)
        return values


class PPOGNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, edge_index, device, hidden=64):
        super().__init__()
        self.device = device
        self.num_skus, self.feat_dim = state_dim
        self.edge_index = edge_index.to(device)
        self.gnn = WarehouseGNN(self.feat_dim, hidden, hidden)
        self.sku_fc = nn.Linear(hidden, 1)
        self.value_fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.action_dim = action_dim

    def forward(self, states, lstm_datas):
        batch_size, num_skus, feat_dim = states.shape
        x = states.view(-1, feat_dim)
        data = Data(x=x, edge_index=self.edge_index)
        gnn_outs = []
        for i in range(batch_size):
            data = Data(x=states[i], edge_index=self.edge_index)
            gnn_outs.append(self.gnn(data))
        gnn_out = torch.stack(gnn_outs, dim=0)

        sku_logits = self.sku_fc(gnn_out).squeeze(-1)
        global_feat = gnn_out.mean(dim=1)
        values = self.value_fc(global_feat).squeeze(-1)
        return sku_logits, values

    def select_action(self, state, lstm_data, valid_actions_idx):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.forward(state_tensor, None)
            valid_logits = logits[0, valid_actions_idx]
            probs = torch.softmax(valid_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def log_prob(self, states, lstm_datas, actions, valid_actions_idx):
        logits, _ = self.forward(states, lstm_datas)
        valid_logits = logits[:, valid_actions_idx]
        log_probs = torch.log_softmax(valid_logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze()

    def value(self, states, lstm_datas):
        _, values = self.forward(states, lstm_datas)
        return values


class PPOLSTMPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden=64, lstm_hidden=32):
        super().__init__()
        self.device = device
        self.num_skus, self.feat_dim = state_dim
        self.lstm = SalesLSTM(input_dim=3, hidden_dim=lstm_hidden, output_dim=hidden)
        self.sku_fc = nn.Linear(hidden, 1)
        self.value_fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.action_dim = action_dim

    def forward(self, states, lstm_datas):
        batch_size, num_skus, seq_len, input_dim = lstm_datas.shape
        lstm_input = lstm_datas.view(batch_size * num_skus, seq_len, input_dim)
        lstm_out = self.lstm(lstm_input).view(batch_size, num_skus, -1)
        sku_logits = self.sku_fc(lstm_out).squeeze(-1)
        global_feat = lstm_out.mean(dim=1)
        values = self.value_fc(global_feat).squeeze(-1)
        return sku_logits, values

    def select_action(self, state, lstm_data, valid_actions_idx):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.forward(state_tensor, lstm_tensor)
            valid_logits = logits[0, valid_actions_idx]
            probs = torch.softmax(valid_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def log_prob(self, states, lstm_datas, actions, valid_actions_idx):
        logits, _ = self.forward(states, lstm_datas)
        valid_logits = logits[:, valid_actions_idx]
        log_probs = torch.log_softmax(valid_logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze()

    def value(self, states, lstm_datas):
        _, values = self.forward(states, lstm_datas)
        return values


class FullPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, edge_index, device, hidden=64):
        super().__init__()
        self.device = device
        self.num_skus, self.feat_dim = state_dim
        self.edge_index = edge_index.to(device)


        self.gnn = WarehouseGNN(self.feat_dim, hidden // 2, hidden // 2)
        self.lstm = SalesLSTM(input_dim=3, hidden_dim=hidden // 4, output_dim=hidden // 2)


        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32))
        self.fc_policy = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, action_dim))
        self.fc_value = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        self.action_dim = action_dim

    def forward(self, states, lstm_datas):
        batch_size, num_skus, feat_dim = states.shape
        gnn_outs = []
        for i in range(batch_size):
            data = Data(x=states[i], edge_index=self.edge_index)
            gnn_outs.append(self.gnn(data))
        gnn_out = torch.stack(gnn_outs, dim=0)

        if lstm_datas is not None:
            _, _, seq_len, input_dim = lstm_datas.shape
            lstm_input = lstm_datas.view(batch_size * num_skus, seq_len, input_dim)
            lstm_out = self.lstm(lstm_input).view(batch_size, num_skus, -1)
        else:
            lstm_out = torch.zeros_like(gnn_out)

        gnn_agg = gnn_out.mean(dim=1, keepdim=True)
        lstm_agg = lstm_out.mean(dim=1, keepdim=True)
        fused = torch.cat([gnn_agg, lstm_agg], dim=-1).squeeze(1)
        fused = self.fc(fused)

        action_logits = self.fc_policy(fused)
        values = self.fc_value(fused).squeeze(-1)
        return action_logits, values

    def select_action(self, state, lstm_data, valid_actions_idx=None):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.forward(state_tensor, lstm_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def log_prob(self, states, lstm_datas, actions, valid_actions_idx=None):
        logits, _ = self.forward(states, lstm_datas)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze()

    def value(self, states, lstm_datas):
        _, values = self.forward(states, lstm_datas)
        return values


# ==================== Environment ====================
class WarehouseEnv(gym.Env):
    def __init__(self, data_loader,max_steps=350):
        super().__init__()
        self.data_loader = data_loader
        self.max_steps = max_steps
        self.current_step = 0
        self.adjustment_count = 0
        self.max_adjustments = int(len(data_loader.df) * 0.3)
        self.original_adjust_flags = data_loader.df[COL_NEED_ADJUST].values
        self.valid_actions = np.where(self.original_adjust_flags == 1)[0]
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data_loader.df), 5))
        self.state = None
        self.lstm_data = None
        self.original_cost = None
        self.current_cost = None
        self.adjustment_history = []
        self.distance_mapping = dict(zip(data_loader.df[COL_LOCATION], data_loader.df[COL_DISTANCE]))
        self.sku_to_position = dict(zip(data_loader.df[COL_SKU_W], data_loader.df[COL_LOCATION]))
        self.position_to_sku = dict(zip(data_loader.df[COL_LOCATION], data_loader.df[COL_SKU_W]))
        self.original_efficiency = None
        self.current_efficiency = None
        self._precompute_optimal_positions()

    def _precompute_optimal_positions(self):
        sorted_skus = self.data_loader.df.sort_values(COL_SPRINGSUMMER_RANK)[COL_SKU_W].values
        sorted_locations = self.data_loader.df.sort_values(COL_DISTANCE)[COL_LOCATION].values
        self.optimal_assignment = {}
        for i, sku in enumerate(sorted_skus):
            if i < len(sorted_locations):
                self.optimal_assignment[sku] = sorted_locations[i]
            else:
                self.optimal_assignment[sku] = self.sku_to_position[sku]

    def reset(self, seed=None, options=None):
        #super().reset(seed=seed)
        self.current_step = 0
        self.adjustment_count = 0
        self.adjustment_history = []
        self.sku_to_position = dict(zip(self.data_loader.df[COL_SKU_W], self.data_loader.df[COL_LOCATION]))
        self.position_to_sku = dict(zip(self.data_loader.df[COL_LOCATION], self.data_loader.df[COL_SKU_W]))
        base_state = self.data_loader.get_initial_state()
        self.state = base_state.astype(np.float32)
        self.original_cost = self._calculate_total_cost()
        self.current_cost = self.original_cost
        self.original_efficiency = self._calculate_overall_efficiency()
        self.current_efficiency = self.original_efficiency
        current_time = datetime.now()
        self.lstm_data = self.data_loader.get_lstm_data(current_time)
        return self.state, self.lstm_data

    def _calculate_total_cost(self, state=None):
        if state is not None:
            max_dist = max(self.distance_mapping.values())
            distances = state[:, 1] * max_dist
        else:
            distances = np.array([self.distance_mapping[self.sku_to_position[sku]]
                                  for sku in self.data_loader.df[COL_SKU_W]])
        out_volumes = self.data_loader.df[COL_TOTAL_OUTBOUND].values
        return np.sum(out_volumes * distances)

    def _calculate_overall_efficiency(self, state=None):
        if state is not None:
            max_dist = max(self.distance_mapping.values())
            distances = np.clip(state[:, 1] * max_dist, 1, max_dist)
            ranks = np.clip(state[:, 2] * len(self.data_loader.df), 1, len(self.data_loader.df))
            rank_weights = 1 / np.sqrt(ranks)
            hot_weights = np.where(state[:, 4] > 0.5, 1.5, 1.0)
            efficiency = np.sum(hot_weights * rank_weights / np.sqrt(distances))
            return efficiency / len(self.data_loader.df) * 10
        else:
            total = 0
            max_dist = max(self.distance_mapping.values())
            for sku in self.data_loader.df[COL_SKU_W]:
                dist = self.distance_mapping[self.sku_to_position[sku]]
                rank = self.data_loader.rank_mapping[sku]
                is_hot = self.data_loader.sku_to_hot.get(sku, False)
                weight = (1.5 if is_hot else 1.0) / math.sqrt(max(1, rank))
                total += weight / math.sqrt(max(1, dist))
            return total / len(self.data_loader.df) * 10

    def _calculate_reward(self, sku1, sku2, old_efficiency, new_efficiency):
        eff_gain = np.clip((new_efficiency - old_efficiency) / (old_efficiency + 1e-8), -0.1, 0.1) * 5
        sku1_dist = self.distance_mapping[self.sku_to_position[sku1]]
        sku2_dist = self.distance_mapping[self.sku_to_position[sku2]]
        dist_reward = 0
        if self.data_loader.rank_mapping[sku1] < self.data_loader.rank_mapping[sku2]:
            dist_reward = (sku1_dist - sku2_dist) * 0.01
        hot_reward = 0
        if self.data_loader.sku_to_hot.get(sku1, False):
            hot_reward += 0.5 - min(sku2_dist, 50) * 0.01
        adjust_penalty = -0.05
        total = eff_gain + dist_reward + hot_reward + adjust_penalty
        return np.clip(total, -0.5, 1.0)

    def _swap_physical_positions(self, sku1, sku2):
        pos1 = self.sku_to_position[sku1]
        pos2 = self.sku_to_position[sku2]
        self.sku_to_position[sku1] = pos2
        self.sku_to_position[sku2] = pos1
        self.position_to_sku[pos1] = sku2
        self.position_to_sku[pos2] = sku1

    def _update_state_features(self, idx1, idx2):
        pos1 = self.sku_to_position[self.data_loader.id_to_sku[idx1]]
        pos2 = self.sku_to_position[self.data_loader.id_to_sku[idx2]]
        dist1 = self.distance_mapping[pos1]
        dist2 = self.distance_mapping[pos2]
        max_dist = max(self.distance_mapping.values())
        self.state[idx1, 1] = dist1 / max_dist
        self.state[idx2, 1] = dist2 / max_dist
        self.state[idx1, 3] = 0
        self.state[idx2, 3] = 0

    def _find_best_candidates(self, sku):
        current_pos = self.sku_to_position[sku]
        current_dist = self.distance_mapping[current_pos]
        sku_rank = self.data_loader.rank_mapping[sku]
        candidates = []
        for other_sku in self.data_loader.df[COL_SKU_W]:
            if other_sku == sku:
                continue
            other_pos = self.sku_to_position[other_sku]
            other_dist = self.distance_mapping[other_pos]
            other_rank = self.data_loader.rank_mapping[other_sku]
            if (sku_rank < other_rank and current_dist > other_dist) or \
               (self.data_loader.sku_to_hot.get(sku, False) and not self.data_loader.sku_to_hot.get(other_sku, False)):
                improvement = (current_dist - other_dist) * (1.0 / math.log(other_rank + 1))
                candidates.append((other_sku, improvement))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in candidates]

    def step(self, action):
        if self.adjustment_count >= self.max_adjustments:
            return self.state, self.lstm_data, -1, True, False, {}

        original_action = self.valid_actions[action]
        selected_sku = self.data_loader.id_to_sku[original_action]
        candidate_skus = self._find_best_candidates(selected_sku)
        if not candidate_skus:
            return self.state, self.lstm_data, -0.5, False, False, {}
        best_sku = random.choice(candidate_skus[:3])
        best_sku_idx = self.data_loader.sku_to_id[best_sku]


        old_efficiency = self.current_efficiency
        old_cost = self.current_cost

        selected_sku_old_loc = self.sku_to_position[selected_sku]
        best_sku_old_loc = self.sku_to_position[best_sku]

        self._swap_physical_positions(selected_sku, best_sku)
        self._update_state_features(original_action, best_sku_idx)


        new_efficiency = self._calculate_overall_efficiency()
        self.current_efficiency = new_efficiency

        new_cost = self._calculate_total_cost()
        self.current_cost = new_cost


        reward = self._calculate_reward(selected_sku, best_sku, old_efficiency, new_efficiency)
        # ==========================================

        self.adjustment_count += 1
        self.adjustment_history.append({'step': self.current_step, 'from_sku': selected_sku, 'to_sku': best_sku,
                                        'from_old_loc': selected_sku_old_loc, 'to_old_loc': best_sku_old_loc,
                                        'reward': reward, 'cost_change': new_cost - old_cost})
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (self.adjustment_count >= self.max_adjustments)
        current_time = datetime.now() + timedelta(days=self.current_step)
        self.lstm_data = self.data_loader.get_lstm_data(current_time)
        info = {'from_sku': selected_sku, 'to_sku': best_sku,
                'from_old_loc': selected_sku_old_loc, 'to_old_loc': best_sku_old_loc}
        return self.state, self.lstm_data, reward, done, False, info


# ==================== PPO Trainer ====================
class PPO:
    def __init__(self, policy, env, valid_actions_idx, learning_rate=1e-4, n_steps=128, gamma=0.99, clip_epsilon=0.15):
        self.policy = policy
        self.env = env
        self.valid_actions_idx = valid_actions_idx
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.scaler = GradScaler()
        self.reward_history = []
        self.cost_history = []
        self.eval_timesteps = []
        self.eval_costs = []
        self.eval_best_costs = []
        self.eval_best_effs = []
        self.update_count = 0

    def learn(self, total_timesteps):
        timesteps = 0
        episode_rewards = []
        best_efficiency = -np.inf
        best_cost_recorded = np.inf
        pbar = tqdm(total=total_timesteps, desc="Training")
        while timesteps < total_timesteps:
            states, lstm_datas, actions, rewards, next_states, next_lstm_datas, dones = [], [], [], [], [], [], []
            episode_reward = 0
            state, lstm_data = self.env.reset()
            for _ in range(self.n_steps):
                action = self.policy.select_action(state, lstm_data, self.valid_actions_idx)
                next_state, next_lstm_data, reward, done, truncated, info = self.env.step(action)
                if reward > -0.5:
                    states.append(state); lstm_datas.append(lstm_data); actions.append(action); rewards.append(reward)
                    next_states.append(next_state); next_lstm_datas.append(next_lstm_data); dones.append(done or truncated)
                state, lstm_data = next_state, next_lstm_data
                episode_reward += reward
                timesteps += 1
                pbar.update(1)
                if done or truncated or timesteps >= total_timesteps:
                    break
            if len(states) > 0:
                self.update_policy(states, lstm_datas, actions, rewards, next_states, next_lstm_datas, dones)

            self.update_count += 1


            if self.update_count % 5 == 0 or timesteps >= total_timesteps:
                current_efficiency = self.evaluate_policy()
                current_cost = self.evaluate_cost()

                best_cost_recorded = min(best_cost_recorded, current_cost)
                self.eval_timesteps.append(timesteps)
                self.eval_costs.append(current_cost)
                self.eval_best_costs.append(best_cost_recorded)
                self.cost_history.append(current_cost)

                if current_efficiency > best_efficiency:
                    best_efficiency = current_efficiency
                    torch.save(self.policy.state_dict(), f"best_model_{SELECTED_MODEL}.pth")
                self.eval_best_effs.append(best_efficiency)

            episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)
            if len(episode_rewards) % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])

                show_cost = self.eval_costs[-1] if self.eval_costs else 0
                tqdm.write(
                    f"Steps: {timesteps}, Avg Reward: {avg_reward:.2f}, Best Eff: {best_efficiency:.7f}, Current Cost: {show_cost:.2f}")

        pbar.close()
        return episode_rewards

    def evaluate_cost(self):
        state, lstm_data = self.env.reset()
        done, truncated = False, False
        step = 0
        while not done and not truncated and step < self.env.max_steps:
            action = self.policy.select_action(state, lstm_data, self.valid_actions_idx)
            state, lstm_data, _, done, truncated, _ = self.env.step(action)
            step += 1
        return self.env._calculate_total_cost(state)

    def update_policy(self, states, lstm_datas, actions, rewards, next_states, next_lstm_datas, dones):
        mini_batch_size = 8
        for _ in range(1):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                states_tensor = torch.tensor(np.array([states[i] for i in batch_idx]), dtype=torch.float32).to(self.policy.device)
                lstm_datas_tensor = torch.tensor(np.array([lstm_datas[i] for i in batch_idx]), dtype=torch.float32).to(self.policy.device)
                actions_tensor = torch.tensor([actions[i] for i in batch_idx], dtype=torch.long).to(self.policy.device)
                rewards_tensor = torch.tensor([rewards[i] for i in batch_idx], dtype=torch.float32).to(self.policy.device)
                next_states_tensor = torch.tensor(np.array([next_states[i] for i in batch_idx]), dtype=torch.float32).to(self.policy.device)
                next_lstm_datas_tensor = torch.tensor(np.array([next_lstm_datas[i] for i in batch_idx]), dtype=torch.float32).to(self.policy.device)
                dones_tensor = torch.tensor([dones[i] for i in batch_idx], dtype=torch.float32).to(self.policy.device)


                with torch.no_grad(), autocast():
                    next_values = self.policy.value(next_states_tensor, next_lstm_datas_tensor)
                    returns = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)

                old_log_probs = self.policy.log_prob(states_tensor, lstm_datas_tensor, actions_tensor, self.valid_actions_idx).detach()

                self.optimizer.zero_grad()
                with autocast():
                    values = self.policy.value(states_tensor, lstm_datas_tensor)
                    advantages = (returns - values).detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

                    new_log_probs = self.policy.log_prob(states_tensor, lstm_datas_tensor, actions_tensor, self.valid_actions_idx)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.MSELoss()(values, returns)
                    loss = policy_loss + 0.5 * value_loss

                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                del states_tensor, lstm_datas_tensor, actions_tensor, rewards_tensor
                del next_states_tensor, next_lstm_datas_tensor, dones_tensor
                torch.cuda.empty_cache()
    def evaluate_policy(self):
        total_efficiency = 0
        eval_episodes = 3
        for _ in range(eval_episodes):
            state, lstm_data = self.env.reset()
            done, truncated = False, False
            step = 0
            while not done and not truncated and step < self.env.max_steps:
                action = self.policy.select_action(state, lstm_data, self.valid_actions_idx)
                state, lstm_data, _, done, truncated, _ = self.env.step(action)
                step += 1
            total_efficiency += self.env._calculate_overall_efficiency()
        return total_efficiency / eval_episodes
# ==================== Main ====================
def run_single_model():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running model: {SELECTED_MODEL}")

    print("Loading order data...")
    sales_loader = SalesDataLoader("order data.xlsx")
    print("Loading warehouse data...")
    warehouse_loader = WarehouseDataLoader("initial storage location assignment.xlsx", sales_loader)
    print("Creating environment...")
    env = WarehouseEnv(warehouse_loader)
    valid_actions_idx = env.valid_actions
    action_dim = len(valid_actions_idx)
    state_shape = (len(warehouse_loader.df), 5)

    # Build graph
    adj_matrix = warehouse_loader.get_adjacency_matrix()
    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0.01:
                edge_list.append([i, j]); edge_list.append([j, i])
    if not edge_list:
        for i in range(0, min(100, adj_matrix.shape[0]), 2):
            j = i+1
            if j < adj_matrix.shape[0]:
                edge_list.append([i, j]); edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)

    # Select policy
    policy_map = {
        "PPO_only": PPOOnlyPolicy(state_shape, action_dim, device),
        "PPO_GNN": PPOGNPolicy(state_shape, action_dim, edge_index, device),
        "PPO_LSTM": PPOLSTMPolicy(state_shape, action_dim, device),
        "Full": FullPolicy(state_shape, action_dim, edge_index, device)
    }
    if SELECTED_MODEL not in policy_map:
        raise ValueError(f"Invalid model: {SELECTED_MODEL}")
    policy = policy_map[SELECTED_MODEL].to(device)

    print(f"Initializing PPO for {SELECTED_MODEL}...")
    ppo = PPO(policy, env, valid_actions_idx, learning_rate=1e-4, n_steps=128, gamma=0.99, clip_epsilon=0.15)

    print(f"Starting training {SELECTED_MODEL}...")
    start_time = time.perf_counter()


    total_timesteps = 10000
    rewards = ppo.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - start_time


    print("Loading best model for final evaluation...")
    ppo.policy.load_state_dict(torch.load(f"best_model_{SELECTED_MODEL}.pth", map_location=device))
    ppo.policy.eval()


    final_cost = ppo.evaluate_cost()
    best_cost = min(ppo.cost_history) if ppo.cost_history else final_cost

    print(f"\n{SELECTED_MODEL} finished!")
    print(f"Final Cost: {final_cost:.2f}")
    print(f"Best Cost: {best_cost:.2f}")
    print(f"Time: {elapsed:.2f}s")

    #  save
    df_result = pd.DataFrame([{
        'Model': SELECTED_MODEL,
        'Final_Cost': final_cost,
        'Best_Cost': best_cost,
        'Training_Time_sec': elapsed
    }])
    df_result.to_excel(f"ablation_result_{SELECTED_MODEL}.xlsx", index=False)

    #  History
    final_best_eff = ppo.eval_best_effs[-1] if ppo.eval_best_effs else 0
    history_df = pd.DataFrame({
        'timesteps': ppo.eval_timesteps + [total_timesteps],
        'eval_cost': ppo.eval_costs + [final_cost],
        'best_eff': ppo.eval_best_effs + [final_best_eff],
    })
    history_file = f"{SELECTED_MODEL}_ablation_full_history.xlsx"
    history_df.to_excel(history_file, index=False)
    print(f"Full history saved to {history_file}")
    print(f"Best Cost: {best_cost:.2f}, Final Cost: {final_cost:.2f}")

    # Save optimized layout
    max_dist = max(env.distance_mapping.values())
    optimized_df = warehouse_loader.df.copy()
    optimized_df['optimized_distance'] = env.state[:, 1] * max_dist
    optimized_df['optimized_distance_normalized'] = env.state[:, 1]
    optimized_df['final_cost_per_sku'] = optimized_df[COL_TOTAL_OUTBOUND] * optimized_df['optimized_distance']
    optimized_df['current_location'] = optimized_df[COL_LOCATION]
    optimized_df['optimized_location'] = optimized_df[COL_LOCATION]
    optimized_df['optimized_location_mapped'] = optimized_df[COL_SKU_W].map(env.sku_to_position)
    optimized_df.to_excel(f"optimized_layout_{SELECTED_MODEL}.xlsx", index=False)
    print(f"Optimized layout saved to optimized_layout_{SELECTED_MODEL}.xlsx")

    print("\nAll tasks completed.")


if __name__ == "__main__":
    run_single_model()