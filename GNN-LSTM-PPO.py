
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import gym
from gym import spaces
import random
from datetime import datetime, timedelta
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==================== Column name constants (English) ====================
# Order data columns
COL_ORDER_DATE = 'order date'
COL_ORDER_TIME = 'order time'
COL_ORDER_NUMBER = 'order number'
COL_FABRIC_TOTAL = 'fabric total outbound volume (unit:volume)'
COL_SKU = 'SKU encodes'
COL_SKU_VOLUMES = 'the outbound quantity of each SKU'
COL_ORDER_CATEGORY = 'Order category'
CAT_OUTOFTOWN = 'outof-town order'
CAT_INTOWN = 'in-town order'

# Warehouse data columns
COL_OVERALL_RANK = 'Overall Rank'
COL_SKU_W = 'SKU'
COL_LOCATION = 'storage location code'
COL_DISTANCE = "storage location's Manhattan distance"
COL_FULLYEAR_RANK = 'Full-Year Rank'
COL_SPRINGSUMMER_RANK = 'Spring-Summer Rank'
COL_AUTUMNWINTER_RANK = 'Autumn-Winter Rank'

# Internal derived columns (created in code)
COL_SKU_LIST = 'sku_list'
COL_VOLUME_LIST = 'volume_list'
COL_RANK_DIFF = 'rank_diff'
COL_NEED_ADJUST = 'need_adjust'
COL_HOT = 'hot_sku'
COL_SKU_ID = 'sku_id'
COL_TOTAL_OUTBOUND = 'total_outbound'


# ==================== Data Loaders ====================
class SalesDataLoader:
    def __init__(self, order_file_path):
        self.order_df = pd.read_excel(order_file_path)
        self._preprocess_data()
        self._process_sku_sales()

    def _preprocess_data(self):
        print("\n[Debug] Raw data sample:")
        print(self.order_df[[COL_ORDER_DATE, COL_ORDER_TIME]].head(2).to_dict('records'))

        # Convert order date
        if COL_ORDER_DATE in self.order_df.columns:
            if any('月' in str(x) for x in self.order_df[COL_ORDER_DATE].head()):
                self.order_df[COL_ORDER_DATE] = '2025年' + self.order_df[COL_ORDER_DATE].astype(str)
                self.order_df[COL_ORDER_DATE] = pd.to_datetime(self.order_df[COL_ORDER_DATE], format='%Y年%m月%d日',
                                                               errors='coerce')
            elif pd.api.types.is_numeric_dtype(self.order_df[COL_ORDER_DATE]):
                self.order_df[COL_ORDER_DATE] = pd.to_datetime(self.order_df[COL_ORDER_DATE], unit='D',
                                                               origin='1899-12-30')
                self.order_df[COL_ORDER_DATE] = self.order_df[COL_ORDER_DATE].map(lambda x: x.replace(year=2025))
            else:
                self.order_df[COL_ORDER_DATE] = pd.to_datetime(self.order_df[COL_ORDER_DATE], errors='coerce')

        # Combine date and time
        try:
            self.order_df['order_time'] = pd.to_datetime(
                self.order_df[COL_ORDER_DATE].dt.strftime('%Y-%m-%d') + ' ' + self.order_df[COL_ORDER_TIME].astype(str),
                errors='coerce'
            )
        except Exception as e:
            print(f"Date time merge error: {e}")
            raise

        print("\n[Debug] Processed time sample:")
        print(self.order_df[[COL_ORDER_DATE, 'order_time']].head(2).to_dict('records'))

        if self.order_df['order_time'].isna().any():
            print(f"Warning: {self.order_df['order_time'].isna().sum()} invalid date-times, dropping rows")
            self.order_df = self.order_df.dropna(subset=['order_time'])


        # Add an internal cleaning function to remove spaces and convert to pure string
        def clean_sku(s):
            s = str(s).strip()
            if s.endswith('.0'):
                s = s[:-2]
            return s

        # Split and clean all SKUs in the order data, and remove leading zeros  to unify the format
        self.order_df[COL_SKU_LIST] = self.order_df[COL_SKU].apply(
            lambda x: [clean_sku(s).lstrip('0') if clean_sku(s).lstrip('0') else '0' for s in str(x).split(',')]
        )

        self.order_df[COL_VOLUME_LIST] = self.order_df[COL_SKU_VOLUMES].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [int(v.strip()) for v in
                                                                                            str(x).split(',')]
        )
        self.sku_sales = {}





    def _process_sku_sales(self):
        for _, row in self.order_df.iterrows():
            time = row['order_time']
            skus = row[COL_SKU_LIST]
            volumes = row[COL_VOLUME_LIST]
            order_type = row[COL_ORDER_CATEGORY]
            if len(skus) != len(volumes):
                continue
            for sku, volume in zip(skus, volumes):
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
        self.sku_count = len(self.df)
        self.sales_data_loader = sales_data_loader
        self._validate_data()
        self._preprocess_data()
        self._build_sku_mapping()
        print("WarehouseDataLoader initialized")

    def _validate_data(self):
        required_columns = {COL_OVERALL_RANK, COL_SKU_W, COL_LOCATION, COL_DISTANCE,
                            COL_FULLYEAR_RANK, COL_SPRINGSUMMER_RANK, COL_AUTUMNWINTER_RANK}
        assert set(self.df.columns) >= required_columns, f"Missing columns: {required_columns - set(self.df.columns)}"



    def _preprocess_data(self):
        # === Added: Standardize the SKU column in the warehouse table ===
        def clean_sku_w(s):
            s = str(s).strip()
            if s.endswith('.0'):
                s = s[:-2]
            return s.lstrip('0') if s.lstrip('0') else '0'

        self.df[COL_SKU_W] = self.df[COL_SKU_W].apply(clean_sku_w)
        # =================================
        self.df[COL_RANK_DIFF] = self.df[COL_OVERALL_RANK] - self.df[COL_SPRINGSUMMER_RANK]







        self.df[COL_NEED_ADJUST] = np.where(np.abs(self.df[COL_RANK_DIFF]) >= 2000, 1, 0)
        top_k = int(0.2 * len(self.df))
        self.df[COL_HOT] = 0
        self.df.loc[self.df[COL_SPRINGSUMMER_RANK].rank() <= top_k, COL_HOT] = 1
        self.df[COL_SKU_ID] = range(len(self.df))

        # Compute total outbound volume per SKU from sales data
        sku_total = {}
        for sku, data in self.sales_data_loader.sku_sales.items():
            sku_total[sku] = sum(data['volumes'])
        self.df[COL_TOTAL_OUTBOUND] = self.df[COL_SKU_W].map(sku_total).fillna(0)

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
            skus = row[COL_SKU_LIST]
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


# ==================== Neural Network Modules ====================
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


class FusionModel(nn.Module):
    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gnn = WarehouseGNN(gnn_input_dim, hidden_dim // 2, hidden_dim // 2)
        self.lstm = SalesLSTM(lstm_input_dim, hidden_dim // 4, hidden_dim // 2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, gnn_data, lstm_data):
        x = gnn_data.x
        if x.size(1) != self.gnn.conv1.in_channels:
            x = x[:, :self.gnn.conv1.in_channels]
        gnn_data.x = x
        gnn_out = self.gnn(gnn_data)

        batch_size, num_skus, seq_len, input_dim = lstm_data.shape
        lstm_input = lstm_data.view(batch_size * num_skus, seq_len, input_dim)
        lstm_out = self.lstm(lstm_input)
        lstm_out = lstm_out.view(batch_size, num_skus, -1)

        gnn_agg = torch.mean(gnn_out, dim=0, keepdim=True)
        lstm_agg = torch.mean(lstm_out, dim=1)
        fused = torch.cat([gnn_agg.expand_as(lstm_agg), lstm_agg], dim=1)
        return self.fc(fused)


class PolicyNetwork(nn.Module):
    def __init__(self, fusion_model, state_dim, action_dim, device):
        super().__init__()
        self.fusion_model = fusion_model
        self.device = device
        self.fc_policy = nn.Sequential(nn.Linear(fusion_model.fc[-1].out_features, 32), nn.ReLU(), nn.Linear(32, action_dim))
        self.fc_value = nn.Sequential(nn.Linear(fusion_model.fc[-1].out_features, 32), nn.ReLU(), nn.Linear(32, 1))
        self.edge_index = None

    def set_graph_data(self, edge_index):
        self.edge_index = edge_index.to(self.device)

    def forward(self, states, lstm_datas):
        batch_size = states.size(0)
        num_skus = states.size(1)

        fused_features_list = []
        with autocast():
            # Loop through each sample in the batch  to compute graph features independently
            for i in range(batch_size):
                state_i = states[i]
                lstm_data_i = lstm_datas[i:i + 1]

                gnn_data = Data(x=state_i, edge_index=self.edge_index)

                fused_feat = self.fusion_model(gnn_data, lstm_data_i)
                fused_features_list.append(fused_feat)

            fused_features = torch.cat(fused_features_list, dim=0)

            action_logits = self.fc_policy(fused_features)
            state_values = self.fc_value(fused_features)

        return action_logits, state_values




    def select_action(self, state, lstm_data):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(), autocast():
            logits, _ = self.forward(state_tensor, lstm_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def log_prob(self, states, lstm_datas, actions):
        logits, _ = self.forward(states, lstm_datas)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze()

    def value(self, states, lstm_datas):
        _, values = self.forward(states, lstm_datas)
        return values.squeeze()


# ==================== PPO Trainer ====================
class PPO:
    def __init__(self, policy, env, learning_rate=1e-4, n_steps=128, batch_size=16, gamma=0.99, clip_epsilon=0.15):
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.scaler = GradScaler()
        self.reward_history = []
        self.efficiency_history = []

    def learn(self, total_timesteps):
        timesteps = 0
        episode_rewards = []
        best_efficiency = -np.inf
        pbar = tqdm(total=total_timesteps, desc="Training Progress")
        while timesteps < total_timesteps:
            states, lstm_datas, actions, rewards, next_states, next_lstm_datas, dones = [], [], [], [], [], [], []
            episode_reward = 0
            state, lstm_data = self.env.reset()
            for _ in range(self.n_steps):
                action = self.policy.select_action(state, lstm_data)
                next_state, next_lstm_data, reward, done, _ = self.env.step(action)


                if reward > -0.5:
                    states.append(state)
                    lstm_datas.append(lstm_data)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    next_lstm_datas.append(next_lstm_data)
                    dones.append(done)


                state = next_state
                lstm_data = next_lstm_data
                episode_reward += reward
                timesteps += 1
                pbar.update(1)
                if done or timesteps >= total_timesteps:
                    break


            if len(states) > 0:
                self.update_policy(states, lstm_datas, actions, rewards, next_states, next_lstm_datas, dones)


            current_efficiency = self.evaluate_policy()
            if current_efficiency > best_efficiency:
                best_efficiency = current_efficiency
                torch.save(self.policy.state_dict(), "best_model.pth")

            episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)
            if len(episode_rewards) % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])
                tqdm.write(f"Timesteps: {timesteps}, Avg Reward: {avg_reward:.2f}, Best Eff: {best_efficiency:.7f}")
        pbar.close()
        return episode_rewards

    def evaluate_policy(self):
        total_efficiency = 0
        eval_episodes = 3
        for _ in range(eval_episodes):
            state, lstm_data = self.env.reset()
            done = False
            while not done:
                action = self.policy.select_action(state, lstm_data)
                state, lstm_data, _, done, _ = self.env.step(action)
            total_efficiency += self.env._calculate_overall_efficiency()
        return total_efficiency / eval_episodes

    def update_policy(self, states, lstm_datas, actions, rewards, next_states, next_lstm_datas, dones):
        mini_batch_size = 8
        for _ in range(1):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                states_tensor = torch.tensor(np.array([states[i] for i in batch_idx]), dtype=torch.float32)
                lstm_datas_tensor = torch.tensor(np.array([lstm_datas[i] for i in batch_idx]), dtype=torch.float32)
                actions_tensor = torch.tensor([actions[i] for i in batch_idx], dtype=torch.long)
                rewards_tensor = torch.tensor([rewards[i] for i in batch_idx], dtype=torch.float32)
                next_states_tensor = torch.tensor(np.array([next_states[i] for i in batch_idx]), dtype=torch.float32)
                next_lstm_datas_tensor = torch.tensor(np.array([next_lstm_datas[i] for i in batch_idx]), dtype=torch.float32)
                dones_tensor = torch.tensor([dones[i] for i in batch_idx], dtype=torch.float32)

                states_tensor = states_tensor.to(self.policy.device)
                lstm_datas_tensor = lstm_datas_tensor.to(self.policy.device)
                actions_tensor = actions_tensor.to(self.policy.device)
                rewards_tensor = rewards_tensor.to(self.policy.device)
                next_states_tensor = next_states_tensor.to(self.policy.device)
                next_lstm_datas_tensor = next_lstm_datas_tensor.to(self.policy.device)
                dones_tensor = dones_tensor.to(self.policy.device)


                # === Fix: value loss in the PPO update ===
                with torch.no_grad(), autocast():
                    # Only compute next_values as the target
                    next_values = self.policy.value(next_states_tensor, next_lstm_datas_tensor)
                    returns = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)

                old_log_probs = self.policy.log_prob(states_tensor, lstm_datas_tensor, actions_tensor).detach()

                self.optimizer.zero_grad()
                with autocast():
                    # Values here must retain gradients
                    values = self.policy.value(states_tensor, lstm_datas_tensor)
                    # Advantages are only used for the Actor, so detach them
                    advantages = (returns - values).detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    new_log_probs = self.policy.log_prob(states_tensor, lstm_datas_tensor, actions_tensor)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Compute MSE Loss using values with gradients
                    value_loss = nn.MSELoss()(values, returns)

                    loss = policy_loss + 0.5 * value_loss
                # === end ===



                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                del states_tensor, lstm_datas_tensor, actions_tensor, rewards_tensor
                del next_states_tensor, next_lstm_datas_tensor, dones_tensor
                torch.cuda.empty_cache()

    def plot_rewards(self, filename="training_rewards.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.reward_history, label='Reward')
        plt.plot(self.efficiency_history, label='Efficiency')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()


# ==================== Environment ====================
class WarehouseEnv(gym.Env):
    def __init__(self, data_loader, max_steps=350):
        super().__init__()
        self.data_loader = data_loader
        self.max_steps = max_steps
        self.current_step = 0
        self.adjustment_count = 0
        self.max_adjustments = int(len(data_loader.df) * 0.3)
        self.original_adjust_flags = data_loader.df[COL_NEED_ADJUST].values
        self.original_sku_ids = data_loader.df[COL_SKU_ID].values
        self.valid_actions = np.where(self.original_adjust_flags == 1)[0]
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data_loader.df), 5))
        self.state = None
        self.lstm_data = None
        self.original_efficiency = None
        self.current_efficiency = None
        self.adjustment_history = []

        self.physical_positions = data_loader.df[COL_LOCATION].values.copy()
        self.distance_mapping = dict(zip(data_loader.df[COL_LOCATION], data_loader.df[COL_DISTANCE]))
        self.sku_to_position = dict(zip(data_loader.df[COL_SKU_W], data_loader.df[COL_LOCATION]))
        self.position_to_sku = dict(zip(data_loader.df[COL_LOCATION], data_loader.df[COL_SKU_W]))
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

    def reset(self):
        self.current_step = 0
        self.adjustment_count = 0
        self.adjustment_history = []
        self.physical_positions = self.data_loader.df[COL_LOCATION].values.copy()
        self.sku_to_position = dict(zip(self.data_loader.df[COL_SKU_W], self.data_loader.df[COL_LOCATION]))
        self.position_to_sku = dict(zip(self.data_loader.df[COL_LOCATION], self.data_loader.df[COL_SKU_W]))

        base_state = self.data_loader.get_initial_state()
        self.state = base_state.astype(np.float32)
        self.original_efficiency = self._calculate_overall_efficiency()
        self.current_efficiency = self.original_efficiency
        current_time = datetime.now()
        self.lstm_data = self.data_loader.get_lstm_data(current_time)
        return self.state, self.lstm_data

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

    def step(self, action):
        if self.adjustment_count >= self.max_adjustments:
            return self.state, self.lstm_data, -1, True, {}

        original_action = self.valid_actions[action]
        selected_sku = self.data_loader.id_to_sku[original_action]

        candidate_skus = self._find_best_candidates(selected_sku)
        if not candidate_skus:
            return self.state, self.lstm_data, -0.5, False, {}

        best_sku = random.choice(candidate_skus[:3])
        best_sku_idx = self.data_loader.sku_to_id[best_sku]

        old_efficiency = self.current_efficiency
        selected_sku_old_loc = self.sku_to_position[selected_sku]
        best_sku_old_loc = self.sku_to_position[best_sku]

        self._swap_physical_positions(selected_sku, best_sku)
        self._update_state_features(original_action, best_sku_idx)
        new_efficiency = self._calculate_overall_efficiency()
        self.current_efficiency = new_efficiency

        reward = self._calculate_reward(selected_sku, best_sku, old_efficiency, new_efficiency)

        self.adjustment_count += 1
        self.adjustment_history.append({
            'step': self.current_step,
            'from_sku': selected_sku,
            'to_sku': best_sku,
            'from_old_loc': selected_sku_old_loc,
            'to_old_loc': best_sku_old_loc,
            'from_new_loc': self.sku_to_position[selected_sku],
            'to_new_loc': self.sku_to_position[best_sku],
            'from_rank': self.data_loader.rank_mapping[selected_sku],
            'to_rank': self.data_loader.rank_mapping[best_sku],
            'reward': reward,
            'efficiency_change': new_efficiency - old_efficiency
        })

        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (self.adjustment_count >= self.max_adjustments)
        current_time = datetime.now() + timedelta(days=self.current_step)
        self.lstm_data = self.data_loader.get_lstm_data(current_time)

        info = {'from_sku': selected_sku, 'to_sku': best_sku,
                'from_old_loc': selected_sku_old_loc, 'to_old_loc': best_sku_old_loc}
        return self.state, self.lstm_data, reward, done, info

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
        self.state[idx1, 1] = dist1 / max_dist # dist1 correctly belongs to idx1 after swap
        self.state[idx2, 1] = dist2 / max_dist # dist2 correctly belongs to idx2 after swap
        self.state[idx1, 3] = 0
        self.state[idx2, 3] = 0


# ==================== Application Functions ====================
def apply_dynamic_adjustment(policy_net, env, data_loader):
    state, lstm_data = env.reset()
    done = False
    adjustments = 0
    improvement_history = []


    while not done and adjustments < env.max_adjustments:
        action = policy_net.select_action(state, lstm_data)
        next_state, next_lstm_data, reward, done, info = env.step(action)

        if reward > -0.5:
            from_sku = info['from_sku']
            to_sku = info['to_sku']
            from_old_loc = info['from_old_loc']
            to_old_loc = info['to_old_loc']

            from_sku_rank = data_loader.rank_mapping[from_sku]
            to_sku_rank = data_loader.rank_mapping[to_sku]
            from_sku_dist = data_loader.distance_mapping[from_old_loc]
            to_sku_dist = data_loader.distance_mapping[to_old_loc]

            print(f"\nAdjustment {adjustments + 1}:")
            print(f"  Swap SKU-{from_sku} (rank:{from_sku_rank}, old dist:{from_sku_dist:.1f}, old loc:{from_old_loc})")
            print(f"  with SKU-{to_sku} (rank:{to_sku_rank}, old dist:{to_sku_dist:.1f}, old loc:{to_old_loc})")
            print(f"  Reward: {reward:.2f}")

            adjustments += 1
            current_efficiency = env._calculate_overall_efficiency(next_state)
            improvement = (current_efficiency - env.original_efficiency) / env.original_efficiency * 100
            improvement_history.append(improvement)

        state = next_state
        lstm_data = next_lstm_data

    final_efficiency = env._calculate_overall_efficiency(state)
    improvement = (final_efficiency - env.original_efficiency) / env.original_efficiency * 100
    print(f"\nOptimization completed! {adjustments} effective adjustments performed.")
    print(f"Initial efficiency: {env.original_efficiency:.7f}")
    print(f"Final efficiency: {final_efficiency:.7f}")
    print(f"Improvement: {improvement:.2f}%")



    adjustment_df = pd.DataFrame(env.adjustment_history)
    adjustment_df.to_excel("adjustment_log.xlsx", index=False)

    optimized_df = data_loader.df.copy()
    optimized_df['optimized_distance'] = state[:, 1]
    optimized_df['optimized_rank'] = state[:, 2]
    optimized_df.to_excel("optimized_layout.xlsx", index=False)


def save_optimized_results(data_loader, optimized_state, adjustment_log):
    optimized_df = data_loader.df.copy()
    optimized_df['optimized_Manhattan_distance'] = optimized_state[:, 1]
    optimized_df['optimized_SpringSummer_rank'] = optimized_state[:, 2]
    optimized_df['optimized_need_adjust'] = optimized_state[:, 3]

    top_skus = pd.DataFrame(adjustment_log)['sku'].value_counts().head(100)
    top_skus.to_excel("top_adjusted_SKUs.xlsx")

    optimized_df.to_excel("optimized_layout_final.xlsx", index=False)
    print("Optimized layout saved to: optimized_layout_final.xlsx")

    log_df = pd.DataFrame(adjustment_log)
    log_df.to_excel("adjustment_log_final.xlsx", index=False)
    print("Adjustment log saved to: adjustment_log_final.xlsx")


# ==================== Main ====================
def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading order data...")
    sales_data_loader = SalesDataLoader("order data.xlsx")

    print("Loading warehouse data...")
    warehouse_data_loader = WarehouseDataLoader("initial storage location assignment.xlsx", sales_data_loader)

    # Validate SKU matching rate
    order_skus = set([sku for sublist in sales_data_loader.order_df[COL_SKU_LIST] for sku in sublist])
    warehouse_skus = set(warehouse_data_loader.df[COL_SKU_W])
    matched = order_skus.intersection(warehouse_skus)
    print(
        f"Total unique SKUs in orders: {len(order_skus)}, Total unique SKUs in warehouse: {len(warehouse_skus)}, Matched SKUs: {len(matched)}")

    print("Creating environment...")
    env = WarehouseEnv(warehouse_data_loader)

    # ================= print: Distance-feature updates =================
    # print("\n--- Distance-feature updates ---")
    #state, lstm_data = env.reset()

    # Force a step to trigger the location swap and state update
    # action = 0
    # next_state, next_lstm_data, reward, done, info = env.step(action)

    # from_sku = info['from_sku']
    # to_sku = info['to_sku']
    # from_idx = env.data_loader.sku_to_id[from_sku]
    # to_idx = env.data_loader.sku_to_id[to_sku]

    # Retrieve the normalized distance from the state matrix
    # max_dist = max(env.distance_mapping.values())
    # state_from_dist = next_state[from_idx, 1] * max_dist
    # state_to_dist = next_state[to_idx, 1] * max_dist

    # Retrieve the actual physical distance from the dictionary
    # actual_from_dist = env.distance_mapping[env.sku_to_position[from_sku]]
    # actual_to_dist = env.distance_mapping[env.sku_to_position[to_sku]]

    # print(f"Swap occurred between SKU {from_sku} and SKU {to_sku}")
    # print(
        # f"SKU {from_sku} (idx {from_idx}): Actual physical dist = {actual_from_dist:.2f}, State feature dist = {state_from_dist:.2f}")
    # print(
        # f"SKU {to_sku} (idx {to_idx}): Actual physical dist = {actual_to_dist:.2f}, State feature dist = {state_to_dist:.2f}")

    # if abs(actual_from_dist - state_from_dist) < 1e-3 and abs(actual_to_dist - state_to_dist) < 1e-3:
        # print("Validation PASSED: State distance features perfectly match physical positions.")
    # else:
        # print("Validation FAILED: State features do not match physical positions.")
    # print("----------------------------------------------------------------------------------\n")
    # ==================================================================================

    initial_efficiency = env._calculate_overall_efficiency()
    print(f"Initial system efficiency: {initial_efficiency:.7f}")

    print("Building adjacency matrix...")
    adj_matrix = warehouse_data_loader.get_adjacency_matrix()
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0.01:
                edge_index.append([i, j])
                edge_index.append([j, i])
    if not edge_index:
        print("Warning: No valid edges found, adding random edges")
        for i in range(0, min(100, adj_matrix.shape[0]), 2):
            j = i + 1
            if j < adj_matrix.shape[0]:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    print("Initializing fusion model...")
    fusion_model = FusionModel(gnn_input_dim=5, lstm_input_dim=3, hidden_dim=64, output_dim=32).to(device)

    print("Creating policy network...")
    policy_net = PolicyNetwork(fusion_model=fusion_model,
                               state_dim=env.observation_space.shape,
                               action_dim=env.action_space.n,
                               device=device).to(device)
    policy_net.set_graph_data(edge_index.to(device))

    print("Initializing PPO...")
    ppo = PPO(policy=policy_net, env=env, learning_rate=1e-4, n_steps=128, batch_size=16, gamma=0.99, clip_epsilon=0.15)

    # print("\n---  PPO Critic Gradient Update ---")
    # import copy
    # 1. Record initial weights of the Critic network (the value function)
    # initial_critic_weights = copy.deepcopy(policy_net.fc_value[0].weight.data.clone())

    # 2. Run 128 steps to trigger exactly one PPO update (n_steps=128 in your PPO init)
    # print("Running 128 timesteps to trigger one PPO update...")
    # ppo.learn(total_timesteps=128)

    # 3. Compare weights after the update
    # updated_critic_weights = policy_net.fc_value[0].weight.data.clone()

    # Calculate detailed metrics
    # weight_diff_L1 = torch.sum(torch.abs(updated_critic_weights - initial_critic_weights)).item()
    # norm_initial = torch.norm(initial_critic_weights).item()
    # norm_updated = torch.norm(updated_critic_weights).item()
    # relative_change = (weight_diff_L1 / norm_initial) * 100 if norm_initial > 0 else 0

    # print("\n" + "=" * 60)
    # print("PPO Critic Gradient Verification Results:")
    # print(f"  - Initial Critic Weight Norm (L2) : {norm_initial:.6f}")
    # print(f"  - Updated Critic Weight Norm (L2) : {norm_updated:.6f}")
    # print(f"  - Total Absolute Difference (L1)  : {weight_diff_L1:.6f}")
    # print(f"  - Relative Weight Change          : {relative_change:.4f}%")

    # if weight_diff_L1 > 1e-6:
        # print("  => Validation PASSED: Critic weights successfully updated.")
        # print("  => Proof: The gradient is flowing and the network is learning.")
    # else:
        # print("  => Validation FAILED: Critic weights did NOT update.")
        # print("  => Error: Gradient flow is still blocked.")
    # print("=" * 60 + "\n")



    print("Starting training...")
    rewards = ppo.learn(total_timesteps=10000)

    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'fusion_state_dict': fusion_model.state_dict(),
        'edge_index': edge_index
    }, "warehouse_optimization_model.pth")

    policy_net.load_state_dict(torch.load("best_model.pth", map_location=device))
    policy_net.eval()
    apply_dynamic_adjustment(policy_net, env, warehouse_data_loader)


if __name__ == "__main__":
    main()
