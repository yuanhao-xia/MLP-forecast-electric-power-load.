import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 生成符合居民用电特性的模拟数据集
# ======================
def generate_electricity_load(start_date='2023-01-01', days=365):
    """
    生成具有真实感的居民用电负荷数据（单位：kW）
    特性：日周期性、周周期性、周末效应、趋势、噪声
    """
    np.random.seed(42)
    hours = days * 24
    time_index = pd.date_range(start=start_date, periods=hours, freq='H')
    
    # 基础负荷（均值2.0）
    base_load = 2.0
    
    # 日周期性（振幅0.8，居民白天高、夜间低）
    hour_sin = np.sin(2 * np.pi * (np.arange(hours) % 24) / 24)
    daily_pattern = 0.8 * hour_sin
    
    # 周周期性（工作日高、周末低，振幅0.5）
    day_of_week = time_index.dayofweek  # 0=周一, 6=周日
    weekly_pattern = 0.5 * np.sin(2 * np.pi * day_of_week / 7)
    
    # 周末效应（周六日降低20%）
    weekend_mask = (day_of_week >= 5).astype(int)  # 周六日为1
    weekend_effect = -0.4 * weekend_mask
    
    # 缓慢增长趋势（模拟用户增长）
    trend = 0.0005 * np.arange(hours)
    
    # 随机噪声（高斯+偶尔尖峰）
    noise = 0.15 * np.random.randn(hours)
    spike_events = (np.random.rand(hours) < 0.01).astype(int) * np.random.uniform(0.3, 0.8, hours)
    
    # 合成负荷（确保>0）
    load = base_load + daily_pattern + weekly_pattern + weekend_effect + trend + noise + spike_events
    load = np.maximum(load, 0.3)  # 避免负值
    
    # 创建DataFrame
    df = pd.DataFrame({
        'datetime': time_index,
        'load': load
    })
    df.set_index('datetime', inplace=True)
    return df

excel_file_path = r'C:\Users\lenovo\Desktop\第30期大创立项多智能体协同优化\数据汇总.xlsx'  # 请替换为你的实际文件路径


try:
    # 读取Excel数据
    time_data = pd.read_excel(excel_file_path, sheet_name='数据汇总', header=None)
    electricity_load = time_data.iloc[1:8761, 1].values.astype(float)  # 读取第一列数据
    
    # 创建DataFrame替换原有df
    start_date = '2023-01-01'
    time_index = pd.date_range(start=start_date, periods=len(electricity_load), freq='H')
    df = pd.DataFrame({
        'datetime': time_index,
        'load': electricity_load
    })
    df.set_index('datetime', inplace=True)
    
    print("✅ 真实数据加载成功！")
    print(f"数据形状: {df.shape} | 时间范围: {df.index.min()} 至 {df.index.max()}")
    print(f"负荷统计: 最小={df['load'].min():.2f}kW, 最大={df['load'].max():.2f}kW, 均值={df['load'].mean():.2f}kW")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    print("使用原始模拟数据...")
    df = generate_electricity_load(days=365)  # 备用方案
    print(f"数据形状: {df.shape} | 时间范围: {df.index.min()} 至 {df.index.max()}")
    print(f"负荷统计: 最小={df['load'].min():.2f}kW, 最大={df['load'].max():.2f}kW, 均值={df['load'].mean():.2f}kW")

# ======================
# 3. 可视化原始数据（验证合理性）
# ======================
def plot_sample_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 全年趋势
    axes[0].plot(df.index, df['load'], linewidth=0.8, color='steelblue')
    axes[0].set_title('全年用电负荷趋势', fontsize=14)
    axes[0].set_ylabel('负荷 (kW)')
    
    # 一周示例（第10周）
    week_sample = df['2023-03-06':'2023-03-12']  # 选一周
    axes[1].plot(week_sample.index, week_sample['load'], marker='o', markersize=3)
    axes[1].set_title('单周负荷波动（展示日周期性）', fontsize=14)
    axes[1].set_ylabel('负荷 (kW)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 一日示例（工作日）
    day_sample = df['2023-03-08 00:00':'2023-03-08 23:00']
    axes[2].plot(day_sample.index, day_sample['load'], 'ro-', linewidth=2)
    axes[2].set_title('单日负荷曲线（典型工作日）', fontsize=14)
    axes[2].set_ylabel('负荷 (kW)')
    axes[2].set_xlabel('时间')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('load_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_sample_data(df)

# ======================
# 4. 数据预处理：归一化 + 构造监督学习样本
# ======================
def create_dataset(data, look_back=168, look_forward=24):
    """
    将时间序列转换为监督学习格式
    X: [样本数, look_back, 特征数]  -> 过去168小时
    y: [样本数, look_forward]       -> 未来24小时
    """
    X, y = [], []
    total_len = len(data)
    for i in range(total_len - look_back - look_forward + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + look_forward)])
    return np.array(X), np.array(y)

# 归一化（仅对负荷值）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_load = scaler.fit_transform(df[['load']]).flatten()  # 转为1D数组

# 构造样本：输入168小时，预测24小时
LOOK_BACK = 168  # 7天历史
LOOK_FORWARD = 24  # 预测24小时
X, y = create_dataset(scaled_load, LOOK_BACK, LOOK_FORWARD)

print(f"\n✅ 样本构造完成！")
print(f"输入X形状: {X.shape} -> (样本数, 时间步168, 特征1)")
print(f"输出y形状: {y.shape} -> (样本数, 预测步长24)")
print(f"总样本数: {len(X)} | 可覆盖 {len(X)/24:.1f} 天的训练窗口")

# ======================
# 5. 严格按时间顺序划分数据集（禁止shuffle!）
# ======================
# 计算划分点（70%训练, 15%验证, 15%测试）
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"\n✅ 数据集划分完成（严格时序）:")
print(f"训练集: {X_train.shape} | 验证集: {X_val.shape} | 测试集: {X_test.shape}")

# ======================
# 6. 构建MLP模型（替换原CNN-LSTM）
# ======================
def build_mlp_model(input_dim, output_steps):
    """
    构建优化的MLP模型（针对时序展平特征）
    输入: (样本数, 168) -> 168维特征向量（7天历史负荷）
    输出: (样本数, 24) -> 未来24小时负荷预测
    """
    model = Sequential([
        # 输入层 + 第一隐藏层（大幅增加容量）
        Dense(1024, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),  # 减少dropout防止信息丢失
        
        # 批量归一化层 - 重要改进！
        tf.keras.layers.BatchNormalization(),
        
        # 第二隐藏层
        Dense(512, activation='relu'),
        Dropout(0.15),
        tf.keras.layers.BatchNormalization(),
        
        # 第三隐藏层
        Dense(256, activation='relu'),
        Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        
        # 第四隐藏层（新增，增强表达能力）
        Dense(128, activation='relu'),
        Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        
        # 输出层（线性激活，回归任务）
        Dense(output_steps, activation='linear')
    ])
    
    # 编译：保持与原模型一致的优化器和指标
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.005),
        metrics=['mae']
    )
    return model

# 创建模型（输入维度=168, 输出=24）
model = build_mlp_model(LOOK_BACK, LOOK_FORWARD)
print("\n✅ MLP模型结构:")
model.summary()

# ======================
# 7. 训练模型（含早停和学习率调整）
# ======================
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.7, 
        patience=5, 
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,  # MLP通常需更多轮次收敛（原100→150）
    batch_size=64,  # 稍大batch提升梯度稳定性（原32→64）
    callbacks=callbacks,
    verbose=1
)

# ======================
# 8. 模型评估与结果可视化
# ======================
# 预测（测试集）
y_pred_scaled = model.predict(X_test, verbose=0)

# 反归一化到原始尺度
y_test_inv = scaler.inverse_transform(y_test)  # [samples, 24]
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

# 计算整体指标（将所有预测点展平计算）
flat_true = y_test_inv.flatten()
flat_pred = y_pred_inv.flatten()
mae = mean_absolute_error(flat_true, flat_pred)
rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
r2 = r2_score(flat_true, flat_pred)

print(f"\n✅ 测试集评估结果（反归一化后）:")
print(f"MAE: {mae:.3f} kW | RMSE: {rmse:.3f} kW | R²: {r2:.4f}")

# 可视化：损失曲线
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='训练损失', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('模型训练损失曲线', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印预测值的24个点
print(f"\n📊 测试集第1个样本的24小时预测结果:")
print("=" * 50)
for i, (true_val, pred_val) in enumerate(zip(y_test_inv[0], y_pred_inv[0])):
    hour = i + 1
    error = abs(true_val - pred_val)
    print(f"第{hour:2d}小时 | 真实值: {true_val:6.2f}kW | 预测值: {pred_val:6.2f}kW | 误差: {error:5.2f}kW")

# 计算并显示统计信息
mae_sample = mean_absolute_error(y_test_inv[0], y_pred_inv[0])
rmse_sample = np.sqrt(mean_squared_error(y_test_inv[0], y_pred_inv[0]))
print("=" * 50)
print(f"📊 该样本统计指标:")
print(f"平均绝对误差(MAE): {mae_sample:.3f} kW")
print(f"均方根误差(RMSE): {rmse_sample:.3f} kW")
print(f"最大误差: {np.max(np.abs(y_test_inv[0] - y_pred_inv[0])):.3f} kW")
print(f"最小误差: {np.min(np.abs(y_test_inv[0] - y_pred_inv[0])):.3f} kW")

print(f"\n📋 其他样本预测示例:")
print("-" * 30)
for sample_idx in [1, 2, 3]:  # 显示前3个测试样本
    if sample_idx < len(y_test_inv):
        sample_mae = mean_absolute_error(y_test_inv[sample_idx], y_pred_inv[sample_idx])
        print(f"测试样本{sample_idx}: MAE={sample_mae:.3f}kW")

# 可视化：预测效果对比（选取测试集第一个样本）
plt.figure(figsize=(14, 6))
hours = np.arange(1, LOOK_FORWARD + 1)
plt.plot(hours, y_test_inv[0], 'bo-', label='真实值', linewidth=2, markersize=6)
plt.plot(hours, y_pred_inv[0], 'r^--', label='预测值', linewidth=2, markersize=6)
plt.title(f'未来24小时负荷预测示例（测试集第1个样本）\nMAE={mean_absolute_error(y_test_inv[0], y_pred_inv[0]):.3f}kW', fontsize=14)
plt.xlabel('未来小时数')
plt.ylabel('负荷 (kW)')
plt.xticks(hours[::2])  # 每2小时标一个刻度
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 模型训练完毕！结果已保存为：training_loss.png 和 prediction_comparison.png")

# 可视化：预测效果对比（将前5个测试样本连成120个点）
plt.figure(figsize=(15, 6))

# 将前5个样本的真实值和预测值连接成120个点
y_test_concat = np.concatenate([y_test_inv[i] for i in range(min(5, len(y_test_inv)))])
y_pred_concat = np.concatenate([y_pred_inv[i] for i in range(min(5, len(y_pred_inv)))])

# 创建120个小时的时间轴
hours_120 = np.arange(1, len(y_test_concat) + 1)

# 绘制连接的120个点
plt.plot(hours_120, y_test_concat, 'bo-', label='真实值(前5样本)', linewidth=1.5, markersize=4)
plt.plot(hours_120, y_pred_concat, 'r^--', label='预测值(前5样本)', linewidth=1.5, markersize=4)

# 添加每24小时的分隔线来标识不同的样本
for i in range(1, 5):
    plt.axvline(x=i*24, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    plt.text(i*24-12, plt.ylim()[1]*0.95, f'样本{i}', ha='center', va='top', 
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 计算整体MAE
overall_mae = mean_absolute_error(y_test_concat, y_pred_concat)

plt.title(f'连续120小时负荷预测对比（前5个测试样本）\n总体MAE={overall_mae:.3f}kW', fontsize=14)
plt.xlabel('连续小时数 (120小时 = 5个样本 × 24小时)')
plt.ylabel('负荷 (kW)')
plt.xticks(range(0, 121, 12))  # 每12小时标一个刻度
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_comparison_120hours.png', dpi=300, bbox_inches='tight')
plt.show()