import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ( Input, Dense, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, Add )
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# 构建改进后的特征融合模型
def create_advanced_fusion_model(seq_shape, mRNA_co_loc_shape, mRNA_net_shape, dis_shape, num_classes):
    # 输入层
    seq_input = Input(shape=(seq_shape,), name='seq_feature')
    mRNA_co_loc_input = Input(shape=(mRNA_co_loc_shape,), name='mRNA_co_loc_feature')
    mRNA_net_input = Input(shape=(mRNA_net_shape,), name='mRNA_net_feature')
    dis_input = Input(shape=(dis_shape,), name='dis_feature')

    # 特征提取层
    seq_dense = Dense(64, activation='relu')(seq_input)
    mRNA_co_loc_dense = Dense(64, activation='relu')(mRNA_co_loc_input)
    mRNA_net_dense = Dense(64, activation='relu')(mRNA_net_input)
    dis_dense = Dense(64, activation='relu')(dis_input)

    # 特征选择模块
    dense_concat = Concatenate()([seq_dense, mRNA_co_loc_dense, mRNA_net_dense, dis_dense])
    attention_weights = Dense(dense_concat.shape[-1], activation='softmax')(dense_concat)  # 输出维度匹配 dense_concat

    selected_features = tf.multiply(dense_concat, attention_weights)  # 按权重筛选特征

    # 调整形状（增加时间维度）
    selected_features_expanded = tf.expand_dims(selected_features, axis=1)  # [batch_size, 1, feature_dim]
    key_value_features = tf.expand_dims(dense_concat, axis=1)  # [batch_size, 1, feature_dim]

    # 特征融合层：交互注意力
    attention_output = MultiHeadAttention(num_heads=10, key_dim=64)(
        query=selected_features_expanded, value=key_value_features, key=key_value_features
    )

    # 去掉时间维度
    attention_output_squeezed = tf.squeeze(attention_output, axis=1)  # [batch_size, feature_dim]

    # 残差连接 + 层归一化
    residual_output = Add()([dense_concat, attention_output_squeezed])
    normalized_output = LayerNormalization()(residual_output)

    # Dropout 层（动态）
    dropout_output = Dropout(rate=tf.random.uniform([], minval=0.1, maxval=0.5))(normalized_output)

    # 输出层
    output = Dense(num_classes, activation='sigmoid', name='output')(dropout_output)

    # 构建模型
    model = Model(inputs=[seq_input, mRNA_co_loc_input, mRNA_net_input, dis_input], outputs=output)
    return model

# 多标签评估指标
def multi_evaluate(y_pred, y_test):
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_bin: precision_score(y_true, y_bin, zero_division=0),
        "recall": lambda y_true, y_bin: recall_score(y_true, y_bin, zero_division=0),
        "f1": lambda y_true, y_bin: f1_score(y_true, y_bin, zero_division=0),
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score
    }
    results = {key: [] for key in metrics}
    for class_idx in range(y_test.shape[1]):
        y_true = y_test[:, class_idx]
        y_prob = y_pred[:, class_idx]
        y_bin = (y_prob > 0.5).astype(int)
        for metric_name, metric_func in metrics.items():
            try:
                result = metric_func(y_true, y_bin if metric_name != "roc_auc" else y_prob)
            except ValueError:
                result = np.nan
            results[metric_name].append(result)
    return results

if __name__ == '__main__':
    # 读取数据
    df_seq_feature = pd.read_csv('../feature/miRNA_seq_feature_64.csv', index_col='Index')
    df_mRNA_co_loc_feature = pd.read_csv('../feature/miRNA_mRNA_co-localization_feature.csv', header=None)
    df_mRNA_net_feature = pd.read_csv('../feature/gate_feature_mRNA_0.8_128_0.01.csv', header=None)
    df_dis_feature = pd.read_csv('../feature/gate_feature_disease_0.8_128_0.01.csv', header=None)
    df_loc = pd.read_csv('../dataset/miRNA_localization.csv', header=None)
    df_loc_index = pd.read_csv('../dataset/miRNA_have_loc_information_index.txt', header=None)

    loc_index = df_loc_index[0].tolist()
    select_row = np.array([value == 1 for value in loc_index])

    # 提取特征
    seq_feature = df_seq_feature.values
    mRNA_co_loc_feature = df_mRNA_co_loc_feature.values
    mRNA_net_feature = df_mRNA_net_feature.values
    dis_feature = df_dis_feature.values
    miRNA_loc = df_loc.values

    # 合并特征
    merge_feature = np.concatenate((seq_feature, mRNA_co_loc_feature, mRNA_net_feature, dis_feature), axis=1)

    # 数据归一化
    n_splits = 10  # K折交叉验证
    scaler = StandardScaler()
    merge_feature_scaled = scaler.fit_transform(merge_feature)

    # 多标签数据
    miRNA_loc_multilabel = miRNA_loc[select_row]
    merge_feature_scaled_multilabel = merge_feature_scaled[select_row]

    num_classes = 7
    random_seed = 42
    np.random.seed(random_seed)
    auc_ls = [0] * 7
    aupr_ls = [0] * 7
    class_name = ['Cytoplasm', 'Exosome', 'Nucleolus', 'Nucleus', 'Extracellular vesicle', 'Microvesicle',
                  'Mitochondrion']

    # 模型训练和评估
    fold_size = len(merge_feature_scaled_multilabel) // n_splits

    with open("evaluation_results.txt", "w") as f:
        for i in range(n_splits):
            # 数据切分
            X, y = shuffle(merge_feature_scaled, miRNA_loc, random_state=random_seed)

            # 定义测试集和训练集索引
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            test_indices = range(test_start, test_end)
            train_indices = [j for j in range(len(X)) if j not in test_indices]

            # 分割数据
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # 特征分拆
            seq_shape = seq_feature.shape[1]
            mRNA_co_loc_shape = mRNA_co_loc_feature.shape[1]
            mRNA_net_shape = mRNA_net_feature.shape[1]
            dis_shape = dis_feature.shape[1]

            X_train_seq, X_train_mRNA_co_loc, X_train_mRNA_net, X_train_dis = \
                X_train[:, :seq_shape], \
                X_train[:, seq_shape:seq_shape + mRNA_co_loc_shape], \
                X_train[:, seq_shape + mRNA_co_loc_shape:seq_shape + mRNA_co_loc_shape + mRNA_net_shape], \
                X_train[:, seq_shape + mRNA_co_loc_shape + mRNA_net_shape:]

            X_test_seq, X_test_mRNA_co_loc, X_test_mRNA_net, X_test_dis = \
                X_test[:, :seq_shape], \
                X_test[:, seq_shape:seq_shape + mRNA_co_loc_shape], \
                X_test[:, seq_shape + mRNA_co_loc_shape:seq_shape + mRNA_co_loc_shape + mRNA_net_shape], \
                X_test[:, seq_shape + mRNA_co_loc_shape + mRNA_net_shape:]

            # 创建模型
            model = create_advanced_fusion_model(
                seq_shape=seq_shape,
                mRNA_co_loc_shape=mRNA_co_loc_shape,
                mRNA_net_shape=mRNA_net_shape,
                dis_shape=dis_shape,
                num_classes=num_classes
            )

            # 编译模型
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()

            # 训练模型
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(
                [X_train_seq, X_train_mRNA_co_loc, X_train_mRNA_net, X_train_dis], y_train,
                validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stopping]
            )

            # 预测
            y_pred = model.predict([X_test_seq, X_test_mRNA_co_loc, X_test_mRNA_net, X_test_dis])

            # 评估指标
            for class_idx in range(num_classes):
                true_label = y_test[:, class_idx]
                pre_label = y_pred[:, class_idx]

                accuracy = accuracy_score(true_label, (pre_label > 0.5).astype(int))
                precision = precision_score(true_label, (pre_label > 0.5).astype(int), zero_division=0)
                recall = recall_score(true_label, (pre_label > 0.5).astype(int), zero_division=0)
                f1 = f1_score(true_label, (pre_label > 0.5).astype(int), zero_division=0)
                auc_score = roc_auc_score(true_label, pre_label)
                aupr_score = average_precision_score(true_label, pre_label)

                # 输出到文件
                f.write(
                    f"Class {class_idx} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}\n"
                )

                auc_ls[class_idx] += auc_score
                aupr_ls[class_idx] += aupr_score
        avg_auc = np.array(auc_ls) / n_splits
        avg_aupr = np.array(aupr_ls) / n_splits
        # 输出最终评估结果到文件
        f.write("\n-----------------------Final Result-----------------------\n")
        for i in range(num_classes):
            f.write(f"Localization {class_name[i]}:\n")
            f.write(f"AUC: {avg_auc[i]:.4f}, AUPR: {avg_aupr[i]:.4f}\n")
        overall_avg_auc = np.mean(avg_auc)
        overall_avg_aupr = np.mean(avg_aupr)

        # 输出平均值到文件
        f.write("\nOverall Results:\n")
        f.write(f"Average AUC: {overall_avg_auc:.4f}\n")
        f.write(f"Average AUPR: {overall_avg_aupr:.4f}\n")