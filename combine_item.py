import polars as pl
'''
没有融合其他特征,仅仅使用item_emb.parquet中的item_emb_d128_v3特征,
原始的item_tags特征,
将自带的likes_level和views_level特征放在item_emb_d128之后
'''
# 读取item_info数据并选择所需列
item_info = pl.read_parquet("data/MicroLens_1M_x1/item_info.parquet").select([
    "item_id",
    "item_tags",
])

# 选择"item_emb_d128_v3"特征
item_emb_v3 = pl.read_parquet("data/item_emb.parquet").select([
    "item_id",
    "item_emb_d128_v3",
])

# 读取item_feature数据并选择所需列
item_feature = pl.read_parquet("data/item_feature.parquet").select([
    "item_id",
    "likes_level",
    "views_level"
])

# 执行左连接（保留所有item_info的行）
item_info = item_info.join(
    item_emb_v3,
    on="item_id",
    how="left"
)
item_info = item_info.join(
    item_feature,
    on="item_id",
    how="left"
)

# 填充缺失值为0并重命名列
item_info = item_info.with_columns(
    # pl.col("item_emb_d128_v3").fill_null(0),
    pl.col("item_emb_d128_v3").alias("item_emb_d128"),
    pl.col("likes_level").fill_null(0),
    pl.col("views_level").fill_null(0)
)
zeros_array = pl.lit([0.0] * 128).cast(pl.Array(pl.Float32, 128))

item_info = item_info.with_columns(
    # 处理item_emb_d128：item_id=0设为全0，其他情况填充null为全0
    pl.col("item_emb_d128").fill_null(zeros_array) )
    

# 确保保留原始顺序（Polars默认保留左表顺序）
# 显式选择最终列顺序
item_info = item_info.select([
    "item_id",
    "item_tags",
    "item_emb_d128",
    "likes_level",
    "views_level"
])

# 保存结果文件
item_info.write_parquet("data/MicroLens_1M_x1/item_info_v3.parquet")
print("合并完成！数据地址为：" + "data/MicroLens_1M_x1/item_info_v3.parquet")
print("合并后数据预览：")
print(item_info.head(3))