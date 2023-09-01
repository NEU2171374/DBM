# ######### 单个 runSingle() ###########
# 深度参数
depth_walk_length = 10
depth_num_walks = 80 # 80
depth_window_size = 5 # 5

# 广度参数
breadth_walk_length = 75 # 150 MovieTweet(100000); 75 MovieLens(10000)
breadth_num_walks = 80 # 80
breadth_window_size = 10 # 10

embed_size = 64 # 128

set = 10000 # 100000 MovieTweet;       10000 MovieLens

# 推荐参数
top_N = 20 # 20
top_people = 10 # 10

# ############# 批量 runAll() ##################
#
# 深度参数
depth_walk_length_ = [10, 20, 30, 40, 50]
#depth_num_walks_ = 80
#depth_window_size_ = 5

# 广度参数
breadth_walk_length_ = [75, 100, 125, 150, 175, 200]
#breadth_num_walks_ = 80
#breadth_window_size_ = 10

embed_size_ = [64, 32] # 128, 256, 384, 512
set_ = [100000, 10000]

top_N_ = [10, 20, 30, 40, 50]
top_people_ = [5, 10, 15, 20, 25, 30]

# #######使用runAll()时, 其他变量选用_default###########
# 默认值
depth_walk_length_default = 10

depth_num_walks_default = 80
depth_window_size_default = 5

# 广度参数
#breadth_walk_length_default = 200

breadth_num_walks_default = 80
breadth_window_size_default = 10

# 推荐参数
top_N_default = 20 # 20
top_people_default = 10 # 10
embed_size_default = 128 # 128