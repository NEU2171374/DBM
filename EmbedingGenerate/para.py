# ######### 单个 runSingle() ###########
# 深度参数
depth_walk_length = 10
depth_num_walks = 80
depth_window_size = 5
depth_embed_size = 128
depth_type = 0  # 1 film_node; 0 people_node
depth_set = 100000  # 100000 MovieTweet;       10000 MovieLens

# 广度参数
breadth_walk_length = 75 # 150 MovieTweet;        75 MovieLens
breadth_num_walks = 80
breadth_window_size = 10
breadth_embed_size = 64
breadth_type = 1  # 1 film_node; 0 people_node
breadth_set = 10000  # 100000 MovieTweet;       10000 MovieLens


# ############# 批量 runAll() ##################

# 深度
depth_walk_length_ = [10, 20, 30, 40, 50] # 10
depth_num_walks_ = [80] # 80
depth_window_size_ = [5] # 5

# 宽度
breadth_walk_length_ = [200, 175, 150, 125, 100, 75, 50] # 200
breadth_num_walks_ = [80] # 150 400
breadth_window_size_ = [10] # 10

embed_size_ = [64, 32] # 128, 256, 384, 512
type_ = [0, 1] # 1 film_node; 0 people_node
set_ = [100000, 10000] # 100000 MovieTweet;       10000 MovieLens

##############################################
# 默认
depth_walk_length_default = 10
depth_num_walks_default = 80
depth_window_size_default = 5

breadth_walk_length_default = 75 # 75 MovieLens, 150 MovieTweet
breadth_num_walks_default = 80 # 150 400
breadth_window_size_default = 10

embed_size_default = 128
