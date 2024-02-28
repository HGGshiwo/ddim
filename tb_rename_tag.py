
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter  
from tqdm import tqdm

input_path = '/home/bingwenzhang/ddim/exp/tensorboard/2024-01-19-11-22-13/events.out.tfevents.1705634533.star-SYS-4029GP-TRT.2498875.0'  # 输入需要指定event文件
output_path = '/home/bingwenzhang/ddim/exp/tensorboard/2024-01-19-11-22-13'  # 输出只需要指定文件夹即可
 
# 读取需要修改的event文件
ea = event_accumulator.EventAccumulator(input_path)
ea.Reload()
tags = ea.scalars.Keys()  # 获取所有scalar中的keys
 
# 写入新的文件
writer = SummaryWriter(output_path)  # 创建一个SummaryWriter对象
for tag in tqdm(tags):
    scalar_list = ea.scalars.Items(tag)
 
    if tag.startswith('layer'):  # 修改一下对应的tag即可
        tag = tag.split("/")[0].replace("layer", "")
        tag = int(tag)
        tag = tag * 100
        tag = f'layer{tag}/loss'
 
    for scalar in scalar_list:
        writer.add_scalar(tag, scalar.value, scalar.step, scalar.wall_time)  # 添加修改后的值到新的event文件中
writer.close()  # 关闭SummaryWriter对象