#导入和配置模块
import tensorflow as tf

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False#设置不显示网格线

import numpy as np
import PIL.Image
import time

#将张量转换为图像的函数
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:#如果张量维度大于三，会进行降维操作
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

#下载图片（可以从网上直接下载图片）
#content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
#style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
#print(content_path)
#或者可以直接将图片先保存到电脑中
#这里可以改成想要改变风格的图片(这里需要提供绝对路径）
content_path="E:\pythonProject\style\maomao.jpeg"
style_path="E:\pythonProject\style\莫.jpeg"

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)# 获取图像的形状（高度、宽度、通道数）
  long_dim = max(shape)
  scale = max_dim / long_dim# 计算缩放比
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape) # 调整图像大小
  img = img[tf.newaxis, :]# 增加一个维度，以符合期望的输入形状
  return img


#处理图片
content_image = load_img(content_path)
style_image = load_img(style_path)

#选择中间层的输出表示图像的内容和风格
content_layers = ['block5_conv2']#内容层
#样式层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

#建立模型
def vgg_layers(layer_names):
    #这里直接调用预训练模型VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False#禁用模型可训练性

    # 获取指定层的输出，并构建新的模型
    # 也就是内容层和样式层
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

'''
#可以查看一下风格层下的各个输出
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()
'''

#风格计算
#此处是计算gram矩阵
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)#对两个张量进行乘法运算
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)#最后得到归一化的Gram矩阵

#提取风格和内容
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers) #总vgg模型
    self.style_layers = style_layers #样式层
    self.content_layers = content_layers #内容层
    self.num_style_layers = len(style_layers) #样式层层数
    self.vgg.trainable = False #不允许vgg模型再进行训练

  def call(self, inputs):
    inputs = inputs*255.0
    #将预处理后的输入张量传递给VGG模型，得到输出结果
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    #将输出结果划分为样式层输出和内容层输出
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    #计算样式层的gram矩阵
    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]
    #将内容层名称与内容层输出值一一对应，存储在字典中
    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}
    #将样式层名称与样式层输出值一一对应，存储在字典中
    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    #返回两个字典
    return {'content': content_dict, 'style': style_dict}
#设置好风格内容模型
extractor = StyleContentModel(style_layers, content_layers)

#梯度下降过程
#设置目标值
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

def clip_0_1(image):#为了使像素值保持在0-1之间
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

#optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

#损失函数（两个损失加权）
style_weight=1e-2
content_weight=1e4
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


#初始化优化图像
image = tf.Variable(content_image)

#更新图像函数
total_variation_weight=30#总变化权重，用于控制图像的总变化对损失的贡献
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)#这里加上图像总变化作为总的损失

    grad = tape.gradient(loss, image)#计算梯度
    opt.apply_gradients([(grad, image)])#使用优化器将梯度应用于输入图像，更新模型参数
    image.assign(clip_0_1(image))#让图像的像素值保持在0-1之间


#进行图像风格转移
start = time.time()
epochs = 2
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  tensor_to_image(image).show()
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))

