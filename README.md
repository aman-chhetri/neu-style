# Neural Style Transfer (NST)

> Neural Style Transfer (NST) is a technique which combines two images (content image for the object of the image and the style image from which only the style is extracted) into a third target image. 

![Cover_Image](/assets/cover_img.jpg)


Basically, in Neural Style Transfer we have two images- style and content. We need to copy the style from the style image and apply it to the content image. By, style we basically mean, the patterns, the brushstrokes, etc.

`Orginal Paper`: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

---

The Basic concept is simple: 

* Take two losses and minimize them as much as possible. 
* The first loss describes the distance in content between the content image and the target image.
* While the second loss describes the distance in style between the style image and the style of the input image.

---
> [!NOTE]  
> Want to try out the `app version` of this paper? Check out here: [NeuraCanvas](https://github.com/aman-chhetri/NeuraCanvas)


---
## Theory of the Concept (paper):

**`Losses`:** The complete loss is a combination of the content and style losses: simply put, we try to minimize the jointly losses on the target image:

![Losses](/assets/Losses.png)

So, we need to define the content loss and the style loss while alpha and beta are weighted factors for every loss.


**`Content Loss`:** Out of the two losses that describe NST, the content loss is the simpler one. We feed the network both the content image and the target image, which return the feature representation of each image from the intermediate layers. To get the content loss, we calculate the squared error loss between the two feature vectors:

![Content_Loss](/assets/Content_Loss_Eqn.png)

In formal words, the above function calculates the loss, which takes the content image p, the target image x, and processed layer l. F and P are the feature representation of the content image and the target image on the layer l. The gradient is calculated with the simple error back propagation based on which we can change the target image until the feature vector is the same as the content image feature vector. With that we have our content loss.

```
def get_content_loss(target, content):
  return torch.mean((target-content)**2)
```
The implementation is quite simple as you can see.


**`Style Loss`:** Before being able to calculate the style loss, we first need a way to grasp a style of an image and the measure of correlation between features after each layer. To get the needed correlations, we can use Gram matrices which the authors have uesd in the paper. <br> As mentioned in the paper, the style loss is equal to computing the Maximum Mean Discrepancy between two images, so as long the measure is an implementation of the MMD algorithm the loss is computed adequately.

![Gram_Matrix](/assets/Gram_Matrix.png)

The gram matrix is calculated for every layer. To reconstruct the style, a gradient descent is performed on the target image by calculating the mean-squared distance between two gram matrices. The total loss is the sum of every mean-squared distance for every layer E times the weighted factor (the influence factor) of every layer:

![Style_Loss_Eqn](/assets/Style_Loss_Eqn.png)

```
def gram_matrix(input, c, h, w):
  #c-channels; h-height; w-width 
  input = input.view(c, h*w) 
  #matrix multiplication on its own transposed form
  G = torch.mm(input, input.t())
  return G
  
def get_style_loss(target, style):
  _, c, h, w = target.size()
  G = gram_matrix(target, c, h, w) #gram matrix for the target image
  S = gram_matrix(style, c, h, w) #gram matrix for the style image
  return torch.mean((G-S)**2)/(c*h*w)
```
Unlike the content loss, implementing the style loss is a bit harder but nothing too complicated. We use the pytorch function for matrices multiplication therefor the process is straightforward. Our matrix dimension is equal to our image size and channel.


## The Model Architecture:
After we have defined the distances we want to minimize, we need to manipulate a pretrained model in a way to get features after every layer. Gatys, with his coauthors, has used the VGG19 model for image classification. The model was trained on the ImageNet dataset and is 19 layers deep.

![Model](/assets/Model.png)
In the paper the authors use only one layer for the content image and five layers for the style image. As mentioned before, we need convolutional layers after MaxPool layers which are conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1 layers (indices: 0, 5, 10, 19, 28).

```
#class for loading the vgg19 model with wanted layers

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.select_features = ['0', '5', '10', '19', '28'] #conv layers
    self.vgg = models.vgg19(pretrained=True).features
  
  def forward(self, output):
    features = []
    for name, layer in self.vgg._modules.items():
      output = layer(output)
      if name in self.select_features:
        features.append(output)
    return features

#load the model
vgg = VGG().to(device).eval()
```
The last part that we need for the implementation is the model. We instance the model with just the needed layers mentioned above; we simply go through every layer and extract the layers we need on selected indices.


## TLDR; of theory (paper):

After all the theory explained, this should be quite easy to understand.

* Extract the feature vector for every image
* Calculate the two losses by summing the losses of every layer
* Calculate the total loss
* Gradient Optimization and update the parameters

## Results

Have a look at the sample images that I have generated with this repository. The content and style images are located at the left corner and center respectively, and the right corner contains the output (stylized) image.

![Model](/assets/sample_img.png)


## Acknowledgements

These are some of the resources I referred to while working on this project. You might want to check them out.

* The original paper on neural style transfer by [Gatys et al](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) .
* PyTorch's [tutorial on NST](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
* The original paper on [VGG19](https://arxiv.org/abs/1409.1556).

I found these repos useful: (while implementing the model)
* [fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) (PyTorch, feed-forward method)
* [neural-style-tf](https://github.com/cysmith/neural-style-tf/) (TensorFlow, optimization method)
* [neural-style](https://github.com/anishathalye/neural-style/) (TensorFlow, optimization method)

I found some of the content/style images from here:
* [Rawpixel](https://www.rawpixel.com/board/537381/vincent-van-gogh-free-original-public-domain-paintings?sort=curated&mode=shop&page=1)
* [Wikimedia](https://commons.wikimedia.org/wiki/Category:Images)
* [Unsplash](https://unsplash.com/)
* [Pexels](https://www.pexels.com/)