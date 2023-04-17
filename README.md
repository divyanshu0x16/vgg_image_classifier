## VGG Image Classifier

Table with model comparision between models:-

|Model                       |Training time|Training loss|Training accuracy|Testing accuracy|Number of model parameter|
|----------------------------|-------------|-------------|-----------------|----------------|-------------------------|
|VGG(1 block)                |195.099      |0.0206       |1                |0.725           |40961153                 |
|VGG(3 block)                |205.74       |0.1331       |0.9821           |0.725           |10333505                 |
|VGG(3 block) -Data augmented|261.7763922  |0.3806       |0.8125           |0.8             |10333505                 |
|Tranfer Learning(VGG16)     |743.9707832  |1.28E-14     |1                |3.05E-13        |17926209                 |
|MLP Model                   |172.1257512  |0.1665       |0.9625           |0.725           |17292049                 | 

### VGG 1

![vgg1_accuracy](https://user-images.githubusercontent.com/62815174/232404540-445df7d8-8f49-4aba-86d6-2500d9a85c41.png)

![vgg1_loss](https://user-images.githubusercontent.com/62815174/232404570-5e2fb515-6df8-44af-baf6-d62d76ed6f26.png)

### VGG 3

![vgg3_accuracy](https://user-images.githubusercontent.com/62815174/232404751-b5bc40bc-220a-4049-ba64-73fa8c284577.png)

![vgg3_loss](https://user-images.githubusercontent.com/62815174/232404780-ebb67e64-d603-4a9d-a4b0-570ba88241d6.png)


### VGG 3 - Data Augmentation

![vgg3da_accuracy](https://user-images.githubusercontent.com/62815174/232404906-d52315c9-2206-45bd-a488-e33b1b963e1b.png)

![vgg3da_loss](https://user-images.githubusercontent.com/62815174/232404923-4eefe50f-ab37-412b-a96b-918ec6096e7c.png)

### VGG 16

![vgg16_accuracy](https://user-images.githubusercontent.com/62815174/232405044-b210ec92-0052-43a7-bd18-bfceaefe226b.png)

![vgg16_loss](https://user-images.githubusercontent.com/62815174/232405062-0f894ae8-c48c-4948-8ccf-9531b8d876e5.png)

### MLP Model

![mlp_accuracy](https://user-images.githubusercontent.com/62815174/232411706-ae1759db-ff55-43b5-8f40-d14f804ba9cf.png)

![mlp_loss](https://user-images.githubusercontent.com/62815174/232411738-9b91dc43-0fcd-4b96-92bd-8ffa1cc7d8df.png)
