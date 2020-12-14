# Seq2seq-text-Dream

DeepDream: https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

## Description
DeepDream is a technique to generate image by applying gradient from image classification model. My concern is whether it is applicable to text(which is known for having descrete data structure, making hard to apply gradient step). I applied DeepDream to Seq2Seq De-En translator model, and checked the performance of the DeepDream.

### Customized Embedding 
Since the purpose of torch.Embedding module is converting descrete index(integer) into vector(float), it's impossible to apply gradient step. So I Customized Embedding module to generate Sum of the vector of softmax output, which enables backpropagations for DeepDream

### Entropy Loss
Input is in the form of Integer, which means the softmax output(probablity) has low entropy. So I added entropy loss to imitate input distribution

### Pseudo Attention
I configured that attention alignment is not formed well for DeepDream. So I forced attention alignement to have desirable shape with the hope of better performance

## Result
### Dream
```
1. Strand hinteren alter Freundin indische Stehen Freundin Pendler
2. Schnee seltsames aussehendes Weihnachtsmann hinunterfährt spielen rotes Gerüst
3. Hintergrund Hand männliche gelber älteres Brüder buntes um
4. Matratze denen anhat gekleidet Gebäude Schnee University Flaschen
```
### Target
```
1. A man in an orange hat starring at something . 
2. A Boston Terrier is running on lush green grass in front of a white fence . 
3. A girl in karate uniform breaking a stick with a front kick . 
4. Five people wearing winter jackets and helmets stand in the snow , with <unk> in the background . 
```
I conclued that DeepDream doesn't work well for text, especially with Seq2Seq model. If there is any method or result to improve the text-dream experiment, please let me know.
