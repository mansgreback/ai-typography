# Creating Typefaces through Machine Learning

Author: Måns Grebäck, www.mansgreback.com
 
## Abstract
Using artificial intelligence, original typefaces can be created and existing lettering can be complemented. 

The process described here demonstrates this functionality through training a model on a range of categories, <i>tokens</i>. By feeding the model a text and/or image prompt, it generates never-before-seen typefaces.

The text prompt will depend on what categories it is trained on. It can be `font`, `serif`, `handwriting` or more specific combinations of tokens. 

The image prompt can be employed as an indicative starting point or, through inpainting technique to define which characters should be referenced and which should be generated, to substitute characters or complement typefaces. 

The features of this process include category combination, complementation, new typeface generation and stylizing input typeface.

Using the features, the technology can be used as a starting point for creating new fonts and a rough background sketch for ideas in a specified category. It can serve as a quick way to add missing characters to a font. As the AI follows style, it can be utilized as an indicator of how to proceed in type design. 
It is powerful as an aid for combining two distinct typographic categories, which in turn can be used as both inspiration and testing out "unlikely" combinations. 

There is even greater potential with human post-production, e. g. cherry-picking and merging results, vectorization, manual re-drawing of incorrect/undesired characters, and font composition for broader dynamic usage.

Some challenges are data collection, eventual auto-captioning of input and training images, as well as balancing the weights to avoid undesired predominancy. 

The main limitation of the process is that creativity rarely moves outside of the scope of the trained data, requiring either a large dataset or specialized models. A future possibility is to create new sub-styles of an input font, for example making an italic version from a serif font, maintaining characteristics. Another improvement would be to build a vector counterpart. 

<b>Figure 1.</b> Stylizing input image. 
![Stylizing input image, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/208267214-bbdd6846-d725-44b7-90cc-63944247f90d.png)

<b>Figure 2.</b> Complementing typeface. 
![Stylizing input image, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/208267208-4948bf7e-26d9-467d-9125-ef480341e9d8.png)

## Abbreviations and keywords
Machine Learning (ML), Artificial Intelligence (AI), <i>Stable Diffusion</i> (SD)[^1], Graphics, <i>Dreambooth</i> (DB)[^2], Legibility, Computer Vision and Pattern Recognition, Typography, Design

## Subjects
Computer Vision and Pattern Recognition (cs.CV); Graphics (cs.GR); Machine Learning (cs.LG)

## Introduction

As a full-time typeface designer over the last decade, I have always been intrigued by the advances of technology in typeface design and font creation, as well as questions about what constitutes a qualitative typeface, what makes a typeface original and how to automate/streamline typeface production. 

In recent years, I have developed an interest for the new technology of artificial intelligence, and something that has sparked my interest in particular is how I can integrate it into the creation process of my work. I began experimentation in Disco Diffusion[^3], which consisted primarily of stylizing type. With the release of Stable Diffusion[^1] in August 2022, I was able to conduct further experiments, approaching the subject of typography and AI from different angles. 

Upon learning how to train my own models and using all available resources, I eventually managed to develop a process where I successfully could use the technology to generate new types and to complement a limited character set.

## Purpose

The goal of the process development is to use the power of computation and AI to make type design more innovative and efficient. With its help, the hope is to be able to invent new styles and compliment styles with as little inference as possible. The optimal outcome is a process that follows typographic rules, such as weight and characteristics, without being stale and repetitive.

## Method

   1. <a href="#data-collection">Data collection</a>
      1. <a href="#sourcing">Selection and sourcing of material</a>
      2. <a href="#homogenization">Homogenization</a>
         1. <a href="#glyph">Glyph selection</a>
         2. <a href="#image">Image rendering</a>
      3. <a href="#captioning">Captioning</a>
   2. <a href="#training-1">Training</a>
      1. <a href="#training-setup">Setup</a>
      2. <a href="#training-2">Training</a>
   3. <a href="#generation">Generation</a>
      1. <a href="#generation-setup">Setup</a>
      2. <a href="#generating">Generating</a>
         1. <a href="#text-prompt">Text prompt</a>
            - <a href="#simple">Simple style</a>
            - <a href="#combining">Combining styles</a>
         2. <a href="#image-prompt">Image prompt</a>
            - <a href="#initial-image">Initial image</a>
            - <a href="#inpainting">Inpainting</a>
      3. <a href="#selection">Selection</a>
   
### Diagram
<b>Figure 3.</b> Diagram over process to create typefaces through machine learning. 
![Diagram over creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/206999698-6b8ac3d2-8745-4d8d-896d-7947b59476b4.png)

### <a id="data-collection" href="#data-collection">Data collection</a>

The data collection and creation is the preparation of the source material upon which the model will be built. It serves as the reference, and what is included here will be the "knowledge" that the model possesses. It consists of sourcing and preparing material, converting it into a readable resource--in this case, images with accompanying captions--that will be used for the training.
#### <a id="sourcing" href="#sourcing">Selection and sourcing of material</a>
As a first step, I selected a few distinct categories: `blackletter`, `boldscript`, `brushscript`, `handwriting`, `scriptwriting`, `serif`, `tallsans`. 
I selected 10 fonts for each one of the categories, making sure that each font fitted into one category only, in order to avoid overlapping between categories. In addition to being of the correct category, my requirements were that the fonts were licensed to be used for any public purpose and that they contained a wide character set. I listed these fonts in a spreadsheet with name and direct link to the OTF file. 
#### <a id="regularization" href="#homogenization">Homogenization</a>
The theory behind homogenization is to assist the AI to understand which symbol responds to which symbol across typefaces. It is done by selecting what glyphs to train on and building training images. 
##### <a id="glyph" href="#glyph">Glyph selection</a>
There are two main approaches to train the AI on fonts; to include as many characters as possible, or to make a very specific selection and then manually create the remaining characters. In the latter approach, one could for example create `ABEGHJK...` instead of `ABCDEFGHIJKL...`, arguing that the missing characters could easily be drawn. For example, `E` already contains `F` and `L`, `G` contains a `C`, `W` contains `V` etc. 

I chose, however, to go with the first approach, since I have found the variation to be significant enough. Keeping all letters creates a desired variation while also providing a vaster training set. 

In addition to all uppercase letters, `A-Z`, I included all lowercase letters, `a-z`, numbers, `0-9` and symbols `{ $ € £ @ % & ! ? - )`. 
##### <a id="image" href="#image">Image rendering</a>
Using SD 1.5[^1] and corresponding DB[^2] repository[^4], my required image format was 512×512 pixels. 
With the purpose of leaving as little whitespace as possible, I divided the characters to seven rows, as such:
```
ABCDEFGHI
JKLMNOPOR
STUVWXYZ
abcdefghijklm
nopqrstuvwxyz
1234567890
{$€£@%&!?-)
```
I built a page in HTML which, by obtaining a parameter `fontname` in the URL, used CSS to load a font, which in turn was provided from the spreadsheet where I had listed my selected training fonts. I then saved each individual page as a PNG image. 

The images were automatically cropped and resized using scripting in Automator[^5]. 

<b>Figure 4.</b> Training set example. 
![Training set example, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207095396-29a736c3-a1db-4760-bdd8-a616a687bacd.png)


#### <a id="captioning" href="#captioning">Captioning</a>

How complex the keywords (<i>tokens</i>) can be is determined by how many images the model is trained on, how well the selection has been made and is also dependent on the logic of the caption. An incorrect or contradictory token may eradicate the effectiveness across categories. 
Captioning could be done by adding only the main style, or theoretically by including more characteristics of that individual type. 
It could be as simple as `serif`, or as complex as `serif, roman, bold, italic, extra-wide, high-contrast`, using any styles or keywords. 

I chose to go with fewer training images, which does not allow for a very complex keyword network. Instead, I simply set a single token. As a prefix, I set <i>class</i> `font`. My full caption would consequently be for example `font brushscript` for one of the images depicting a font categorized as `brushscript`, `font serif` for any of the images categorized as `serif` etc.

The images were named as `{number}@font {category}.png`, for example `00005@font scriptwriting.png`, according to the repository's caption function[^8]. 

### <a id="training-1" href="#training-1">Training</a>
#### <a id="training-setup" href="#training-setup">Setup</a>
Training was done on a rented Nvidia GeForce RTX 3090 GPU through a Vast.ai[^9] Jupyter notebook. 

After renting and starting up the machine, I opened and cloned the trainer repository[^4]. 

I selected Stable Diffusion 1.5[^1] as my base model. 

In file `v1-finetune_unfrozen.yaml` I set image logging, `batch_frequency`, to `200` in order to be able to follow the progress during training. 
I set the learning rate, `model_lr`, to `1.0e-7`, with the purpose of doing a slow, soft and precise training, instead of a quick and rough one.  
I also switched off image flipping, `flip_p=0.0`, in the file personalized.py, which only works for symmetric training subjects. 
Total steps were set to `14000` which corresponds to two epochs: `10 images * 7 categories * 100 repeats = 7000 steps`

#### <a id="training-2" href="#training-2">Training</a>
With a speed of training around 1.3 seconds per iteration, the training took just over five hours. 
During this time I monitored the training by following the progress and checking the sample images as they were generated, confirming that it was picking up desired characteristics. As expected, the first images had only a slight resemblance to the training images, while the last ones were almost indistinguishable. 
After the training finished, the model was pruned and saved. 

### <a id="generation" href="#generation">Generation</a>
To get an actual output, the model is run through a machine (local or dedicated) with an eventual user interface. 
The input prompt can consist of text or image, and several techniques and tricks can be employed. Examples include initial image, inpainting, innovative text prompting, prompt merging and mixing, as well as manual selection and editing. 
Combining such techniques, there are multiple ways to manipulate the model to get a desired output. 

#### <a id="generation-setup" href="#generation-setup">Setup</a>
I used Google Colab[^6] to load the model and run AUTOMATIC1111's user interface[^7]. 

#### <a id="generating" href="#generating">Generating</a>
As input, the experiments consist of text and image prompts. 

##### <a id="text-prompt" href="#text-prompt">Text prompt</a>
A text prompt is an input that the AI converts to tokens and consequently uses as guidance when generating the image. 

###### <a id="simple" href="#simple">Simple category</a>
Perhaps the primary and most scaled back way to use the model is simple text prompting. 
For example, prompting for `font` will give an output of what the AI associated with the training images. In the case of this trained model, that results in a serif typeface. This happens because the serif category is predominant in the model. More variance with a simple `font` prompt could probably be achieved by balancing the weights during training. 

In my approach, however, a category was required for variance. The prompt to generate outputs from my trained categories were `font blackletter`, `font boldscript`, `font brushscript`, `font handwriting`, `font scriptwriting`, `font serif`, `font tallsans`. The class `font` could be omitted, but maintaining it is helpful to guide the AI though latent space, especially in cases where categories are more ambiguous.

In addition to this, characteristics that are <i>not</i> included in the training can also be prompted for, with varied results, since the model is trained on SD 1.5[^1] which already contains millions of tokens. Hypothetical examples are `font decorative`, `font blue` and `font graffiti`. 


Settings:
```
Steps: 50, Sampler: Euler a, CFG scale: 7, Size: 512x512
```
<b>Figure 5.</b> The trained categories, outputs.
![Trained categories, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207600087-635b4b85-c4af-44f2-97df-63a326b296b2.png)

###### <a id="combining" href="#combining">Combining categories</a>
Similarly, the AI can interpret a combination of tokens, such as `font sans boldscript` or `font handwriting blackletter`.

An example like `font handwriting serif` resulted in a predominance from the serif category, since no extensive weighting of the tokens has been done. 

With the help of the user interface[^7], we have the ability to steer the outputs in desired direction with `font (handwriting) [serif]` in order to increase (parentheses) and decrease (brackets) importance of a token. 

The UI also allows for more complex merging of tokens, such as alternating between categories or change a category after a certain step, with prompts such as `font [blackletter|script]` or `font [sans|handwriting|0.25]`, among others. These features can also be combined, creating very complex prompt recipes. 

```
Steps: 20, Sampler: Euler a, CFG scale: 7, Size: 512x512
```
<b>Figure 6.</b> Category combination.
![Combining categories, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207604525-0ca4fc77-edc8-4639-84a2-3e656b38a9fa.png)

##### <a id="image-prompt" href="#image-prompt">Image prompt</a>
An image prompt is a bitmap graphic that is fed into the AI and used as a base to continue generating from. 

An image prompt will most of the time require a text prompt as well, in the same format as above. 
The text prompt, in that case, can be contradictory to the style of the image, or attempt to be corresponding to the style.

###### <a id="initial-image" href="#initial-image">Initial image</a>

I used the initial image technique (<i>init</i> image) to turn a font style into a new one, for example feeding an image of the typeface <i>Times New Roman Regular</i> as initial image, but text prompting `font handwriting`. 

In addition to the initial image, a <i>denoising strength</i> number needs to be provided. This number determines how similar to the input image the output image will be. Using a number of `0.0` would render an output exactly the same as the initial image, and `1.0` would be without any resemblance whatsoever. A number between `0.1`-`0.9` is the most likely to be useful. 

Settings:
``` 
Steps: 20, Sampler: Euler a, CFG scale: 7, Size: 512x512
```

<b>Figure 7.</b> Stylizing initial image. 
![Initial image, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207602073-91c9c508-1dc8-4139-8287-3ac4dbed07ee.png)

###### <a id="inpainting" href="#inpainting">Inpainting</a>

Using the inpainting technique, I masked part of an image, and fed the remaining part as image prompt. For style, I tried to match the appearance of the image as closely as possible with my trained styles, `font` + category/categories.    

<b>Figure 8.</b> Inpainting to generate new characters. 
![Inpainting to create to characters, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207598064-f6639a1e-7cd8-45ce-b8ae-72e27806f02a.png)

I performed a test with multiple styles combinations to see which one gave the desired result. 

Settings: 
```
Steps: 30, Sampler: Euler a, CFG scale: 7, Size: 512x512, Denoising strength: 1, Mask blur: 0
```

<b>Figure 9.</b> Prompting experiments. `font` + token. Bold serif style. 
![Prompting experiments of bold serif typeface, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207582944-f89bb595-79b2-44f9-af3a-0aa6aee604b3.jpg)

<b>Figure 10.</b> Prompting experiments. `font` + token. Brush handwriting style. 
![Prompting experiments of brush typeface, creating typefaces through machine learning](https://user-images.githubusercontent.com/56417185/207585482-8a903bbc-6371-4ede-8e4b-ef58fc97a593.png)

#### <a id="selection" href="#selection">Selection</a>

As each generation uses a different seed, the results are always different. After generating multiple outputs, a crucial step is to select which images have potential to be used for one's purpose. 
Many outputs contain typographical errors to some extent, but there are also aesthetic factors to consider. 

Moreover, one may commit to more complex tasks such as combining the results of different outputs into an image of a complete typeface. Such an image could in turn be regularized through, for example, the initial image technique. 
Similarly, another way to post-process an output is to use inpainting on individual characters or details to replace or adjust them. 

## Result

My conceptual model was created with the intention of being able to make new styles within trained categories, with a certain level of creativity. 

I have found that the concept does work, and my objective of generating new styles has been achieved. 
Using my process, the artificial intelligence can create new typefaces from inputs: text and image. 

The trained categories can be merged and balanced against each other.  

I have also presented evidence of the flexibility of the model, being able to both complement fonts, as well as stylizing it or turn it into a typeface of a different category. 

In addition, the generated styles can vary greatly, which can be dependent on using available scripts to get a desired result, i. e. human interference, but also by simply using a different seed, i. e. low human interference. 

The potential usage is mainly as an assistant tool in typeface creation, including as a guide or rough base sketch on which to build a vector design. It can serve as a source of inspiration, or be used for font substitution or placeholder. 
It can also work to create alternates based on an already finished typeface. 

The limitation of the process is that it almost only works within and between trained categories. Similarly, the creativity rarely extends outside of the training data.  

Hence, one challenge is to make a larger model data set, in order to allow for even wider flexibility between typeface categories. This in turn brings a new challenge of training data collection: typefaces, captioning and image rendering. 

The resulting images are not of high resolution, and moreover, outputs are not always sharp, which makes the quality of the image suboptimal, often requiring post-processing to be practically useful. 

## Discussion
This approach of using a base model, training and category prompting has not been implemented earlier, which makes the process the first of its kind. 

The most exceptional results are when complementing a typeface that qualifies in a trained category, as it creates aesthetically pleasing results of decent quality. 

Unlike previous research[^12][^10], my technique manages to keep a consistency across the full typeface without any type of single glyph classification or selection. Other resources[^11] have used machine learning for other specific tasks of typeface creation, and not for the creation of an actual comprehensive typeface. 

The process is a proof of concept with great future potential. 

## Conclusion
An artificial intelligence can, through a process, learn typographical categories and consequently create new styles within and between the trained categories. 

## References

[^1]: Stable Diffusion: https://stability.ai/blog/stable-diffusion-public-release
[^2]: Dreambooth: https://arxiv.org/abs/2208.12242 [cs.CV]
[^3]: Disco Diffusion: https://github.com/alembics/disco-diffusion
[^4]: Joe Penna’s Dreambooth repository: https://github.com/JoePenna/Dreambooth-Stable-Diffusion
[^5]: Automator by Apple: https://support.apple.com/guide/automator/welcome/mac
[^6]: Google Colaboratory: https://research.google.com/colaboratory/faq.html
[^7]: AUTOMATIC1111's Stable Diffusion Web UI: https://github.com/AUTOMATIC1111/stable-diffusion-webui
[^8]: Caption functionality by Fabio Torchetti: https://github.com/mrwho
[^9]: Vast.ai: https://docs.vast.ai/vast.ai-overview/basics
[^10]: Machine Learning Font by NaN: http://www.machinelearningfont.com/
[^11]: Machine Learning of Fonts, Antanas Kascenas: https://project-archive.inf.ed.ac.uk/ug4/20170911/ug4_proj.pdf
[^12]: GlyphGAN: https://arxiv.org/pdf/1905.12502




