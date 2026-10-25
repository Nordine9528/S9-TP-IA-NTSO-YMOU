## 1: Data analysis

### Q1 : According to the code above, deduce :
- the task(s) associated with this dataset
- the size of the images 
- the number of channels for each image

Chaque image possède une taille de 28x28, avec 1 chaîne pour chaque images 

### Q2 : explain what does what kind of transformations *data_transform* perform.
transforms.Compose applique trois transformations, dans l'ordre :

  transforms.ToTensor().
  Convertit l'image PIL (pixels entiers de 0 à 255) en tenseur PyTorch de type float, avec les valeurs mises à l'échelle dans [0, 1].  La forme devient [C, H, W] = [1, 28, 28].

  transforms.Normalize(mean=[.5], std=[.5])
  Applique (x - 0.5) / 0.5 à chaque pixel, ce qui recentre les valeurs dans [-1, 1]. 

  Les données centrées autour de 0 stabilisent et accélèrent généralement l'entraînement.

  transforms.Lambda(torch.flatten)
  Applique une fonction personnalisée : torch.flatten aplatit le tenseur [1, 28, 28] en un vecteur de 784 valeurs. C'est nécessaire si le modèle est un réseau entièrement connecté (MLP), qui attend un vecteur en entrée et non une image 2D. Un CNN, lui, aurait besoin de garder la forme [1, 28, 28].

  Image PIL 28×28 → tenseur [1, 28, 28] dans [0, 1] → valeurs dans [-1, 1] → vecteur de taille 784.

## 2: Preparing the data

The code below is used to prepare the DataLoader, which will be used to pass the images through the neural network.

````python
batch_size = 64

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, drop_last=False, shuffle=True)
val_dataloader = DataLoader(dataset_valid, batch_size=batch_size, drop_last=False, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=False)

for X, y in train_dataloader:
    print(f"Shape of X [N, C]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
````
### Q3 : What do the variables N and C correspond to? 

Dans [N, C], N est le nombre d'images dans le batch et C le nombre de canaux (channels) de chaque image.

X.shape contient 4 variables, H ; W co: [N, C, H, W].

N : taille du batch, donc 64 ici (batch_size). Le dernier batch peut être plus petit, puisque drop_last=False garde les images restantes.
C : nombre de canaux. Pour PneumoniaMNIST, les radios sont en niveaux de gris, donc C = 1 (une image RGB aurait C = 3)

## 3 : Network development and learning

#### Neural networks: 

Now that our data are ready to be processed, we now have to define our deep learning model. For the moment, only one type of neural network is known: the multi-layer perceptron (MLP), which is made up of linear layers followed by activation functions. 

First step consists in defining a new class that inherit from the torch.nn.Module class. For this, we have to define 2 methods : the \_\_init__ method and the forward method. The \_\_init__ method is used to instantiate the different layers of our model (convolutional, activation function or pooling layer for example). In PyTorch, activation function are considered as layers, but in general, it is the combination of activation function with parameteizable linear function that are considered as layers. On the other hand, the forward method use the previously defined layer to compute our model.  

Define both \_\_init__ and forward methods according to the following requirements : 
1. We want 2 Linear layers
2. First Linear layer is followed by a ReLU activation. 
3. Second Linear layer is followed by a Softmax activation
4. First Linear layer is made of 128 hidden units 
5. Second Linear layer is used for decision. Deduce the number of hidden units according to this.

To help you, here are the different function you will need to use : [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html),[Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear), [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#softmax)
        
__A little word about functions named with double underscore as prefixe and suffixe :__ you may have noticed that the functions we defined have a double underscore before and after their name. These methods are called _dunder_ (for double underscore) methods or _magic_ methods. In Python, dunder methods are methods that allow instances of a class to interact with the built-in functions and operators of the language. They are never directly called by the programmer, which is why they are called _magic_. If you are familiar with Java, it is a bit similar to the method toString(). This is also what allow the '+' operator to concatenate two lists, through the dunder method \_\_add__.

__Training and validation loop__

Now, we have to define our training and testing process. For this, we decide here to define 2 functions train and test, but this could be done any other way. 

On one side, the train function is designed to optimize the parameters of the model according to the training data and a loss function. For each batch of data, this function compute the prediction of the model, compute a loss by comparing the prediction and the ground truth, and backpropagate this loss.

On the other side, the test function is only here to evaluate our model through a loss function. 

1. Complete the train loop. 
2. Complete the test loop. 

### Q4 : What is the fundamental difference between those 2 functions ?
```python
class TinyMLP(nn.Module):
    def __init__(self):
        super(TinyMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # 28*28 = 784 entrées -> 128 unités cachées
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)     # 2 classes en sortie
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
```
```train``` met à jour les poids du modèle (rétropropagation + optimizer.step()), alors que test ne fait que les évaluer sans jamais les modifier.
  
__Plotting the training__

By using the matplotlib library, plot both losses (valid and train). 

### Q5 : Explain and analyse what you observe on the curve ?

Les courbes de loss d'entraînement (bleu) et de validation (orange) décroissent rapidement durant les 10 premières epochs, ce qui montre que le modèle apprend. Après environ 15 epochs, les deux courbes se séparent : la loss d'entraînement continue de diminuer lentement (jusqu'à environ 0,33 à l'epoch 100), tandis que la loss de validation se stabilise autour de 0,355 sans plus s'améliorer. On observe donc un léger surapprentissage : le modèle ajuste de mieux en mieux les données d'entraînement, mais cela ne se traduit plus par un gain en généralisation. L'écart reste faible (environ 0,02) et la loss de validation ne remonte pas, donc le surapprentissage est modéré. Un arrêt plus tôt autour de l'epoch 20-30 aurait donné des performances de validation similaire.
  
