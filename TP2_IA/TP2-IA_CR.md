# TP2 IA : Classification d'images médicales (diagnostic COVID)

## Étape 1 : Préparation du dataset

### 1. Compléter la classe `CovidDataset`

- Les arguments du constructeur sont `covid_txt` et `non_covid_txt` : les chemins vers les fichiers texte listant les images COVID et non COVID.
- Pour les non COVID, on procède comme pour les COVID, avec le label 0 :

```python
non_covid_data = [img_dir + "CT_NonCOVID/" + line.strip() for line in non_covid_lines]
non_covid_labels = [0 for _ in non_covid_lines]
```

- `__len__` renvoie le nombre d'images : `return len(self.data)`.
- `__getitem__` ouvre l'image à l'indice `idx`, la convertit en RGB, récupère son label et applique la transformation.

### 2. Définir `valset` et `testset`

```python
trainset = CovidDataset(img_dir + "trainCT_COVID.txt", img_dir + "trainCT_NonCOVID.txt", transform=transformer)
valset   = CovidDataset(img_dir + "valCT_COVID.txt",   img_dir + "valCT_NonCOVID.txt",   transform=transformer)
testset  = CovidDataset(img_dir + "testCT_COVID.txt",  img_dir + "testCT_NonCOVID.txt",  transform=transformer)
```

Résultat obtenu : **425 images d'entraînement, 118 de validation, 203 de test**.

### 3. Que fait `transformer` ?

Il applique trois opérations à chaque image :

- `Resize((256,256))` : toutes les images ont la même taille.
- `ToTensor()` : conversion en tenseur `[3, 256, 256]` avec des valeurs entre 0 et 1.
- `Normalize(mean, std)` : centrage et réduction de chaque canal RGB (valeurs d'ImageNet), ce qui facilite et stabilise l'entraînement.

---

## DataLoaders

### 1. Définir `val_dataloader` et `test_dataloader`

```python
val_dataloader  = DataLoader(valset,  batch_size=batch_size, drop_last=False, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False)
```

### 2. Faut-il mélanger les données de validation et de test ?

Non. L'évaluation ne modifie pas le modèle, donc l'ordre des images n'a aucune importance. Le shuffle ne sert qu'à l'entraînement : il évite que les batchs soient toujours identiques et limite l'apprentissage de l'ordre des données.

### 3. Que fait `drop_last` ?

Quand la taille du dataset n'est pas un multiple de `batch_size`, le dernier batch est plus petit. Avec `drop_last=True`, il est supprimé ; avec `False`, il est conservé.

Dans notre cas : 425 = 6 × 64 + 41. Avec `False`, on a 7 batchs dont le dernier de 41 images (visible dans l'affichage). Avec `True`, on n'aurait que 6 batchs et 41 images ne seraient pas utilisées à cette epoch.

---

## Étape 2 : Définition et apprentissage du modèle

### Définir `__init__` et `forward`

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 64 * 64, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
```

Taille de l'entrée du `Linear` : l'image fait 256×256, chaque pooling divise par 2 (256 → 128 → 64), et on a 64 canaux. On obtient donc 64 × 64 × 64 = 262 144 valeurs en entrée, pour 2 sorties.

### Fonctions `train` et `test`

- Dans `train`, on calcule la prédiction `pred = model(X)`, puis la loss `loss = loss_fn(pred, y)`, puis on rétropropage (`zero_grad`, `backward`, `step`).
- Dans `test`, on calcule seulement la prédiction et la loss, sous `torch.no_grad()`.

**Différence fondamentale :** `train` met à jour les poids du modèle (`loss.backward()` et `optimizer.step()`, avec `model.train()`), alors que `test` ne fait qu'évaluer : pas de gradient, pas de mise à jour, et `model.eval()`.

### Assemblage : entraînement du modèle

Paramètres utilisés : modèle `CNN`, optimiseur Adam (`lr = 1e-3`), loss `CrossEntropyLoss`, 10 epochs, entraînement sur CPU. Les poids de la meilleure epoch (loss de validation minimale) sont sauvegardés dans `best_model.pth`.

Évolution des losses :

| Epoch | Loss train | Loss validation |
|------:|-----------:|----------------:|
| 1  | 1.611 | 0.694 |
| 2  | 0.622 | 0.685 |
| 3  | 0.550 | 0.786 |
| 4  | 0.491 | 0.668 |
| 5  | 0.444 | **0.626** (meilleure) |
| 6  | 0.372 | 0.635 |
| 7  | 0.328 | 0.758 |
| 8  | 0.270 | 0.853 |
| 9  | 0.255 | 0.799 |
| 10 | 0.194 | 0.876 |

### Que observe-t-on sur les courbes ?

La loss d'entraînement diminue de façon continue (de 1.61 à 0.19), alors que la loss de validation passe par un minimum à l'epoch 5 (0.626) puis remonte jusqu'à 0.876 à l'epoch 10.

C'est du **surapprentissage (overfitting)** : à partir de l'epoch 5, le modèle mémorise les images d'entraînement au lieu d'apprendre des caractéristiques générales, et généralise de moins en moins bien. Cela s'explique par le faible nombre d'images (425) par rapport au nombre de paramètres du modèle (la couche `Linear` à elle seule en a plus de 500 000). Il ne faut donc pas ajouter d'epochs. La sauvegarde du meilleur modèle (epoch 5) permet de garder le meilleur point.

---

## Étape 3 : Évaluation sur le jeu de test

### Explication du code

- `load_state_dict(torch.load("best_model.pth"))` recharge les poids de la meilleure epoch, et `model.eval()` passe en mode évaluation.
- `torch.no_grad()` désactive le calcul des gradients, car on n'entraîne plus.
- Pour chaque batch, `argmax(dim=1)` garde la classe au score le plus élevé (0 = non COVID, 1 = COVID).
- Les 4 compteurs s'obtiennent en comparant prédictions et vrais labels :
  - **TP** : prédit COVID, réellement COVID
  - **TN** : prédit non COVID, réellement non COVID
  - **FP** : prédit COVID, mais sain
  - **FN** : prédit non COVID, mais malade

### Résultats

|  | Prédit COVID | Prédit non COVID |
|---|---:|---:|
| **Réellement COVID** | TP = 62 | FN = 36 |
| **Réellement non COVID** | FP = 32 | TN = 73 |

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.665 |
| Précision | 0.660 |
| Recall (rappel) | 0.633 |
| F1-score | 0.646 |

### Quelle métrique est pertinente ?

Pour un diagnostic médical, la métrique la plus importante est le **recall** : il mesure la part des vrais malades détectés. Un faux négatif (patient COVID non détecté) est plus grave qu'un faux positif (fausse alerte, qui sera vérifiée par un autre examen). L'accuracy et le F1 complètent l'analyse, car les classes sont presque équilibrées (98 COVID et 105 non COVID dans le test).

### Interprétation

Le modèle détecte seulement 63 % des patients COVID (36 sur 98 sont ratés), pour une accuracy de 66,5 %. Ces résultats sont modestes et insuffisants pour un usage médical réel, ce qui est cohérent avec le surapprentissage observé.

---

## Étape 4 : Réduire le surapprentissage



###  2 couches convolutives

```python
class CNN2(nn.Module):
    ''''
    passage de 4 à 2 couches convolutives afin de réduire le surapprentissage
    '''''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 64 * 64, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
```


###  Dropout après chaque couche convolutive

```python
class CNNDrop(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(64 * 64 * 64, 2)

    def forward(self, x):
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(self.relu(self.conv3(x)))
        x = self.drop(self.relu(self.conv4(x)))
        x = self.pool(x)
        return self.fc(x.flatten(start_dim=1))
```

Le dropout met aléatoirement une partie des activations à 0 pendant l'entraînement, ce qui empêche le modèle de se reposer sur quelques neurones précis et l'aide à mieux généraliser.


### Conclusion

Le modèle initial (4 couches) surapprend : la loss de validation remonte dès l'epoch 5 alors que celle d'entraînement continue de baisser. Les deux stratégies (réduire le nombre de couches, ajouter du dropout) visent à réduire cet écart. La meilleure est celle qui obtient la loss de validation la plus basse et le meilleur recall sur le test. 
