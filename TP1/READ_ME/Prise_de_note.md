# Note de cours
## Informations supplémentaires

#### Sites bac à sable:
````url
https://playground.tensorflow.org/
````

#### Flatten
Le flatten est contraint par la taille de l'image d'entrée, il est nécessaire que chaque image en entrée possède la meme taille sinon le changement du panel d'image d'un tableau 2D vers une seule ligne 1D ne sera pas effective
````python
torch.flatten
````

#### Séquences
Pour des séquences de lettres, lors de la predictions de la lettre suivante on lis la s"quence suivant des timestamps à chaque timestamos on regarde si on arrive a predire la lettre suivante, si ce n'est pas possible alors on attends le timestamps suivant pour pouvoir essayer de predire la lettre suivante (cf diapo .79)

#### Transformers
Aujourd'hui la methode utilisée dans les IA modernes (chatGPT, Claude, etc..)  est la notion de transformers`, les informations de chaque timestamps sont croisées afin de générer des informations fines et précises, cependant via cette méthode la position dans le temps de l'information est perdues, on rajoute donc des balises temporelles à chaque croisement afin de les positionner virtuellement.

### Points importants:
- quelle est la taches sur laquelle on veux travailler
- quelle est le type de la donnée d'entrée et comment la normaliser et/ou la standardiser
- quelle est la volumetrie de donnée 
- comment séparer et segmenter la donnée afin de créer des protocoles de test et d'entrainements utiles
- ajouter des métriques de tracking pour mesurer l'efficience et l'efficacité du réseau (controle de positif, faux et faux-positif), `calcul du F1-score et de la courbe ROC` (air sous la courbe metrique tres utiles pour l'evaluation de la qualité/précision du modèle)

### Bibliotheques
Structuration, traitement, gestion et stratification de la donnée
````python
import numpy as np
````
Traitement de tensor, génération de vecteur et matrice complexe
````python
import pytorch 
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4)
````

## Graphs 
### Spatial GNNs
Les réseaux neuronaux sur graphes spatiaux (GNN) sont des modèles spécialisés d'apprentissage profond qui traitent des données de graphes non euclidiens tout en intégrant explicitement des relations spatiales, des coordonnées physiques ou des réseaux topologiques.

## Diapo utiles
### Numéro 67 à 77
Convolutionnal layer, padding et filtre
### Numéro xx à xx

